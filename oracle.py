"""
Project Panopticon — Phase 6: The Glasshouse (server half, reality in)
=======================================================================
oracle.py

The Environmental Oracle: the one-way valve through which the real world
leaks into the simulation. It ingests external event streams (market
tickers, weather, news RSS — anything reducible to "something happened,
possibly somewhere") and converts them into perception tasks on the
MLFQScheduler, so agents visibly react to reality.

INJECTION SEMANTICS
  * GLOBAL events ("market crash") — every agent receives an INTERACTIVE
    (Q0) dialogue task simultaneously: the town-crier moment. The AIMD
    limiter + graceful degradation absorb the thundering herd; the Phase 5
    crucible is precisely the proof that this is safe at 50 agents.
  * LOCAL events ("lightning strike at (45,12) radius 10") — only agents
    within the STRICT Euclidean radius (d² ≤ r², boundary inclusive) are
    tasked. Distance is measured against live World positions at injection
    time.

RADIXATTENTION-SAFE BY CONSTRUCTION
  Reality is ephemeral; the cache tree is forever. An injected event NEVER
  touches the system message — SHARED_WORLD_PREFIX ships byte-identical on
  every task — and the event text rides exclusively in the dynamic user
  leaf. A market crash therefore costs 50 leaf prefills against a warm
  shared root, not 50 full-tree rebuilds. (Events that should become
  standing world knowledge belong in the slow-rotating segment-[2] globals
  per the prefix contract — an explicit operator action, not an Oracle
  side effect.)

FAIL-OPEN
  Feed outages, malformed events, scheduler pressure: counted, logged,
  skipped. The Oracle can die outright and the simulation keeps running on
  organic triggers alone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

import aiohttp

from tick_engine import AgentTask, QueueLevel, SHARED_WORLD_PREFIX

log = logging.getLogger("panopticon.oracle")


# ============================================================================
# 1. Event model + feed sources
# ============================================================================

@dataclass(frozen=True)
class WorldEvent:
    """A validated, injectable event. `scope` is 'global' or 'local';
    local events carry a position and radius in world units."""
    scope: str
    text: str
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0

    @staticmethod
    def parse(raw: dict[str, Any]) -> "WorldEvent | None":
        """Strict validation of untrusted feed rows → None on any violation
        (the caller counts and skips; fail-open)."""
        scope = raw.get("scope")
        text = raw.get("text")
        if scope not in ("global", "local"):
            return None
        if not isinstance(text, str) or not text.strip():
            return None
        text = text.strip()[:280]                    # leaf hygiene: bounded
        if scope == "global":
            return WorldEvent(scope="global", text=text)
        try:
            x, y = float(raw["x"]), float(raw["y"])
            radius = float(raw["radius"])
        except (KeyError, TypeError, ValueError):
            return None
        if radius <= 0 or not all(map(_finite, (x, y, radius))):
            return None
        return WorldEvent(scope="local", text=text, x=x, y=y, radius=radius)


def _finite(v: float) -> bool:
    return v == v and abs(v) != float("inf")


class FeedSource(Protocol):
    """Anything that yields raw event dicts forever."""
    def events(self) -> AsyncIterator[dict[str, Any]]: ...


class JsonPollFeed:
    """Polls an HTTP endpoint returning {"events": [...]} — the mock-friendly
    stand-in for RSS/market adapters (each real adapter reduces to this same
    shape). New events are deduplicated by a monotonically consumed cursor
    supplied by the feed ("id"), so restarts don't replay history."""

    def __init__(self, url: str, *, poll_interval_s: float = 5.0) -> None:
        self._url = url
        self._poll_s = poll_interval_s
        self._cursor: Any = None
        self._session: aiohttp.ClientSession | None = None
        self.poll_failures = 0

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        self._session = self._session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10))
        backoff = self._poll_s
        while True:
            try:
                params = {} if self._cursor is None else {"after": self._cursor}
                async with self._session.get(self._url, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                for raw in data.get("events", []):
                    if "id" in raw:
                        self._cursor = raw["id"]
                    yield raw
                backoff = self._poll_s                      # healthy: reset
            except asyncio.CancelledError:
                raise
            except Exception:
                self.poll_failures += 1
                backoff = min(backoff * 2, 60.0)            # fail-open + retreat
                log.warning("Feed poll failed (x%d) — backing off %.0fs",
                            self.poll_failures, backoff)
            await asyncio.sleep(backoff)


class StaticFeed:
    """Test/replay source: yields a fixed list once, then idles forever."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        for row in self._rows:
            yield row
        await asyncio.Event().wait()


# ============================================================================
# 2. The Oracle
# ============================================================================

@dataclass(frozen=True)
class OracleConfig:
    min_global_interval_s: float = 10.0   # town criers need breath between
    min_local_interval_s: float = 1.0     # announcements; locals are cheaper
    reset_reasoning_cooldown: bool = True # injected perception counts as the
                                          # agent's reasoning turn (prevents an
                                          # immediate organic double-fire)


class Oracle:
    def __init__(
        self,
        feed: FeedSource,
        scheduler: Any,          # MLFQScheduler
        world: Any,              # World (live positions for radius scoping)
        cfg: OracleConfig | None = None,
    ) -> None:
        self._feed = feed
        self._scheduler = scheduler
        self._world = world
        self._cfg = cfg or OracleConfig()
        self._last_global = 0.0
        self._last_local = 0.0
        # Telemetry
        self.events_ingested = 0
        self.events_malformed = 0
        self.events_rate_limited = 0
        self.global_injections = 0
        self.local_injections = 0
        self.agents_notified = 0

    # ------------------------------------------------------------------ run

    async def run(self) -> None:
        """Supervised consumption loop. Feed exceptions are the feed's
        problem (JsonPollFeed already retries); anything that escapes is
        absorbed here — the Oracle never takes the engine down."""
        log.info("Oracle up: feed=%s", type(self._feed).__name__)
        while True:
            try:
                async for raw in self._feed.events():
                    self.events_ingested += 1
                    event = WorldEvent.parse(raw)
                    if event is None:
                        self.events_malformed += 1
                        log.warning("Malformed feed event skipped: %.120r", raw)
                        continue
                    self.inject(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Oracle feed loop crashed — restarting in 5s "
                              "(fail-open)")
                await asyncio.sleep(5.0)

    # --------------------------------------------------------------- inject

    def inject(self, event: WorldEvent) -> int:
        """Synchronous, allocation-light, callable from anywhere (tests
        drive it directly). Returns the number of agents tasked."""
        now = time.monotonic()
        if event.scope == "global":
            if now - self._last_global < self._cfg.min_global_interval_s:
                self.events_rate_limited += 1
                return 0
            self._last_global = now
            targets = list(self._world.agents)
            leaf = (f"A town crier shouts news that reaches everyone: "
                    f"{event.text} React.")
            self.global_injections += 1
        else:
            if now - self._last_local < self._cfg.min_local_interval_s:
                self.events_rate_limited += 1
                return 0
            self._last_local = now
            r2 = event.radius * event.radius
            targets = [a for a in self._world.agents
                       if (a.x - event.x) ** 2 + (a.y - event.y) ** 2 <= r2]
            leaf = (f"Something happens right where you stand: "
                    f"{event.text} React.")
            self.local_injections += 1

        for agent in targets:
            self._scheduler.submit_nowait(AgentTask(
                agent_id=agent.id,
                kind="dialogue",
                level=QueueLevel.INTERACTIVE,
                messages=[
                    # BYTE-STABLE root — never event-contaminated.
                    {"role": "system", "content": SHARED_WORLD_PREFIX},
                    # Ephemeral leaf — the only place reality touches.
                    {"role": "user", "content": leaf},
                ],
            ))
            if self._cfg.reset_reasoning_cooldown:
                agent.last_reasoned_at = now
        self.agents_notified += len(targets)
        log.info("Injected %s event to %d agent(s): %.60r",
                 event.scope, len(targets), event.text)
        return len(targets)
