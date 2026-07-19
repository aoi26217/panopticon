"""
Project Panopticon — Phase 10: The Narrative Director
======================================================
director.py

An omniscient observer with two narrow systemic levers and no voice of its
own. The Director never speaks to agents, never expands the action schema,
never touches an LLM: it reads engine telemetry and adjusts the WORLD, then
lets fifty goal-seeking minds discover what changed. Emergence through
levers, not scripts.

SCHEDULING (stated deviation): the spec asks for an "ultra-low-priority
queue," but the Director performs zero LLM generations — an MLFQ lane would
govern nothing. It runs instead as the lowest-frequency supervised daemon in
the system, and defers its one potentially-costly lever (prefix rotation)
whenever the AIMD limiter shows congestion. Lowest priority, realized as
politeness rather than queueing.

LEVER A — ECONOMY (spawn scarcity)
  If physical contention (delta of intent_stats.contested_losses per cycle)
  runs below threshold, the plaza has gone slack: nobody wants the same
  thing. The Director finds the densest agent cluster (O(n) cell binning)
  and spawns a high-value relic at its centroid. The causal chain is
  entirely pre-built machinery: the Q3 GoalSeek daemon sees the relic in
  "nearby objects" → goals form → move_towards/pick_up fire → the intent
  board arbitrates → contested_losses rises → the Director goes quiet.
  A closed-loop scarcity thermostat.

LEVER B — ENVIRONMENT (rotate world conditions)
  The weather line is RadixAttention SEGMENT [2]: a short, slow-rotating
  suffix appended AFTER the immutable [1] root inside the system message.
  Cache-invalidation math: RadixAttention shares cached KV up to the first
  divergent token, so rotating [2] preserves the entire [1] prefix (~200
  tokens, the expensive part) and re-prefills only the ~12-token conditions
  tail plus downstream — per-request cost measured in single-digit
  milliseconds, NOT a tree rebuild. Three amortization guards on top:
    1. ATOMIC swap: one string assignment on the shared event loop; every
       task built after it uses the new epoch, none sees a torn prefix.
    2. CONGESTION GATE: rotation defers while aimd.active ≥ 80% of the
       ceiling, so the (small) re-prefill wave never lands on a saturated
       server and never manufactures an AIMD backoff.
    3. MIN INTERVAL: epochs live ≥ min_rotation_interval_s, so each
       rotation's cost amortizes over hundreds of warm-branch requests.

Fail-open as always: a dead Director changes nothing — the plaza simply
stops receiving providence.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("panopticon.director")

_relic_ids = itertools.count(1)

DEFAULT_WEATHER_CYCLE = (
    "a bright, dry afternoon; the plaza stones are warm",
    "grey clouds massing; vendors eye the sky and hurry",
    "steady rain; awnings drum, the crowd thins to the arcades",
    "the storm has passed; puddles mirror the lamplight",
)


@dataclass(frozen=True)
class DirectorConfig:
    interval_s: float = 15.0
    # Lever A
    low_contention_threshold: int = 1     # contested losses / cycle below this
    cluster_cell: float = 6.0             # binning cell for density scan
    min_cluster_size: int = 4
    relic_cap: int = 3
    # Lever B
    min_rotation_interval_s: float = 120.0
    congestion_defer_ratio: float = 0.8
    weather_cycle: tuple[str, ...] = DEFAULT_WEATHER_CYCLE


class Director:
    def __init__(self, cfg: DirectorConfig | None = None) -> None:
        self._cfg = cfg or DirectorConfig()
        self._engine: Any = None
        self._world: Any = None
        self._limiter: Any = None
        self._dispatcher: Any = None
        self._last_contested = 0
        self._last_rotation = 0.0
        self._weather_i = 0
        # Telemetry
        self.cycles = 0
        self.relics_spawned = 0
        self.rotations = 0
        self.rotations_deferred = 0

    def bind(self, engine: Any, world: Any, limiter: Any,
             dispatcher: Any) -> None:
        self._engine = engine
        self._world = world
        self._limiter = limiter
        self._dispatcher = dispatcher
        self._last_contested = engine.intent_stats.contested_losses

    # ------------------------------------------------------------------ run

    async def run(self) -> None:
        log.info("Director up: interval=%.0fs (zero LLM calls; pure levers)",
                 self._cfg.interval_s)
        while True:
            await asyncio.sleep(self._cfg.interval_s)
            try:
                self.cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Director cycle failed — providence pauses "
                              "(fail-open)")

    def cycle(self) -> None:
        """One synchronous decision pass (tests drive it directly)."""
        self.cycles += 1
        self._lever_economy()
        self._lever_environment()

    # ------------------------------------------------------ Lever A: economy

    def _lever_economy(self) -> None:
        stats = self._engine.intent_stats
        delta = stats.contested_losses - self._last_contested
        self._last_contested = stats.contested_losses
        if delta >= self._cfg.low_contention_threshold:
            return                                # plenty of friction already
        relics = sum(1 for e in self._world.entities.all()
                     if e.kind == "relic")
        if relics >= self._cfg.relic_cap:
            return

        centroid = self._densest_cluster_centroid()
        if centroid is None:
            return
        cx, cy = centroid
        nav = getattr(self._world, "navigation", None)
        if nav is not None and nav.blocked(cx, cy):
            return                                # never bury a relic in a wall
        relic = self._world.entities.spawn(
            f"relic_{next(_relic_ids):03d}", "relic", cx, cy,
            state={"value": 100, "aura": "humming"})
        self.relics_spawned += 1
        log.info("Providence: %s manifests at (%.1f, %.1f) — %d agents "
                 "nearby, contention delta was %d",
                 relic.id, cx, cy, self._cfg.min_cluster_size, delta)

    def _densest_cluster_centroid(self) -> tuple[float, float] | None:
        """O(n) cell binning: densest cell wins; centroid of its members."""
        cell = self._cfg.cluster_cell
        bins: Counter[tuple[int, int]] = Counter()
        members: dict[tuple[int, int], list[Any]] = {}
        for a in self._world.agents:
            if a.is_human:
                continue
            key = (int(a.x / cell), int(a.y / cell))
            bins[key] += 1
            members.setdefault(key, []).append(a)
        if not bins:
            return None
        key, count = bins.most_common(1)[0]
        if count < self._cfg.min_cluster_size:
            return None
        agents = members[key]
        return (sum(a.x for a in agents) / len(agents),
                sum(a.y for a in agents) / len(agents))

    # -------------------------------------------------- Lever B: environment

    def _lever_environment(self) -> None:
        now = time.monotonic()
        if now - self._last_rotation < self._cfg.min_rotation_interval_s:
            return
        # Congestion gate: never land a re-prefill wave on a hot limiter.
        if self._limiter.active >= max(1, int(
                self._cfg.congestion_defer_ratio * self._limiter.limit)):
            self.rotations_deferred += 1
            return
        self._weather_i = (self._weather_i + 1) % len(self._cfg.weather_cycle)
        self.rotate_conditions(self._cfg.weather_cycle[self._weather_i])

    def rotate_conditions(self, text: str) -> None:
        """The atomic swap. One assignment on the shared event loop: every
        dialogue prompt built after this line carries the new segment [2];
        the [1] root stays byte-identical and fully cached."""
        self._dispatcher.world_conditions = text
        self._last_rotation = time.monotonic()
        self.rotations += 1
        log.info("Conditions rotate → %r (root cached; ~12-token re-prefill "
                 "per branch, amortized over the epoch)", text)
