"""
Project Panopticon — Phase 8: The Q3 Goal-Seeking Loop
=======================================================
goalseek.py

Intrinsic motivation. A slow daemon (sibling of the Phase 4 consolidator)
that periodically reads what an agent BELIEVES (Neo4j, written by
consolidation), what it CARRIES, and what stands NEARBY — and asks the LLM,
at the lowest scheduling priority in the system, what this agent should be
trying to accomplish. The answer lands on the dispatcher's goal board.

WHERE THE GOAL LIVES IN THE CACHE TREE (the load-bearing decision)
  NOT in EMBODIED_WORLD_PREFIX. A per-agent string in the shared system
  prefix would fork the RadixAttention root fifty ways — the exact failure
  mode the prefix contract exists to prevent. Instead the goal is RadixAtt
  segment [3], the per-agent branch: the dispatcher prepends
  "[Your standing personal goal: …]" as the FIRST bytes of every dialogue
  leaf. Byte-stable between goal updates, so it EXTENDS the agent's warm
  branch; when the daemon replaces a goal, exactly one branch re-prefills.
  Shared root: never touched. Bias: total — every Q0 reaction and every
  structured action decision is generated downstream of the goal tokens.

SCHEDULING
  Goal formulation rides Q3 (QueueLevel.GOAL), below even consolidation:
  strict-priority dequeue means a goal generation runs only when Q0/Q1/Q2
  are all empty. An agent mid-conversation never loses prefill to
  introspection. Zombie reaping, AIMD admission, and server-side abort come
  free via the dispatcher's result_future path, exactly like Phase 4.

FAIL-OPEN
  Bad JSON, timeouts, dead graph: the agent simply keeps its previous goal
  (or none). Purpose is an optimization; the plaza runs without it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from tick_engine import AgentTask, MLFQScheduler, QueueLevel

log = logging.getLogger("panopticon.goalseek")


# Byte-stable Q3 root — one warm branch for ALL goal formulations.
GOALSEEK_PREFIX = (
    "You are the motivation subsystem of a persistent town-square simulation. "
    "Given one agent's long-term beliefs, current inventory, and nearby "
    "objects, formulate ONE concrete, actionable short-term goal grounded in "
    "the world (obtain / deliver / unlock / confront / guard something "
    "specific).\n"
    "Respond with ONLY a JSON object: {\"goal\": \"<one imperative sentence "
    "naming specific entity or agent ids>\"} or {\"goal\": null} if nothing "
    "is worth pursuing.\n"
    "Rules: reference only ids present in the input; no meta-commentary."
)

GOAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {"anyOf": [{"type": "string", "maxLength": 200},
                           {"type": "null"}]},
    },
    "required": ["goal"],
    "additionalProperties": False,
}


class BeliefReader(Protocol):
    async def fetch_beliefs(self, agent_id: str, limit: int
                            ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class GoalSeekConfig:
    interval_s: float = 20.0          # one agent per interval, round-robin
    beliefs_limit: int = 5
    nearby_radius: float = 12.0
    llm_timeout_s: float = 60.0       # Q3 waits politely behind everything
    min_goal_age_s: float = 60.0      # don't thrash an agent's purpose


class GoalSeekDaemon:
    def __init__(
        self,
        graph: BeliefReader | None,
        scheduler: MLFQScheduler,
        world: Any,
        goal_board: dict[str, str],
        action_router: Any | None = None,     # for the goal JSON grammar
        cfg: GoalSeekConfig | None = None,
    ) -> None:
        self._graph = graph
        self._scheduler = scheduler
        self._world = world
        self._board = goal_board
        self._cfg = cfg or GoalSeekConfig()
        self._rr = 0
        self._goal_set_at: dict[str, float] = {}
        # Telemetry
        self.sweeps = 0
        self.sweep_failures = 0
        self.goals_set = 0
        self.goals_cleared = 0
        self.parse_failures = 0
        self.formulation_ms: deque[float] = deque(maxlen=100)

    # ------------------------------------------------------------------ run

    async def run(self) -> None:
        log.info("GoalSeek daemon up: interval=%.0fs (Q3, below everything)",
                 self._cfg.interval_s)
        while True:
            await asyncio.sleep(self._cfg.interval_s)
            agents = [a for a in self._world.agents if not a.is_human]
            if not agents:
                continue
            agent = agents[self._rr % len(agents)]
            self._rr += 1
            if (time.monotonic() - self._goal_set_at.get(agent.id, 0.0)
                    < self._cfg.min_goal_age_s):
                continue                       # purpose deserves persistence
            start = time.monotonic()
            try:
                await self.formulate(agent)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.sweep_failures += 1
                log.exception("Goal formulation failed for %s — keeps its "
                              "old goal (fail-open)", agent.id)
            finally:
                self.sweeps += 1
                self.formulation_ms.append((time.monotonic() - start) * 1000)

    # ------------------------------------------------------------ formulate

    async def formulate(self, agent: Any) -> bool:
        """One goal cycle for one agent. Returns True if the board changed."""
        beliefs: list[dict[str, Any]] = []
        if self._graph is not None:
            try:
                beliefs = await self._graph.fetch_beliefs(
                    agent.id, self._cfg.beliefs_limit)
            except Exception:
                log.warning("Belief fetch failed for %s — formulating from "
                            "the world alone", agent.id)

        leaf = self._build_leaf(agent, beliefs)
        raw = await self._invoke_llm(agent.id, leaf)
        if raw is None:
            return False
        goal = self._parse(raw)
        if goal is _PARSE_FAILED:
            self.parse_failures += 1
            return False

        now = time.monotonic()
        if goal is None:
            if self._board.pop(agent.id, None) is not None:
                self.goals_cleared += 1
                self._goal_set_at[agent.id] = now
                log.info("%s is now content (goal cleared)", agent.id)
                return True
            return False
        if self._board.get(agent.id) == goal:
            return False                        # identical: branch stays warm
        self._board[agent.id] = goal            # atomic: next dialogue task
        self._goal_set_at[agent.id] = now       # generates under this goal
        self.goals_set += 1
        log.info("%s now wants: %.80r", agent.id, goal)
        return True

    def _build_leaf(self, agent: Any, beliefs: list[dict[str, Any]]) -> str:
        lines = [f"Agent: {agent.id}"]
        if beliefs:
            lines.append("Beliefs (by importance):")
            lines += [f"- ({b.get('importance', 0):.2f}) {b.get('text', '')}"
                      for b in beliefs]
        inv = self._world.entities.owned_by(agent.id)
        lines.append("Inventory: " + (", ".join(
            f"{e.id} ({e.kind})" for e in inv) if inv else "empty"))
        r2 = self._cfg.nearby_radius ** 2
        nearby = [e for e in self._world.entities.all()
                  if e.owner_id is None
                  and (e.x - agent.x) ** 2 + (e.y - agent.y) ** 2 <= r2]
        lines.append("Nearby objects: " + (", ".join(
            f"{e.id} ({e.kind}, {json.dumps(e.state, separators=(',', ':'))})"
            for e in sorted(nearby, key=lambda e: e.id)) if nearby else "none"))
        cur = self._board.get(agent.id)
        lines.append(f"Current goal: {cur or 'none'}")
        return "\n".join(lines)

    async def _invoke_llm(self, agent_id: str, leaf: str) -> str | None:
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        await self._scheduler.submit(AgentTask(
            agent_id=agent_id,
            kind="goal_formulation",            # non-dialogue: dispatcher
            level=QueueLevel.GOAL,              # buffers + fulfills future
            messages=[
                {"role": "system", "content": GOALSEEK_PREFIX},   # stable root
                {"role": "user", "content": leaf},                # dynamic leaf
            ],
            payload={
                "result_future": fut,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "panopticon_goal", "strict": True,
                                    "schema": GOAL_JSON_SCHEMA}},
            }))
        try:
            return await asyncio.wait_for(fut, timeout=self._cfg.llm_timeout_s)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None
        except Exception:
            return None


_PARSE_FAILED = object()


def _parse_impl(raw: str) -> Any:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return _PARSE_FAILED
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return _PARSE_FAILED
    goal = obj.get("goal") if isinstance(obj, dict) else _PARSE_FAILED
    if goal is None:
        return None
    if isinstance(goal, str) and goal.strip():
        return goal.strip()[:200]
    return _PARSE_FAILED


GoalSeekDaemon._parse = staticmethod(_parse_impl)  # type: ignore[assignment]
