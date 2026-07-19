"""
Project Panopticon — Phase 4 Consolidation & Belief Extraction Engine
======================================================================
consolidation.py

Agents stop merely recalling what happened and start understanding what it
means. This daemon sweeps each agent's hot episodic memories, compresses
semantically coherent "clumps" into long-term structures, and tombstones the
raw sources out of the retrieval hot path (preserved for audit).

  episodic :Memory ×N ──clump──► LLM (Q2, via MLFQ) ──parse──► mutations:
                                                       ├─ consolidated_memory
                                                       ├─ tombstone ×N
                                                       ├─ relationship (affinity)
                                                       └─ belief (importance)

Placement in the machine:
  * LLM calls are NOT made around the engine — clump prompts are submitted
    as BACKGROUND (Q2) AgentTasks carrying a result_future. They therefore
    inherit strict-priority preemption (interactive speech always wins),
    AIMD admission, and Zombie-Reaper protection for free.
  * Graph mutations flow exclusively through the GraphWriteQueue's single
    writer — the daemon holds no Neo4j write path of its own, so the
    zero-deadlock invariant is preserved untouched.
  * Vector-side tombstoning is a payload flag (archived=true) filtered by
    must_not on the hot search path; points are never deleted (audit).

Idempotency (the load-bearing design decision):
  The consolidated node's id is uuid5(NAMESPACE, agent_id + sorted source
  ids) — DETERMINISTIC. If the daemon crashes after the Qdrant upsert but
  before graph mutations land, or a dead-lettered batch is replayed, the
  retried consolidation regenerates the SAME id: the Qdrant upsert
  overwrites itself and every Cypher statement MERGEs onto existing state.
  Affinity deltas (not naturally idempotent) carry this id as event_id and
  are guarded at-most-once in Cypher.

Fail-open philosophy (matching retrieval):
  Consolidation is an optimization, never a dependency. Every sweep is
  wrapped in a catch-all; an unparseable LLM response skips the clump (the
  sources simply remain hot and are re-swept later); a crashed sweep logs
  and waits for the next interval. If this daemon dies outright, the
  simulation continues on raw episodic memory exactly as it ran in Phase 3.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from tick_engine import AgentTask, GraphWriteQueue, MLFQScheduler, QueueLevel

log = logging.getLogger("panopticon.consolidation")

# uuid5 namespace for deterministic consolidated-memory ids.
_CONSOLIDATION_NS = uuid.UUID("b1a9c5f2-7d3e-4a8b-9c6d-0e1f2a3b4c5d")


# ============================================================================
# 1. Prompt (SHARED_WORLD_PREFIX rules apply: stable prefix, dynamic leaf)
# ============================================================================
# This block is BYTE-STABLE across every consolidation call for every agent:
# no ids, no timestamps, no counts. All consolidation requests share one
# RadixAttention branch; only the leaf (the memory list) varies. Same rules
# as SHARED_WORLD_PREFIX — one byte of drift forks the cache tree.

CONSOLIDATION_PREFIX = (
    "You are the memory-consolidation subsystem of a persistent town-square "
    "simulation. You will receive a numbered list of one agent's raw episodic "
    "memories that are semantically related.\n"
    "Compress them into long-term understanding. Respond with ONLY a JSON "
    "object — no prose, no markdown fences — with exactly these keys:\n"
    "{\n"
    '  "consolidated_summary": one sentence capturing the durable pattern,\n'
    '  "relationship_updates": [{"other_agent_id": str, '
    '"affinity_delta": float in [-0.25, 0.25], "reason": str}],\n'
    '  "belief_updates": [{"belief": one short declarative sentence, '
    '"importance": float in [0, 1]}]\n'
    "}\n"
    "Rules:\n"
    "1. The summary must reflect the SENTIMENT and repetition of the sources.\n"
    "2. Only reference agent ids that appear verbatim in the memories.\n"
    "3. Emit at most 3 relationship_updates and 3 belief_updates.\n"
    "4. Empty lists are valid when nothing durable emerges."
)


# ============================================================================
# 2. Config, data shapes, and the protocols this daemon needs
# ============================================================================

@dataclass(frozen=True)
class ConsolidationConfig:
    sweep_interval_s: float = 30.0      # pause between per-agent sweeps
    min_age_s: float = 60.0             # never consolidate mid-scene
    max_memories_per_sweep: int = 64
    min_clump_size: int = 3             # below this, memories stay episodic
    similarity_threshold: float = 0.80  # cosine floor for clump membership
    llm_timeout_s: float = 45.0         # future backstop (Reaper covers hangs)
    max_affinity_delta: float = 0.25    # clamp LLM enthusiasm


@dataclass
class EpisodicMemory:
    id: str
    summary: str
    t: float
    location: str
    vector: list[float] = field(default_factory=list)


class ConsolidationGraphReader(Protocol):
    async def fetch_unconsolidated_memories(
        self, agent_id: str, min_age_s: float, limit: int
    ) -> list[dict[str, Any]]: ...


class ConsolidationVectorStore(Protocol):
    async def upsert_memory(
        self, memory_id: str, vector: list[float], payload: dict[str, Any]
    ) -> None: ...
    async def get_vectors(self, memory_ids: list[str]) -> dict[str, list[float]]: ...
    async def archive_memories(self, memory_ids: list[str]) -> None: ...


class ConsolidationEmbedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


# ============================================================================
# 3. The daemon
# ============================================================================

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class ConsolidationDaemon:
    """Offline sweep-and-compress loop. One agent per sweep, round-robin."""

    def __init__(
        self,
        embedder: ConsolidationEmbedder,
        vectors: ConsolidationVectorStore,
        graph: ConsolidationGraphReader,
        scheduler: MLFQScheduler,
        graph_queue: GraphWriteQueue,
        agent_ids: Callable[[], list[str]],
        cfg: ConsolidationConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._vectors = vectors
        self._graph = graph
        self._scheduler = scheduler
        self._graph_queue = graph_queue
        self._agent_ids = agent_ids
        self._cfg = cfg or ConsolidationConfig()
        self._rr_index = 0
        # Telemetry
        self.sweeps = 0
        self.sweep_failures = 0
        self.clumps_consolidated = 0
        self.memories_tombstoned = 0
        self.llm_parse_failures = 0
        self.beliefs_written = 0
        self.relationships_written = 0
        self.sweep_ms: deque[float] = deque(maxlen=200)

    # ----------------------------------------------------------------- loop

    async def run(self) -> None:
        """Supervisor loop. NOTHING escapes: a failed sweep logs, counts, and
        waits for the next interval — the simulation never notices."""
        log.info("Consolidation daemon up: interval=%.0fs clump≥%d sim≥%.2f",
                 self._cfg.sweep_interval_s, self._cfg.min_clump_size,
                 self._cfg.similarity_threshold)
        while True:
            await asyncio.sleep(self._cfg.sweep_interval_s)
            agent_id = self._next_agent()
            if agent_id is None:
                continue
            start = time.monotonic()
            try:
                await self.sweep_agent(agent_id)
            except asyncio.CancelledError:
                raise                              # shutdown is not a failure
            except Exception:
                self.sweep_failures += 1
                log.exception("Sweep failed for %s — continuing on raw "
                              "episodic memory (fail-open)", agent_id)
            finally:
                self.sweeps += 1
                self.sweep_ms.append((time.monotonic() - start) * 1000)

    def _next_agent(self) -> str | None:
        ids = self._agent_ids()
        if not ids:
            return None
        agent_id = ids[self._rr_index % len(ids)]
        self._rr_index += 1
        return agent_id

    # ---------------------------------------------------------------- sweep

    async def sweep_agent(self, agent_id: str) -> int:
        """One idempotent sweep for one agent. Returns clumps consolidated."""
        rows = await self._graph.fetch_unconsolidated_memories(
            agent_id, self._cfg.min_age_s, self._cfg.max_memories_per_sweep
        )
        if len(rows) < self._cfg.min_clump_size:
            return 0

        vectors = await self._vectors.get_vectors([r["id"] for r in rows])
        memories = [
            EpisodicMemory(id=r["id"], summary=r["summary"] or "", t=r["t"],
                           location=r.get("location") or "unknown",
                           vector=vectors[r["id"]])
            for r in rows
            if r["id"] in vectors            # graph-only orphans can't clump;
        ]                                    # they stay episodic (harmless)

        consolidated = 0
        for clump in self._clump(memories):
            if len(clump) < self._cfg.min_clump_size:
                continue
            if await self._consolidate_clump(agent_id, clump):
                consolidated += 1
        return consolidated

    def _clump(self, memories: list[EpisodicMemory]) -> list[list[EpisodicMemory]]:
        """Greedy cosine clustering, oldest-seed-first. O(n²) with n ≤ 64 —
        ~2k float dot products, microseconds of CPU; no library needed.
        Every memory joins the first (oldest) seed it's similar enough to,
        so repeated near-identical observations collapse into one clump."""
        clumps: list[list[EpisodicMemory]] = []
        assigned: set[str] = set()
        ordered = sorted(memories, key=lambda m: m.t)
        for seed in ordered:
            if seed.id in assigned:
                continue
            clump = [seed]
            assigned.add(seed.id)
            for cand in ordered:
                if cand.id in assigned:
                    continue
                if _cosine(seed.vector, cand.vector) >= self._cfg.similarity_threshold:
                    clump.append(cand)
                    assigned.add(cand.id)
            clumps.append(clump)
        return clumps

    # ---------------------------------------------------------- consolidate

    async def _consolidate_clump(
        self, agent_id: str, clump: list[EpisodicMemory]
    ) -> bool:
        raw = await self._invoke_llm(agent_id, clump)
        if raw is None:
            return False
        parsed = self._parse_llm_json(raw)
        if parsed is None:
            self.llm_parse_failures += 1
            log.warning("Unparseable consolidation JSON for %s — clump of %d "
                        "stays episodic until next sweep", agent_id, len(clump))
            return False

        source_ids = sorted(m.id for m in clump)
        # DETERMINISTIC id — the idempotency keystone (see module docstring).
        memory_id = str(uuid.uuid5(_CONSOLIDATION_NS,
                                   agent_id + "|" + "|".join(source_ids)))
        location = Counter(m.location for m in clump).most_common(1)[0][0]
        t = time.time()
        summary = parsed["consolidated_summary"]

        # 1) Vector side first (orphan-vector asymmetry, as in Phase 3):
        #    consolidated point becomes hot, sources leave the hot path.
        vec = (await self._embedder.embed([summary]))[0]
        await self._vectors.upsert_memory(
            memory_id, vec,
            payload={"agent_id": agent_id, "summary": summary,
                     "location": location, "t": t, "consolidated": True},
        )
        await self._vectors.archive_memories(source_ids)

        # 2) Structural side: one mutation stream → single writer → batched,
        #    idempotent, deadlock-free.
        await self._graph_queue.submit({
            "type": "consolidated_memory",
            "memory_id": memory_id, "agent_id": agent_id,
            "location": location, "summary": summary, "t": t,
            "source_ids": source_ids, "source_count": len(source_ids),
        })
        for sid in source_ids:
            await self._graph_queue.submit({
                "type": "tombstone",
                "memory_id": sid, "agent_id": agent_id,
                "archived_at": t, "consolidated_into": memory_id,
            })
        for rel in parsed["relationship_updates"]:
            await self._graph_queue.submit({
                "type": "relationship",
                "agent_id": agent_id,
                "other_agent_id": rel["other_agent_id"],
                "affinity_delta": rel["affinity_delta"],
                "reason": rel["reason"],
                "event_id": memory_id,        # at-most-once guard in Cypher
                "t": t,
            })
            self.relationships_written += 1
        for bel in parsed["belief_updates"]:
            await self._graph_queue.submit({
                "type": "belief",
                "agent_id": agent_id,
                "belief": bel["belief"],
                "importance": bel["importance"],
                "t": t,
            })
            self.beliefs_written += 1

        self.clumps_consolidated += 1
        self.memories_tombstoned += len(source_ids)
        log.info("Consolidated %d memories → %s for %s (%d rel, %d belief)",
                 len(source_ids), memory_id[:8], agent_id,
                 len(parsed["relationship_updates"]),
                 len(parsed["belief_updates"]))
        return True

    async def _invoke_llm(
        self, agent_id: str, clump: list[EpisodicMemory]
    ) -> str | None:
        """Submit the consolidation prompt at BACKGROUND priority and await
        the dispatcher-fulfilled future. Q2 semantics mean this generation
        runs only when interactive lanes are idle — consolidation never
        steals prefill from live dialogue."""
        leaf = "Memories:\n" + "\n".join(
            f"{i + 1}. [{m.location}] {m.summary}" for i, m in enumerate(clump)
        )
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        await self._scheduler.submit(AgentTask(
            agent_id=agent_id,
            kind="consolidation",
            level=QueueLevel.BACKGROUND,
            messages=[
                {"role": "system", "content": CONSOLIDATION_PREFIX},  # stable
                {"role": "user", "content": leaf},                    # dynamic
            ],
            payload={"result_future": fut},
        ))
        try:
            return await asyncio.wait_for(fut, timeout=self._cfg.llm_timeout_s)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            log.warning("Consolidation generation timed out for %s", agent_id)
            return None
        except Exception:
            log.warning("Consolidation generation failed for %s", agent_id,
                        exc_info=True)
            return None

    # ----------------------------------------------------------------- parse

    def _parse_llm_json(self, raw: str) -> dict[str, Any] | None:
        """Tolerant-but-strict parsing: tolerate fences/preamble around the
        JSON object, but strictly validate and clamp every field we act on.
        Any structural violation → None → fail-open skip."""
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None

        summary = obj.get("consolidated_summary")
        if not isinstance(summary, str) or not summary.strip():
            return None

        cap = self._cfg.max_affinity_delta
        rels: list[dict[str, Any]] = []
        for r in (obj.get("relationship_updates") or [])[:3]:
            if not isinstance(r, dict):
                continue
            other = r.get("other_agent_id")
            delta = r.get("affinity_delta")
            if not isinstance(other, str) or not isinstance(delta, (int, float)):
                continue
            rels.append({
                "other_agent_id": other,
                "affinity_delta": max(-cap, min(cap, float(delta))),
                "reason": str(r.get("reason", ""))[:200],
            })

        beliefs: list[dict[str, Any]] = []
        for b in (obj.get("belief_updates") or [])[:3]:
            if not isinstance(b, dict):
                continue
            text = b.get("belief")
            imp = b.get("importance")
            if not isinstance(text, str) or not text.strip() \
                    or not isinstance(imp, (int, float)):
                continue
            beliefs.append({
                "belief": text.strip()[:300],
                "importance": max(0.0, min(1.0, float(imp))),
            })

        return {
            "consolidated_summary": summary.strip()[:500],
            "relationship_updates": rels,
            "belief_updates": beliefs,
        }
