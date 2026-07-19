"""Phase 4 verification suite (no GPU / no servers — real AsyncQdrantClient
in ':memory:' mode, real MLFQ/AIMD/GraphWriteQueue plumbing throughout).

1. CLUSTERING     — 5 near-duplicate episodic memories + 2 unrelated ones →
                    exactly ONE consolidated node (source_count=5); the
                    unrelated pair stays episodic (below min_clump_size).
2. TOMBSTONING    — sources archived in BOTH stores: hot Qdrant search
                    excludes them (consolidated node included), audit
                    retrieve-by-id still works, graph edges transitioned
                    REMEMBERS → ARCHIVED_REMEMBERS via the single writer.
3. SENTIMENT      — negative sources → negative consolidated summary,
                    negative affinity delta toward the involved agent,
                    and a belief mutation.
4. Q2 ROUTING     — consolidation prompts flow through the MLFQ at
                    BACKGROUND priority with a byte-stable system prefix.
5. IDEMPOTENCY    — a second sweep is a no-op; the consolidated id is
                    deterministic (uuid5 over agent + sorted sources).
6. FAIL-OPEN      — garbage LLM JSON: clump skipped, memories stay hot,
                    counted; a crashing graph reader: daemon loop survives.
"""
import asyncio
import hashlib
import logging
import math
import time
import uuid

import consolidation as cons
import memory as memory_mod
import tick_engine as te

logging.disable(logging.ERROR)
DIM = 32


def text_vec(text: str) -> list[float]:
    """Word-averaged hash embedding: shared words → high cosine similarity."""
    acc = [0.0] * DIM
    words = [w for w in text.lower().split() if len(w) > 2]
    for w in words:
        h = hashlib.sha256(w.encode()).digest()
        for i in range(DIM):
            acc[i] += h[i] / 255.0 - 0.5   # SIGNED: unrelated ≈ orthogonal
    n = math.sqrt(sum(x * x for x in acc)) or 1.0
    return [x / n for x in acc]


class LocalEmbedder:
    async def embed(self, texts):
        return [text_vec(t) for t in texts]

    async def probe_dim(self):
        return DIM


class RecordingGraphClient:
    """In-memory graph honoring the mutation semantics the daemon relies on,
    so second-sweep idempotency is tested against real state transitions."""

    def __init__(self):
        self.memories: dict[str, dict] = {}   # id -> {summary,t,location,archived,consolidated}
        self.mutations: list[dict] = []
        self.batches_written = 0
        self.mutations_written = 0

    def seed(self, mid, agent_id, summary, location, t):
        self.memories[mid] = {"id": mid, "agent_id": agent_id, "summary": summary,
                              "location": location, "t": t,
                              "archived": False, "consolidated": False}

    async def execute_batch(self, mutations):
        self.batches_written += 1
        self.mutations_written += len(mutations)
        for m in mutations:
            self.mutations.append(m)
            if m["type"] == "tombstone":
                self.memories[m["memory_id"]]["archived"] = True
            elif m["type"] == "consolidated_memory":
                self.memories[m["memory_id"]] = {
                    "id": m["memory_id"], "agent_id": m["agent_id"],
                    "summary": m["summary"], "location": m["location"],
                    "t": m["t"], "archived": False, "consolidated": True,
                }

    async def fetch_memories(self, memory_ids):
        return []

    async def fetch_unconsolidated_memories(self, agent_id, min_age_s, limit):
        cutoff = time.time() - min_age_s
        rows = [m for m in self.memories.values()
                if m["agent_id"] == agent_id and not m["archived"]
                and not m["consolidated"] and m["t"] < cutoff]
        return sorted(rows, key=lambda r: r["t"])[:limit]

    def of_type(self, t):
        return [m for m in self.mutations if m["type"] == t]


class SentimentLLM:
    """Scripted consolidator: reads the memory leaf, keys sentiment off the
    verbs it finds, and emits schema-correct JSON in streamed chunks. Also
    records every request so routing/prefix assertions are possible."""

    NEG = ("cheated", "argued", "insulted", "stole")

    def __init__(self):
        self.requests: list[list[dict]] = []

    async def stream_chat(self, model, messages):
        self.requests.append([dict(m) for m in messages])
        leaf = messages[-1]["content"].lower()
        negative = any(v in leaf for v in self.NEG)
        mood = "hostile" if negative else "friendly"
        delta = -0.2 if negative else 0.2
        out = (
            '{"consolidated_summary": "agent_007 has repeatedly cheated me over '
            f'grain deals; our dealings are {mood}.", '
            '"relationship_updates": [{"other_agent_id": "agent_007", '
            f'"affinity_delta": {delta}, "reason": "repeated grain cheating"}}], '
            '"belief_updates": [{"belief": "Grain deals with agent_007 are '
            'unsafe.", "importance": 0.9}]}'
        )
        for i in range(0, len(out), 40):
            yield out[i:i + 40]


class GarbageLLM:
    async def stream_chat(self, model, messages):
        yield "Sure! Here's my analysis: the memories show {broken json"


class LevelSpyScheduler(te.MLFQScheduler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.submitted_levels: list[te.QueueLevel] = []

    async def submit(self, task):
        self.submitted_levels.append(task.level)
        await super().submit(task)


async def build_stack(llm):
    """Real plumbing: scheduler → dispatcher → AIMD → single-writer queue."""
    cfg = te.EngineConfig()
    graph = RecordingGraphClient()
    scheduler = LevelSpyScheduler(cfg)
    limiter = te.AIMDLimiter(cfg)
    reaper = te.ZombieReaper(cfg)
    gq = te.GraphWriteQueue(cfg, graph)
    dispatcher = te.InferenceDispatcher(cfg, scheduler, limiter, reaper, llm,
                                        te.LiveKitPublisher(), gq)
    vectors = memory_mod.RealQdrantClient(":memory:")
    await vectors.ensure_collection(DIM)
    bg = [asyncio.create_task(dispatcher.run_worker()),
          asyncio.create_task(gq.run_writer())]
    daemon = cons.ConsolidationDaemon(
        LocalEmbedder(), vectors, graph, scheduler, gq,
        agent_ids=lambda: ["agent_001"],
        cfg=cons.ConsolidationConfig(min_age_s=0.0, sweep_interval_s=0.01,
                                     # toy hash embeddings score lower than
                                     # real models; 0.80 default is for bge-m3
                                     similarity_threshold=0.5),
    )
    return daemon, graph, vectors, scheduler, bg


async def seed_world(graph, vectors):
    """5 near-duplicate hostile grain memories + 2 unrelated ones."""
    texts = [
        "agent_007 cheated me on grain prices at the market stall",
        "agent_007 cheated me again over grain prices near the stall",
        "caught agent_007 as he cheated on the grain prices once more",
        "agent_007 cheated me on grain prices, third market visit running",
        "yet again agent_007 cheated me over the grain prices",
        "fed the pigeons breadcrumbs by the fountain this morning",
        "watched the parade pass the northern gate with music",
    ]
    old = time.time() - 3600
    ids = []
    for i, txt in enumerate(texts):
        mid = str(uuid.uuid4())
        ids.append(mid)
        graph.seed(mid, "agent_001", txt, "market" if i < 5 else "plaza", old + i)
        await vectors.upsert_memory(mid, text_vec(txt), payload={
            "agent_id": "agent_001", "summary": txt,
            "location": "market" if i < 5 else "plaza", "t": old + i,
        })
    return ids[:5], ids[5:]


async def main():
    # ---------- Tests 1–5 on one seeded stack --------------------------------
    llm = SentimentLLM()
    daemon, graph, vectors, scheduler, bg = await build_stack(llm)
    grain_ids, other_ids = await seed_world(graph, vectors)

    consolidated = await daemon.sweep_agent("agent_001")
    await asyncio.sleep(0.2)                       # let the single writer drain

    # 1. CLUSTERING
    assert consolidated == 1 and daemon.clumps_consolidated == 1
    cnodes = graph.of_type("consolidated_memory")
    assert len(cnodes) == 1 and cnodes[0]["source_count"] == 5
    assert sorted(cnodes[0]["source_ids"]) == sorted(grain_ids)
    remaining = await graph.fetch_unconsolidated_memories("agent_001", 0, 64)
    assert {r["id"] for r in remaining} == set(other_ids)   # pigeons & parade live on
    print("PASS clustering: 5 duplicates → 1 node; 2 unrelated stay episodic")

    # 2. TOMBSTONING (both stores)
    tombs = graph.of_type("tombstone")
    assert {t["memory_id"] for t in tombs} == set(grain_ids)
    assert all(t["consolidated_into"] == cnodes[0]["memory_id"] for t in tombs)
    hot = await vectors.search(text_vec("agent_007 cheated grain prices"),
                               "agent_001", 10)
    hot_ids = {h.memory_id for h in hot}
    assert not (hot_ids & set(grain_ids)), "archived sources leaked into hot path"
    assert cnodes[0]["memory_id"] in hot_ids, "consolidated node missing from hot path"
    audit = await vectors.get_vectors(grain_ids)
    assert len(audit) == 5                                   # preserved for audit
    print("PASS tombstoning: hot path clean, consolidated node hot, audit intact")

    # 3. SENTIMENT fidelity
    assert "cheated" in cnodes[0]["summary"] and "hostile" in cnodes[0]["summary"]
    rels = graph.of_type("relationship")
    assert len(rels) == 1 and rels[0]["other_agent_id"] == "agent_007"
    assert rels[0]["affinity_delta"] == -0.2                 # negative, clamp-safe
    assert rels[0]["event_id"] == cnodes[0]["memory_id"]     # at-most-once key
    beliefs = graph.of_type("belief")
    assert len(beliefs) == 1 and beliefs[0]["importance"] == 0.9
    print("PASS sentiment: hostile summary, −0.2 affinity to agent_007, belief 0.9")

    # 4. Q2 ROUTING + stable prefix
    assert scheduler.submitted_levels == [te.QueueLevel.BACKGROUND]
    (llm_req,) = llm.requests                                # exactly one call
    assert llm_req[0]["content"] == cons.CONSOLIDATION_PREFIX  # byte-stable root
    assert llm_req[-1]["content"].startswith("Memories:")      # dynamic leaf
    print("PASS routing: Q2 (BACKGROUND), prefix byte-stable, memories in leaf")

    # 5. IDEMPOTENCY
    muts_before = len(graph.mutations)
    assert await daemon.sweep_agent("agent_001") == 0        # nothing hot enough
    await asyncio.sleep(0.1)
    assert len(graph.mutations) == muts_before               # zero new mutations
    expected = str(uuid.uuid5(cons._CONSOLIDATION_NS,
                              "agent_001|" + "|".join(sorted(grain_ids))))
    assert cnodes[0]["memory_id"] == expected                # deterministic id
    print("PASS idempotency: re-sweep is a no-op; id is uuid5-deterministic")

    for t in bg:
        t.cancel()

    # 6a. FAIL-OPEN: unparseable LLM output
    daemon2, graph2, vectors2, _, bg2 = await build_stack(GarbageLLM())
    await seed_world(graph2, vectors2)
    assert await daemon2.sweep_agent("agent_001") == 0
    assert daemon2.llm_parse_failures == 1
    still_hot = await graph2.fetch_unconsolidated_memories("agent_001", 0, 64)
    assert len(still_hot) == 7                               # nothing lost, re-sweepable
    assert not graph2.of_type("tombstone")
    for t in bg2:
        t.cancel()
    print("PASS fail-open(parse): garbage JSON → clump skipped, memories stay hot")

    # 6b. FAIL-OPEN: sweep crash never kills the daemon loop
    class ExplodingGraph(RecordingGraphClient):
        async def fetch_unconsolidated_memories(self, *a, **kw):
            raise RuntimeError("neo4j exploded")

    cfg = te.EngineConfig()
    gq = te.GraphWriteQueue(cfg, ExplodingGraph())
    daemon3 = cons.ConsolidationDaemon(
        LocalEmbedder(), vectors, ExplodingGraph(), te.MLFQScheduler(cfg), gq,
        agent_ids=lambda: ["agent_001"],
        cfg=cons.ConsolidationConfig(sweep_interval_s=0.02),
    )
    runner = asyncio.create_task(daemon3.run())
    await asyncio.sleep(0.15)
    assert not runner.done(), "daemon loop died on sweep failure!"
    assert daemon3.sweep_failures >= 2                       # kept sweeping
    runner.cancel()
    print(f"PASS fail-open(crash): {daemon3.sweep_failures} failed sweeps "
          "absorbed, loop alive")

    print("\nALL PHASE 4 TESTS PASSED")


asyncio.run(main())
