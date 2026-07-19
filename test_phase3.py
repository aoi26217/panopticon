"""Phase 3 verification suite (no GPU / no servers required — Qdrant runs
in-process via the REAL AsyncQdrantClient in ':memory:' mode).

1. RealEmbeddingClient: OpenAI-shape parsing, dim probe, error path.
2. Dual-write: store_memory → Qdrant point + graph mutation share one UUID;
   vector-write failure yields NO graph mutation (orphan asymmetry).
3. Hybrid retrieval: vector seeds → graph expansion receives exactly the
   seed UUIDs → provenance-rich context, well under the 100 ms budget.
4. Fail-open: a slow embedder trips the budget and returns "" on time.
5. Isolation: agent A never retrieves agent B's memories.
6. Leaf-only injection: SHARED_WORLD_PREFIX byte-identical after injection;
   memories land only in the final user message.
7. Slot contract: AIMD slot is released BEFORE dual-write I/O completes.
"""
import asyncio
import hashlib
import logging
import math
import time

from aiohttp import web

import memory as memory_mod
import tick_engine as te

logging.disable(logging.ERROR)
DIM = 32


def text_vec(text: str) -> list[float]:
    """Deterministic pseudo-embedding: same word stem → similar vector."""
    h = hashlib.sha256(text.lower().split()[-1].encode()).digest()
    v = [b / 255.0 for b in h[:DIM]]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


class LocalEmbedder:
    async def embed(self, texts):
        return [text_vec(t) for t in texts]

    async def probe_dim(self):
        return DIM


class SlowEmbedder(LocalEmbedder):
    async def embed(self, texts):
        await asyncio.sleep(0.5)
        return await super().embed(texts)


class RecordingGraph:
    """Fake GraphReader: records the ids it was asked to expand."""
    def __init__(self):
        self.requested_ids = []
        self.rows = {}

    async def fetch_memories(self, memory_ids):
        self.requested_ids.append(list(memory_ids))
        return [self.rows[i] for i in memory_ids if i in self.rows]


async def make_service(embedder=None, graph=None):
    svc = memory_mod.MemoryService(
        embedder or LocalEmbedder(),
        memory_mod.RealQdrantClient(":memory:"),
        graph,
    )
    await svc.connect()
    return svc


# ------------------------------------------------------------------- tests

async def test_embedding_client():
    async def handler(request):
        body = await request.json()
        return web.json_response({"data": [
            {"index": i, "embedding": text_vec(t)}
            for i, t in enumerate(body["input"])
        ][::-1]})  # deliberately reversed: client must sort by index

    async def err(request):
        return web.json_response({"error": "no model"}, status=500)

    app = web.Application(); app.router.add_post("/v1/embeddings", handler)
    runner = web.AppRunner(app); await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18551).start()
    eapp = web.Application(); eapp.router.add_post("/v1/embeddings", err)
    erunner = web.AppRunner(eapp); await erunner.setup()
    await web.TCPSite(erunner, "127.0.0.1", 18552).start()

    c = memory_mod.RealEmbeddingClient("http://127.0.0.1:18551", "bge-m3")
    assert await c.probe_dim() == DIM
    vecs = await c.embed(["alpha", "beta"])
    assert vecs[0] == text_vec("alpha") and vecs[1] == text_vec("beta")  # order restored
    bad = memory_mod.RealEmbeddingClient("http://127.0.0.1:18552", "x")
    try:
        await bad.embed(["x"]); raise AssertionError("expected error")
    except memory_mod.EmbeddingRequestError as exc:
        assert "500" in str(exc)
    await c.close(); await bad.close()
    await runner.cleanup(); await erunner.cleanup()
    print("PASS embeddings: OpenAI shape, index re-sort, dim probe, 500 path")


async def test_dual_write():
    graph = RecordingGraph()
    svc = await make_service(graph=graph)
    muts = await svc.store_memory("agent_001", "saw the market crash", "plaza")
    assert len(muts) == 1 and muts[0]["type"] == "memory"
    mid = muts[0]["memory_id"]
    hits = await svc._vectors.search(text_vec("crash"), "agent_001", 3)
    assert hits and hits[0].memory_id == mid       # same UUID in both stores
    assert hits[0].payload["summary"] == "saw the market crash"

    class BrokenVectors(memory_mod.RealQdrantClient):
        async def upsert_memory(self, *a, **kw):
            raise RuntimeError("qdrant down")
    svc2 = memory_mod.MemoryService(LocalEmbedder(), BrokenVectors(":memory:"), None)
    await svc2.connect()
    assert await svc2.store_memory("a", "s", "l") == []   # no orphan graph node
    await svc.close(); await svc2.close()
    print(f"PASS dual-write: shared UUID {mid[:8]}…, vector-fail → zero mutations")


async def test_hybrid_retrieval():
    graph = RecordingGraph()
    svc = await make_service(graph=graph)
    seeds = {}
    for text in ["argued about grain prices", "argued about weather",
                 "saw a parade", "fed the pigeons", "argued about taxes"]:
        m = (await svc.store_memory("agent_001", text, "plaza"))[0]
        seeds[text] = m["memory_id"]
        graph.rows[m["memory_id"]] = {
            "id": m["memory_id"], "summary": text, "t": m["t"],
            "agent_id": "agent_001", "location": "plaza",
            "involves": ["agent_007"] if "grain" in text else [],
        }
    t0 = time.monotonic()
    ctx = await svc.retrieve_context("agent_001", "a heated argument about grain prices")
    ms = (time.monotonic() - t0) * 1000
    assert ms < 100, f"retrieval took {ms:.1f}ms"
    assert len(graph.requested_ids) == 1 and len(graph.requested_ids[0]) == 3
    lines = ctx.splitlines()
    assert len(lines) == 3
    assert "argued about grain prices at plaza (involving agent_007)" in lines[0]
    await svc.close()
    print(f"PASS hybrid: 3 seeds → graph expansion → provenance context in {ms:.1f}ms")


async def test_fail_open_budget():
    svc = await make_service(embedder=SlowEmbedder())
    t0 = time.monotonic()
    ctx = await svc.retrieve_context("agent_001", "anything")
    ms = (time.monotonic() - t0) * 1000
    assert ctx == "" and svc.retrieval_timeouts == 1
    assert ms < 200, f"fail-open took {ms:.1f}ms"
    await svc.close()
    print(f"PASS fail-open: 500ms embedder → \"\" in {ms:.0f}ms, timeout counted")


async def test_agent_isolation():
    svc = await make_service()
    await svc.store_memory("agent_A", "buried gold under the oak", "forest")
    ctx = await svc.retrieve_context("agent_B", "buried gold under the oak")
    assert ctx == ""
    ctx_a = await svc.retrieve_context("agent_A", "buried gold under the oak")
    assert "buried gold" in ctx_a          # payload fallback (no graph reader)
    await svc.close()
    print("PASS isolation: agent_B cannot retrieve agent_A's memories")


class CapturingSGLang:
    def __init__(self):
        self.seen_messages = None

    async def stream_chat(self, model, messages):
        self.seen_messages = [dict(m) for m in messages]
        yield "Okay."


class SlowWriteMemory:
    """MemoryStore whose store_memory is glacial — proves slot release."""
    def __init__(self):
        self.write_started = asyncio.Event()

    async def retrieve_context(self, agent_id, query_text):
        return "- old rivalry with agent_002"

    async def store_memory(self, agent_id, summary, location):
        self.write_started.set()
        await asyncio.sleep(5.0)
        return []


def build_dispatcher(sglang, mem):
    cfg = te.EngineConfig()
    sched = te.MLFQScheduler(cfg)
    lim = te.AIMDLimiter(cfg)
    return te.InferenceDispatcher(cfg, sched, lim, te.ZombieReaper(cfg), sglang,
                                  te.LiveKitPublisher(),
                                  te.GraphWriteQueue(cfg, te.MockNeo4jClient()),
                                  memory=mem), lim


async def test_leaf_only_injection():
    cap = CapturingSGLang()
    disp, _ = build_dispatcher(cap, SlowWriteMemory())
    task = te.AgentTask(
        agent_id="agent_001", kind="dialogue", level=te.QueueLevel.INTERACTIVE,
        messages=[{"role": "system", "content": te.SHARED_WORLD_PREFIX},
                  {"role": "user", "content": "agent_002 crossed your path. React."}],
    )
    await disp._execute(task)
    system, user = cap.seen_messages
    assert system["content"] == te.SHARED_WORLD_PREFIX      # byte-identical root
    assert "Relevant memories:" in user["content"]
    assert "old rivalry" in user["content"]
    assert user["content"].startswith("agent_002 crossed")  # leaf appended, not replaced
    print("PASS leaf injection: prefix byte-stable, memories in final user msg only")


async def test_slot_released_before_dual_write():
    mem = SlowWriteMemory()
    disp, lim = build_dispatcher(CapturingSGLang(), mem)
    task = te.AgentTask(
        agent_id="agent_001", kind="memory_extraction", level=te.QueueLevel.BACKGROUND,
        messages=[{"role": "user", "content": "Summarize."}],
        payload={"observation": "x", "location": "plaza"},
    )
    await disp._execute(task)                 # returns despite 5 s write
    assert lim.active == 0, "AIMD slot held hostage by dual-write I/O!"
    await asyncio.wait_for(mem.write_started.wait(), 1.0)
    assert len(disp._background_writes) == 1  # detached task alive & referenced
    for t in disp._background_writes:
        t.cancel()
    print("PASS slot contract: _execute returned & slot freed while write runs detached")


async def main():
    await test_embedding_client()
    await test_dual_write()
    await test_hybrid_retrieval()
    await test_fail_open_budget()
    await test_agent_isolation()
    await test_leaf_only_injection()
    await test_slot_released_before_dual_write()
    print("\nALL PHASE 3 TESTS PASSED")


asyncio.run(main())
