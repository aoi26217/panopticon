"""
Project Panopticon — Phase 3 Hybrid Graph-Vector Memory
========================================================
memory.py

Agents remember. Two datastores hold LINKED representations of every memory:

  * Qdrant  — the semantic index: embedding vectors + payload, answers
              "what past events FEEL like this trigger?"
  * Neo4j   — the structural truth: (:Agent)-[:REMEMBERS]->(:Memory)-[:AT]->
              (:Location), answers "what exactly happened, where, with whom?"

They are joined by a shared UUID (Qdrant point id == Neo4j Memory.id).

Retrieval (the simulated DW-PR two-step, ~100 ms hard budget):
  1. VECTOR SEED IDENTIFICATION — embed the trigger event, query Qdrant for
     the top-k semantically nearest memories OF THIS AGENT. This is the
     "semantic alignment" step: a sparsely-connected but highly relevant
     memory (one isolated betrayal) outranks dense background structure.
  2. GRAPH EXPANSION — pass the seed UUIDs to Neo4j for an exact 1-hop
     neighborhood fetch (who/where/when), yielding provenance-rich context
     instead of bare payload text.

Failure philosophy: retrieval FAILS OPEN. A missed memory costs one slightly
amnesiac line of dialogue; a blown latency budget costs the illusion of
real-time. asyncio.timeout enforces the budget; any error returns "".

Write path (dual-write, executed as fire-and-forget background work — never
while holding an AIMD inference slot):
  embed(summary) → upsert Qdrant point (uuid, vector, payload)
                 → emit "memory" mutation → GraphWriteQueue (single writer)
Qdrant first, then Neo4j: if the process dies mid-write, an orphan vector is
harmless (its uuid dereferences to nothing and graph expansion skips it),
whereas a graph node with no vector would be permanently unfindable.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import aiohttp
from qdrant_client import AsyncQdrantClient, models

log = logging.getLogger("panopticon.memory")


# ============================================================================
# 1. Embedding client (OpenAI-compatible /v1/embeddings)
# ============================================================================

class EmbeddingClient(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def probe_dim(self) -> int: ...


class EmbeddingRequestError(RuntimeError):
    pass


class RealEmbeddingClient:
    """aiohttp client for any OpenAI-compatible embeddings endpoint — an
    SGLang/vLLM/TEI-served model (e.g. bge-m3, gte-large) on the multimodal
    node, or a hosted API. Latency budget: one embed call sits INSIDE the
    100 ms retrieval budget, so serve this on-cluster, not over the WAN."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        connect_timeout_s: float = 3.0,
        sock_read_timeout_s: float = 10.0,
    ) -> None:
        self._url = base_url.rstrip("/") + "/v1/embeddings"
        self._model = model
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=connect_timeout_s, sock_read=sock_read_timeout_s
        )
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._session = aiohttp.ClientSession(
                timeout=self._timeout, headers=headers,
                connector=aiohttp.TCPConnector(limit=64),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        session = await self._ensure_session()
        async with session.post(self._url, json={"model": self._model,
                                                 "input": texts}) as resp:
            if resp.status != 200:
                body = (await resp.text())[:300]
                raise EmbeddingRequestError(f"Embeddings HTTP {resp.status}: {body}")
            data = await resp.json()
        # OpenAI shape: {"data": [{"index": i, "embedding": [...]}, ...]}
        rows = sorted(data["data"], key=lambda r: r["index"])
        return [r["embedding"] for r in rows]

    async def probe_dim(self) -> int:
        """One-time startup probe so nobody hardcodes (and later mismatches)
        the model's dimensionality."""
        vec = (await self.embed(["dimension probe"]))[0]
        return len(vec)


# ============================================================================
# 2. Vector store client (Qdrant)
# ============================================================================

@dataclass(frozen=True)
class ScoredMemory:
    memory_id: str
    score: float
    payload: dict[str, Any]


class VectorClient(Protocol):
    async def ensure_collection(self, dim: int) -> None: ...
    async def upsert_memory(
        self, memory_id: str, vector: list[float], payload: dict[str, Any]
    ) -> None: ...
    async def search(
        self, vector: list[float], agent_id: str, limit: int
    ) -> list[ScoredMemory]: ...


class RealQdrantClient:
    """Official AsyncQdrantClient. `location` accepts a server URL
    (http://qdrant:6333), a local path, or ":memory:" — the local modes run
    fully in-process, which is how the test suite exercises this exact class
    and how a laptop dev loop runs with zero infrastructure."""

    COLLECTION = "panopticon_memories"

    def __init__(self, location: str, *, api_key: str | None = None) -> None:
        if location.startswith(("http://", "https://")):
            self._client = AsyncQdrantClient(url=location, api_key=api_key)
        else:
            self._client = AsyncQdrantClient(location)
        # Telemetry
        self.upserts = 0
        self.searches = 0

    async def close(self) -> None:
        await self._client.close()

    async def ensure_collection(self, dim: int) -> None:
        if not await self._client.collection_exists(self.COLLECTION):
            await self._client.create_collection(
                self.COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                ),
            )
            # Payload index: agent_id filtering is on EVERY retrieval's hot
            # path; without the index Qdrant falls back to full-scan filters.
            await self._client.create_payload_index(
                self.COLLECTION, field_name="agent_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            log.info("Qdrant collection %r created (dim=%d)", self.COLLECTION, dim)

    async def upsert_memory(
        self, memory_id: str, vector: list[float], payload: dict[str, Any]
    ) -> None:
        await self._client.upsert(
            self.COLLECTION,
            points=[models.PointStruct(id=memory_id, vector=vector, payload=payload)],
        )
        self.upserts += 1

    async def search(
        self, vector: list[float], agent_id: str, limit: int
    ) -> list[ScoredMemory]:
        res = await self._client.query_points(
            self.COLLECTION,
            query=vector,
            limit=limit,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="agent_id",
                                          match=models.MatchValue(value=agent_id))
                ],
                # HOT PATH ONLY: consolidation tombstones episodic memories by
                # flagging archived=true; they vanish from retrieval here but
                # remain retrievable by id for audit (see archive_memories).
                must_not=[
                    models.FieldCondition(key="archived",
                                          match=models.MatchValue(value=True))
                ],
            ),
        )
        self.searches += 1
        return [
            ScoredMemory(memory_id=str(p.id), score=p.score, payload=p.payload or {})
            for p in res.points
        ]

    async def get_vectors(self, memory_ids: list[str]) -> dict[str, list[float]]:
        """Fetch stored vectors by id (consolidation clumping input).
        Retrieval-by-id ignores the archived flag deliberately: audit and
        offline analytics can always reach tombstoned points."""
        points = await self._client.retrieve(
            self.COLLECTION, ids=memory_ids, with_vectors=True
        )
        return {str(p.id): list(p.vector) for p in points if p.vector is not None}

    async def archive_memories(self, memory_ids: list[str]) -> None:
        """Tombstone in the vector store: flip archived=true so the hot
        search filter (must_not) excludes these points. The vectors and
        payloads are PRESERVED — nothing is deleted — so the episodic record
        stays available for audit/replay. Idempotent by construction
        (set_payload overwrites the same flag)."""
        if memory_ids:
            await self._client.set_payload(
                self.COLLECTION, payload={"archived": True}, points=memory_ids
            )


# ============================================================================
# 3. Memory service: dual-write + budget-bounded hybrid retrieval
# ============================================================================

class GraphReader(Protocol):
    """The slice of GraphClient that retrieval needs (1-hop expansion).
    Reads do NOT go through the single-writer queue — that serializes
    mutations only; Neo4j read transactions are safely concurrent."""

    async def fetch_memories(self, memory_ids: list[str]) -> list[dict[str, Any]]: ...


class MemoryService:
    """Implements the tick_engine.MemoryStore protocol.

    retrieve_context() is called by the InferenceDispatcher BEFORE acquiring
    an AIMD slot, so its latency overlaps queueing rather than occupying GPU
    admission. store_memory() is invoked fire-and-forget after generation, so
    embedding/upsert I/O never holds an inference slot either.
    """

    def __init__(
        self,
        embedder: EmbeddingClient,
        vectors: VectorClient,
        graph: GraphReader | None,
        *,
        top_k: int = 3,
        retrieval_budget_ms: float = 100.0,
    ) -> None:
        self._embedder = embedder
        self._vectors = vectors
        self._graph = graph
        self._top_k = top_k
        self._budget_s = retrieval_budget_ms / 1000.0
        # Telemetry
        self.retrievals = 0
        self.retrieval_timeouts = 0
        self.retrieval_failures = 0
        self.memories_stored = 0
        self.retrieval_ms: deque[float] = deque(maxlen=500)

    async def connect(self) -> None:
        dim = await self._embedder.probe_dim()
        await self._vectors.ensure_collection(dim)
        log.info("MemoryService up: dim=%d top_k=%d budget=%.0fms",
                 dim, self._top_k, self._budget_s * 1000)

    async def close(self) -> None:
        for c in (self._embedder, self._vectors):
            closer = getattr(c, "close", None)
            if closer is not None:
                await closer()

    # ------------------------------------------------------------- retrieval

    async def retrieve_context(self, agent_id: str, query_text: str) -> str:
        """Two-step hybrid retrieval under a HARD latency budget. Returns a
        formatted context block, or "" (fail-open) on timeout/error/no-hits."""
        start = time.monotonic()
        try:
            async with asyncio.timeout(self._budget_s):
                # Step 1 — vector seed identification (semantic alignment).
                vec = (await self._embedder.embed([query_text]))[0]
                seeds = await self._vectors.search(vec, agent_id, self._top_k)
                if not seeds:
                    return ""

                # Step 2 — exact 1-hop graph expansion of the seed UUIDs.
                rows_by_id: dict[str, dict[str, Any]] = {}
                if self._graph is not None:
                    rows = await self._graph.fetch_memories(
                        [s.memory_id for s in seeds]
                    )
                    rows_by_id = {r["id"]: r for r in rows}

                return self._format_context(seeds, rows_by_id)
        except (TimeoutError, asyncio.TimeoutError):
            self.retrieval_timeouts += 1
            log.warning("Memory retrieval blew the %.0fms budget for %s — "
                        "proceeding without context", self._budget_s * 1000, agent_id)
            return ""
        except Exception:
            self.retrieval_failures += 1
            log.exception("Memory retrieval failed for %s — proceeding without", agent_id)
            return ""
        finally:
            self.retrievals += 1
            self.retrieval_ms.append((time.monotonic() - start) * 1000)

    @staticmethod
    def _format_context(
        seeds: list[ScoredMemory], rows_by_id: dict[str, dict[str, Any]]
    ) -> str:
        """Render seeds (highest similarity first) with graph provenance when
        the expansion returned it; fall back to vector payload otherwise —
        an orphan vector (crash between dual-writes) degrades gracefully."""
        lines: list[str] = []
        for s in sorted(seeds, key=lambda x: x.score, reverse=True):
            row = rows_by_id.get(s.memory_id)
            if row:
                where = f" at {row['location']}" if row.get("location") else ""
                who = ""
                involves = [a for a in (row.get("involves") or []) if a]
                if involves:
                    who = f" (involving {', '.join(involves)})"
                lines.append(f"- {row.get('summary', '')}{where}{who}")
            elif s.payload.get("summary"):
                lines.append(f"- {s.payload['summary']}")
        return "\n".join(lines)

    # ------------------------------------------------------------ dual-write

    async def store_memory(
        self, agent_id: str, summary: str, location: str
    ) -> list[dict[str, Any]]:
        """Dual-write. Embeds + upserts to Qdrant HERE (this coroutine runs
        as detached background work), then RETURNS the structural mutation
        for the caller to push onto the GraphWriteQueue — graph writes stay
        the single writer's exclusive job.

        Returns the mutation list ([] if the vector write failed: no vector,
        no graph node — see module docstring on orphan asymmetry)."""
        summary = summary.strip()
        if not summary:
            return []
        memory_id = str(uuid.uuid4())
        t = time.time()
        try:
            vec = (await self._embedder.embed([summary]))[0]
            await self._vectors.upsert_memory(
                memory_id, vec,
                payload={"agent_id": agent_id, "summary": summary,
                         "location": location, "t": t},
            )
        except Exception:
            log.exception("Vector write failed for %s — memory dropped entirely "
                          "(no orphan graph node)", agent_id)
            return []
        self.memories_stored += 1
        return [{
            "type": "memory",
            "memory_id": memory_id,   # ← the join key shared with Qdrant
            "agent_id": agent_id,
            "location": location,
            "summary": summary,
            "t": t,
        }]
