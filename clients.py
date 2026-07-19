"""
Project Panopticon — Phase 1 Production Clients
================================================
clients.py

Real implementations of the `SGLangClient` and `GraphClient` protocols from
tick_engine.py. tick_engine.py itself stays dependency-free; this module owns
the third-party imports (aiohttp, neo4j) and is loaded lazily by run_engine
only when real endpoints are configured.

Contents:
  * RealSGLangClient   — aiohttp streaming client for SGLang's OpenAI-compatible
                         /v1/chat/completions endpoint, with hard server-side
                         abort on asyncio cancellation (frees the remote KV slot).
  * RealNeo4jGraphClient — official async neo4j driver; maps GraphWriteQueue
                         batches into per-type UNWIND statements inside one
                         write transaction, with bounded retry + dead-letter log.
"""

from __future__ import annotations

import asyncio
import json
import os
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncIterator

import aiohttp
import neo4j
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

log = logging.getLogger("panopticon.clients")


# ============================================================================
# 1. SGLang streaming client (aiohttp)
# ============================================================================

class SGLangRequestError(RuntimeError):
    """Non-200 or malformed response from the SGLang router. The dispatcher
    catches this as a generic failure → AIMD multiplicative decrease."""


class RealSGLangClient:
    """Streams chat completions from a MONOLITHIC SGLang instance
    (tensor-parallel across the pod's GPUs; launched e.g. with
    `python -m sglang.launch_server --model Qwen/Qwen3-8B --tp 4`).

    Cancellation contract (the important part)
    ------------------------------------------
    When the ZombieReaper cancels the dispatcher's asyncio.Task, the
    CancelledError surfaces *inside* this generator at its current await
    point (readline). We catch it, force-close the HTTP response — which
    tears down the TCP connection rather than politely draining it — and
    re-raise. SGLang's streaming path watches for client disconnect and
    aborts the in-flight request, immediately releasing its KV cache slot
    and its position in the batch scheduler. Without the hard close, a
    reaped zombie would free our LOCAL AIMD slot while the request kept
    burning GPU decode bandwidth server-side until completion.

    TTFT contract
    -------------
    This client does no buffering/coalescing before the first token: the
    first SSE `data:` line with delta content is yielded immediately, so
    the dispatcher's TTFT measurement (and therefore the AIMD congestion
    signal) reflects true server prefill latency + one network hop.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        max_tokens: int = 96,
        temperature: float = 0.8,
        connect_timeout_s: float = 5.0,
        sock_read_timeout_s: float = 30.0,
    ) -> None:
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        # total=None: streams live as long as tokens flow; sock_read bounds
        # inter-chunk silence so a wedged server can't hold a slot forever
        # even if the Reaper is somehow behind.
        self._timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=connect_timeout_s, sock_read=sock_read_timeout_s
        )
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            # Generous pool: concurrency is governed by the AIMD limiter,
            # not by starving the connector.
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers=headers,
                connector=aiohttp.TCPConnector(limit=256),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def stream_chat(
        self, model: str, messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        session = await self._ensure_session()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if response_format is not None:
            # SGLang structured decoding (xgrammar): the model CANNOT emit
            # tokens violating the schema — action parsing becomes total.
            payload["response_format"] = response_format

        resp = await session.post(self._url, json=payload)
        try:
            if resp.status != 200:
                body = (await resp.text())[:500]
                raise SGLangRequestError(f"SGLang HTTP {resp.status}: {body}")

            # SSE framing: `data: {json}\n`, terminated by `data: [DONE]`.
            while True:
                raw = await resp.content.readline()
                if not raw:                       # server closed the stream
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue                      # comments / keep-alives / blanks
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    log.warning("SGLang sent unparseable SSE line: %.120s", data)
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    yield content

        except asyncio.CancelledError:
            # ZombieReaper (or shutdown) fired. Hard-close the transport so
            # SGLang's disconnect detection aborts the request server-side,
            # freeing the remote KV cache slot NOW — see class docstring.
            resp.close()
            raise
        finally:
            if not resp.closed:
                # Normal completion: release the connection back to the pool.
                resp.release()


# ============================================================================
# 2. Neo4j async client (single-writer consumer)
# ============================================================================

# Each mutation "type" maps to one UNWIND'd, fully parameterized statement.
# The GraphWriteQueue batch is grouped by type, so a 64-item batch commits as
# at most len(_STATEMENTS) queries inside ONE write transaction. Add new
# mutation types here as the agent extraction pipeline grows.
_STATEMENTS: dict[str, str] = {
    "observed": """
        UNWIND $rows AS row
        MERGE (a:Agent {id: row.agent_id})
        MERGE (l:Location {id: row.location})
        MERGE (a)-[r:OBSERVED]->(l)
        SET r.at = row.t, r.summary = row.summary
    """,
    # Dual-write structural half: Memory.id == the Qdrant point UUID.
    # MERGE on the UUID keeps dead-letter replay idempotent.
    "memory": """
        UNWIND $rows AS row
        MERGE (a:Agent {id: row.agent_id})
        MERGE (l:Location {id: row.location})
        MERGE (m:Memory {id: row.memory_id})
        SET m.summary = row.summary, m.t = row.t
        MERGE (a)-[:REMEMBERS]->(m)
        MERGE (m)-[:AT]->(l)
    """,
    # ---- Phase 4: consolidation mutation types ----------------------------
    # ALL statements below are (a) idempotent — dead-letter replay and daemon
    # retry must converge to the same graph state — and (b) deadlock-free by
    # construction: they only ever execute inside THE single writer.

    # The long-term node. Deterministic memory_id (uuid5 over sorted source
    # ids) means a replayed/duplicated consolidation MERGEs onto itself.
    "consolidated_memory": """
        UNWIND $rows AS row
        MERGE (a:Agent {id: row.agent_id})
        MERGE (l:Location {id: row.location})
        MERGE (m:Memory {id: row.memory_id})
        SET m:Consolidated, m.summary = row.summary, m.t = row.t,
            m.source_count = row.source_count
        MERGE (a)-[:REMEMBERS]->(m)
        MERGE (m)-[:AT]->(l)
        WITH m, row
        UNWIND row.source_ids AS src
        OPTIONAL MATCH (s:Memory {id: src})
        FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END |
            MERGE (m)-[:DERIVED_FROM]->(s))
    """,

    # Tombstone: REMEMBERS → ARCHIVED_REMEMBERS. `DELETE r` on a null
    # OPTIONAL MATCH is a Cypher no-op, so replay after the edge is already
    # transitioned converges cleanly. coalesce() preserves the FIRST
    # archived_at across replays.
    "tombstone": """
        UNWIND $rows AS row
        MATCH (a:Agent {id: row.agent_id})
        MATCH (m:Memory {id: row.memory_id})
        SET m.archived_at = coalesce(m.archived_at, row.archived_at),
            m.consolidated_into = row.consolidated_into
        MERGE (a)-[:ARCHIVED_REMEMBERS]->(m)
        WITH a, m
        OPTIONAL MATCH (a)-[r:REMEMBERS]->(m)
        DELETE r
    """,

    # Affinity update. Deltas are NOT naturally idempotent, so each carries
    # the consolidation event_id: the FOREACH guard applies a given event's
    # delta at most once, and the result is clamped to [-1, 1].
    "relationship": """
        UNWIND $rows AS row
        MERGE (a:Agent {id: row.agent_id})
        MERGE (b:Agent {id: row.other_agent_id})
        MERGE (a)-[r:RELATES_TO]->(b)
        FOREACH (_ IN CASE WHEN coalesce(r.last_event, '') <> row.event_id
                      THEN [1] ELSE [] END |
            SET r.affinity = CASE
                    WHEN coalesce(r.affinity, 0.0) + row.affinity_delta > 1.0
                        THEN 1.0
                    WHEN coalesce(r.affinity, 0.0) + row.affinity_delta < -1.0
                        THEN -1.0
                    ELSE coalesce(r.affinity, 0.0) + row.affinity_delta
                END,
                r.last_event = row.event_id,
                r.last_reason = row.reason,
                r.updated_at = row.t)
    """,

    # Belief: last-write-wins on importance — inherently idempotent.
    "belief": """
        UNWIND $rows AS row
        MERGE (a:Agent {id: row.agent_id})
        MERGE (b:Belief {agent_id: row.agent_id, text: row.belief})
        SET b.importance = row.importance, b.updated_at = row.t
        MERGE (a)-[:HOLDS]->(b)
    """,
}

# 1-hop neighborhood expansion for hybrid retrieval (memory.py step 2).
# Read-only: executes OUTSIDE the single-writer queue via execute_read.
_FETCH_MEMORIES = """
    UNWIND $ids AS mid
    MATCH (m:Memory {id: mid})
    OPTIONAL MATCH (a:Agent)-[:REMEMBERS]->(m)
    OPTIONAL MATCH (m)-[:AT]->(l:Location)
    OPTIONAL MATCH (m)-[:INVOLVES]->(other:Agent)
    RETURN m.id AS id, m.summary AS summary, m.t AS t,
           a.id AS agent_id, l.id AS location,
           collect(DISTINCT other.id) AS involves
"""

# Consolidation sweep input: hot (REMEMBERS-linked) episodic memories only.
# :Consolidated nodes are excluded — we never re-consolidate consolidations
# in a sweep — and min-age keeps freshly-formed memories out of clumps so
# in-flight interactions aren't summarized mid-scene.
# Goal-seek input: an agent's strongest held beliefs (Phase 8, Q3 daemon).
_FETCH_BELIEFS = """
    MATCH (a:Agent {id: $agent_id})-[:HOLDS]->(b:Belief)
    RETURN b.text AS text, b.importance AS importance
    ORDER BY b.importance DESC, b.updated_at DESC
    LIMIT $limit
"""

_FETCH_UNCONSOLIDATED = """
    MATCH (a:Agent {id: $agent_id})-[:REMEMBERS]->(m:Memory)
    WHERE NOT m:Consolidated
      AND m.archived_at IS NULL
      AND m.t < $cutoff_t
    OPTIONAL MATCH (m)-[:AT]->(l:Location)
    RETURN m.id AS id, m.summary AS summary, m.t AS t, l.id AS location
    ORDER BY m.t ASC
    LIMIT $limit
"""


class RealNeo4jGraphClient:
    """Executes GraphWriteQueue batches against Neo4j via the official async
    driver.

    Concurrency note: this client is only ever invoked by THE single writer
    coroutine (GraphWriteQueue.run_writer), so writes are strictly serialized.
    That structural guarantee is why plain `session.execute_write` suffices —
    no dummy-property locks, no bipartite batch coloring. Even the classic
    hot-node case (fifty agents MERGE-ing onto one `central_plaza` Location)
    cannot deadlock, because there is never a second writer to deadlock with.

    Failure policy: `execute_write` already retries transient cluster errors
    internally (managed transactions). We add a bounded OUTER retry with
    exponential backoff + jitter for connectivity-level failures
    (ServiceUnavailable / SessionExpired), and after `max_retries` the batch
    is appended to a JSONL dead-letter file so simulation memory loss is
    inspectable and replayable instead of silent.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str = "neo4j",
        max_retries: int = 3,
        backoff_base_s: float = 0.25,
        connection_timeout_s: float = 5.0,
        driver_retry_window_s: float = 3.0,
        dead_letter_path: str | Path | None = None,
    ) -> None:
        # max_transaction_retry_time caps the driver's INTERNAL managed-tx
        # retry loop (default 30s!). Left at default, a Neo4j outage would
        # wedge the single writer for max_retries × 30s per batch, backing up
        # the entire mutation queue. We keep it short — the outer loop below
        # owns retry policy, and the DLQ owns durability.
        self._driver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=connection_timeout_s,
            max_transaction_retry_time=driver_retry_window_s,
        )
        self._database = database
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        # Default resolves under DEAD_LETTER_DIR (set by docker-compose to a
        # durable mount) — never silently into the current checkout.
        if dead_letter_path is None:
            dead_letter_path = (Path(os.environ.get("DEAD_LETTER_DIR", "."))
                                / "panopticon_dead_letter.jsonl")
        self._dead_letter_path = Path(dead_letter_path)
        # Telemetry
        self.batches_written = 0
        self.mutations_written = 0
        self.dead_lettered = 0

    async def close(self) -> None:
        await self._driver.close()

    async def verify_connectivity(self) -> None:
        """Optional fail-fast probe for startup."""
        await self._driver.verify_connectivity()

    async def fetch_memories(self, memory_ids: list[str]) -> list[dict[str, Any]]:
        """1-hop expansion of seed memory UUIDs for hybrid retrieval.
        READ path: runs concurrently with everything, bypassing the
        single-writer queue (which serializes mutations only). Sits inside
        the retrieval latency budget, so no retry loop — a failed read
        fails open in MemoryService."""

        async def _tx(tx: neo4j.AsyncManagedTransaction) -> list[dict[str, Any]]:
            result = await tx.run(_FETCH_MEMORIES, ids=memory_ids)
            return [dict(record) async for record in result]

        async with self._driver.session(database=self._database) as session:
            return await session.execute_read(_tx)

    async def fetch_beliefs(self, agent_id: str, limit: int
                            ) -> list[dict[str, Any]]:
        """Read path for the GoalSeekDaemon; concurrent-safe, no writer."""

        async def _tx(tx: neo4j.AsyncManagedTransaction) -> list[dict[str, Any]]:
            result = await tx.run(_FETCH_BELIEFS, agent_id=agent_id,
                                  limit=limit)
            return [dict(record) async for record in result]

        async with self._driver.session(database=self._database) as session:
            return await session.execute_read(_tx)

    async def fetch_unconsolidated_memories(
        self, agent_id: str, min_age_s: float, limit: int
    ) -> list[dict[str, Any]]:
        """Consolidation sweep input (read path, bypasses the writer queue).
        Returns oldest-first hot episodic memories for one agent."""
        cutoff = time.time() - min_age_s

        async def _tx(tx: neo4j.AsyncManagedTransaction) -> list[dict[str, Any]]:
            result = await tx.run(_FETCH_UNCONSOLIDATED, agent_id=agent_id,
                                  cutoff_t=cutoff, limit=limit)
            return [dict(record) async for record in result]

        async with self._driver.session(database=self._database) as session:
            return await session.execute_read(_tx)

    @staticmethod
    def group_rows(mutations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Pure function: bucket a mixed batch by mutation type, dropping
        (and logging) any type we don't have a statement for."""
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for m in mutations:
            mtype = m.get("type")
            if mtype in _STATEMENTS:
                grouped[mtype].append({k: v for k, v in m.items() if k != "type"})
            else:
                log.error("Unknown mutation type %r — skipped: %.200s", mtype, m)
        return grouped

    async def execute_batch(self, mutations: list[dict[str, Any]]) -> None:
        grouped = self.group_rows(mutations)
        if not grouped:
            return

        async def _tx(tx: neo4j.AsyncManagedTransaction) -> None:
            for mtype, rows in grouped.items():
                await tx.run(_STATEMENTS[mtype], rows=rows)

        attempt = 0
        while True:
            try:
                async with self._driver.session(database=self._database) as session:
                    await session.execute_write(_tx)
                self.batches_written += 1
                self.mutations_written += len(mutations)
                return
            except (ServiceUnavailable, SessionExpired, OSError, Neo4jError) as exc:
                attempt += 1
                if attempt > self._max_retries:
                    await self._dead_letter(mutations, exc)
                    return
                delay = self._backoff_base_s * (2 ** (attempt - 1))
                delay += random.uniform(0, delay)  # full jitter
                log.warning(
                    "Neo4j batch commit failed (attempt %d/%d, %s) — retrying in %.2fs",
                    attempt, self._max_retries, type(exc).__name__, delay,
                )
                await asyncio.sleep(delay)

    async def _dead_letter(self, mutations: list[dict[str, Any]], exc: Exception) -> None:
        """Persist the failed batch as JSON lines for offline replay. File I/O
        is tiny and rare; if it ever matters, hop it to a thread executor."""
        self.dead_lettered += len(mutations)
        record = {
            "ts": time.time(),
            "error": f"{type(exc).__name__}: {exc}",
            "mutations": mutations,
        }
        try:
            with self._dead_letter_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            log.error(
                "Neo4j batch dead-lettered (%d mutations) → %s",
                len(mutations), self._dead_letter_path,
            )
        except OSError:
            log.exception("FAILED TO DEAD-LETTER %d mutations — data lost", len(mutations))
