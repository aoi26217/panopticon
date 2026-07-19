"""
Project Panopticon — Phase 0 "Walking Skeleton"
================================================
tick_engine.py

Core orchestration engine for a real-time, multi-agent spatial simulation.

Architecture (2026 patterns, deliberately boring where boring wins):

  ┌──────────────────────┐   enqueue (Q0/Q1/Q2)   ┌─────────────────────┐
  │  Tick Loop @ 20 Hz   │ ─────────────────────► │   MLFQ Scheduler    │
  │  (physics, triggers) │                        │  + Zombie Reaper    │
  └──────────────────────┘                        └──────────┬──────────┘
            │ never blocks on inference                      │ dequeue
            ▼                                                ▼
  ┌──────────────────────┐                        ┌─────────────────────┐
  │  Graph Write Queue   │ ◄──── mutations ────── │  Inference Workers  │
  │  (SINGLE writer)     │                        │  gated by AIMD      │
  └──────────┬───────────┘                        └──────────┬──────────┘
             ▼ batched, serialized                           ▼ streamed
        Neo4j (mocked)                          SGLang (mocked) → LiveKit stub

Hard rules enforced in this file:
  1. The 20 Hz tick loop NEVER awaits inference or DB I/O.
  2. Monolithic SGLang (tensor parallel) — no RDMA/disaggregation. 8B–32B model.
  3. No busy-wait rate control. AIMD is implemented on asyncio.Condition
     (a Condition is the correct primitive here: asyncio.Semaphore has a
     fixed capacity, and AIMD requires a *dynamic* limit).
  4. MLFQ with quantum-based demotion + a Zombie Reaper that cancels tasks
     stuck > 30 s and returns their slot to the limiter.
  5. All Neo4j mutations flow through ONE writer coroutine consuming an
     asyncio.Queue → deadlocks are structurally impossible, batching is free.
  6. WebRTC/TTS are stubbed. Phase 1 will use LiveKit (SFU + Python agents
     SDK, not raw data channels) and client-side WASM viseme inference.

Run a self-contained demo (no external deps):
    python tick_engine.py --demo-seconds 12 --agents 8
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, AsyncIterator, Protocol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-14s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("panopticon")


# ============================================================================
# 0. Configuration
# ============================================================================

@dataclass(frozen=True)
class EngineConfig:
    # --- Timing ---------------------------------------------------------
    tick_hz: float = 20.0                 # spatial/physics tick rate
    reasoning_cooldown_s: float = 0.5     # per-agent LLM cadence cap (~2 Hz)
    ttft_budget_ms: float = 500.0         # latency SLO for Time-To-First-Token

    # --- AIMD rate control ------------------------------------------------
    aimd_initial_limit: int = 8
    aimd_min_limit: int = 1
    aimd_max_limit: int = 64
    aimd_backoff_factor: float = 0.75     # multiplicative decrease
    aimd_success_window: int = 25         # successes required per +1 increase

    # --- MLFQ -------------------------------------------------------------
    # Execution quantum per level (seconds). A task whose *wall time* exceeds
    # its level's quantum is demoted one level on its next scheduling.
    mlfq_quanta_s: tuple[float, float, float, float] = (1.0, 3.0, 10.0, 30.0)

    # --- Zombie Reaper ------------------------------------------------------
    reaper_scan_interval_s: float = 5.0
    zombie_timeout_s: float = 30.0

    # --- Single-writer graph queue -----------------------------------------
    db_queue_maxsize: int = 10_000
    db_batch_max: int = 64
    db_batch_linger_s: float = 0.05       # wait up to 50 ms to fill a batch

    # --- World -------------------------------------------------------------
    world_size: float = 100.0
    proximity_radius: float = 4.0
    steering_speed: float = 2.5           # units/s while executing move_towards
    interaction_radius: float = 2.0       # reach for pick_up / give / use
    arrival_radius: float = 1.2           # command completes inside this range
    command_ttl_s: float = 8.0            # stale commands self-expire
    model_name: str = "Qwen/Qwen3-8B"     # 8B class: fits the 500 ms TTFT SLO
                                          # on a monolithic TP SGLang deployment


# ============================================================================
# 1. Task model + MLFQ Scheduler
# ============================================================================

class QueueLevel(IntEnum):
    INTERACTIVE = 0   # Q0: agent-to-agent speech, collision reactions
    COGNITIVE = 1     # Q1: internal monologue, pathfinding, planning
    BACKGROUND = 2    # Q2: memory consolidation, GraphRAG extraction
    GOAL = 3          # Q3: slow intrinsic-motivation formulation (Phase 8)


_task_ids = itertools.count(1)


@dataclass
class AgentTask:
    agent_id: str
    kind: str                              # "dialogue" | "reasoning" | "memory_extraction"
    messages: list[dict[str, str]]
    level: QueueLevel
    task_id: int = field(default_factory=lambda: next(_task_ids))
    enqueued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class MLFQScheduler:
    """Three-level Multi-Level Feedback Queue.

    - Strict priority dequeue: Q0 > Q1 > Q2.
    - Feedback: a task that overruns its level's quantum is demoted, so a
      long-winded "cognitive" generation cannot starve interactive speech.
    - No busy-waiting: consumers block on an asyncio.Condition until work
      arrives.
    """

    def __init__(self, cfg: EngineConfig) -> None:
        self._cfg = cfg
        self._queues: list[deque[AgentTask]] = [deque(), deque(), deque(),
                                                deque()]
        self._cond = asyncio.Condition()
        self._closed = False

    async def submit(self, task: AgentTask) -> None:
        async with self._cond:
            self._queues[task.level].append(task)
            self._cond.notify()

    def submit_nowait(self, task: AgentTask) -> None:
        """Lock-free fast path for the tick loop (single-threaded asyncio:
        deque.append is atomic; we schedule the notify without awaiting so
        the 20 Hz loop never blocks on scheduler lock contention)."""
        self._queues[task.level].append(task)
        asyncio.get_running_loop().call_soon(self._notify_soon)

    def _notify_soon(self) -> None:
        # Fire-and-forget wakeup for blocked consumers.
        asyncio.ensure_future(self._wake())

    async def _wake(self) -> None:
        async with self._cond:
            self._cond.notify()

    async def next_task(self) -> AgentTask | None:
        """Blocks until a task is available. Returns None on shutdown."""
        async with self._cond:
            await self._cond.wait_for(lambda: self._closed or any(self._queues))
            if self._closed and not any(self._queues):
                return None
            for q in self._queues:            # strict priority
                if q:
                    task = q.popleft()
                    task.started_at = time.monotonic()
                    return task
        return None  # unreachable, keeps type-checkers happy

    def demote(self, task: AgentTask) -> None:
        """Feedback step: called when a task overran its quantum. The demoted
        level is applied if/when the task (or its follow-up) is re-enqueued."""
        if task.level < QueueLevel.GOAL:
            old = task.level
            task.level = QueueLevel(task.level + 1)
            log.debug("MLFQ demote task=%s %s→%s", task.task_id, old.name, task.level.name)

    def quantum_for(self, level: QueueLevel) -> float:
        return self._cfg.mlfq_quanta_s[level]

    async def close(self) -> None:
        async with self._cond:
            self._closed = True
            self._cond.notify_all()

    def depths(self) -> tuple[int, int, int, int]:
        return tuple(len(q) for q in self._queues)  # type: ignore[return-value]


# ============================================================================
# 2. AIMD Concurrency Limiter (Condition-based dynamic semaphore)
# ============================================================================

class AIMDLimiter:
    """TCP-style Additive-Increase / Multiplicative-Decrease admission control
    for the SGLang cluster.

    Why asyncio.Condition instead of asyncio.Semaphore: a Semaphore's capacity
    is fixed at construction. AIMD needs the ceiling itself to move. We hold
    the invariant `active <= floor(limit)` and park waiters on the Condition —
    zero busy-waiting, waiters are woken exactly when capacity appears (a slot
    is released OR additive increase raises the ceiling).
    """

    def __init__(self, cfg: EngineConfig) -> None:
        self._cfg = cfg
        self._limit = float(cfg.aimd_initial_limit)
        self._active = 0
        self._successes = 0
        self._cond = asyncio.Condition()
        # Telemetry
        self.total_admitted = 0
        self.total_failures = 0
        self.backoff_events = 0
        # Exact admission-invariant check, measured AT the admission point
        # (post-increment active must never exceed the instantaneous ceiling).
        # Guaranteed by the Condition predicate; this counter exists so a
        # future refactor that breaks the lock discipline fails loudly in the
        # crucible instead of silently OOMing a GPU server.
        self.admission_violations = 0

    @property
    def limit(self) -> int:
        return int(self._limit)

    @property
    def active(self) -> int:
        return self._active

    async def acquire(self) -> None:
        async with self._cond:
            await self._cond.wait_for(lambda: self._active < int(self._limit))
            self._active += 1
            self.total_admitted += 1
            if self._active > int(self._limit):
                self.admission_violations += 1

    async def release(self) -> None:
        async with self._cond:
            self._active = max(0, self._active - 1)
            self._cond.notify_all()

    async def record_success(self, ttft_ms: float) -> None:
        """Called once per generation with the observed TTFT."""
        if ttft_ms >= self._cfg.ttft_budget_ms:
            await self.record_failure(reason=f"TTFT {ttft_ms:.0f}ms over budget")
            return
        async with self._cond:
            self._successes += 1
            if self._successes >= self._cfg.aimd_success_window:
                self._successes = 0
                if self._limit < self._cfg.aimd_max_limit:
                    self._limit = min(self._cfg.aimd_max_limit, self._limit + 1.0)
                    log.info("AIMD ↑ additive increase → limit=%d", int(self._limit))
                    self._cond.notify_all()   # new headroom: wake waiters

    async def record_failure(self, reason: str = "error") -> None:
        """Congestion signal: latency spike, cache saturation, or transport
        error. Halve-ish the window; in-flight requests finish naturally,
        only NEW admissions are throttled."""
        async with self._cond:
            old = int(self._limit)
            self._limit = max(float(self._cfg.aimd_min_limit),
                              self._limit * self._cfg.aimd_backoff_factor)
            self._successes = 0
            self.total_failures += 1
            if int(self._limit) != old:
                self.backoff_events += 1
                log.warning("AIMD ↓ multiplicative decrease (%s) limit %d→%d",
                            reason, old, int(self._limit))


# ============================================================================
# 3. Zombie Reaper
# ============================================================================

@dataclass
class _TaskRecord:
    aio_task: asyncio.Task
    task: AgentTask
    registered_at: float


class ZombieReaper:
    """Scans the in-flight task registry every `reaper_scan_interval_s` and
    cancels any asyncio task that has held an execution slot longer than
    `zombie_timeout_s`. Slot release is guaranteed by the dispatcher's
    try/finally (cancellation unwinds through it), so a reaped zombie always
    returns its AIMD slot to the pool."""

    def __init__(self, cfg: EngineConfig) -> None:
        self._cfg = cfg
        self._registry: dict[int, _TaskRecord] = {}
        self.reaped_total = 0

    def register(self, task: AgentTask, aio_task: asyncio.Task) -> None:
        self._registry[task.task_id] = _TaskRecord(aio_task, task, time.monotonic())

    def unregister(self, task: AgentTask) -> None:
        self._registry.pop(task.task_id, None)

    @property
    def in_flight(self) -> int:
        return len(self._registry)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._cfg.reaper_scan_interval_s)
            now = time.monotonic()
            for rec in list(self._registry.values()):
                age = now - rec.registered_at
                if age > self._cfg.zombie_timeout_s and not rec.aio_task.done():
                    self.reaped_total += 1
                    log.warning("REAPER cancelling zombie task=%s agent=%s age=%.1fs",
                                rec.task.task_id, rec.task.agent_id, age)
                    rec.aio_task.cancel()
                    # unregister happens in the dispatcher's finally-block


# ============================================================================
# 4. Client Protocols + Mocks (swap points for Phase 1)
# ============================================================================

class SGLangClient(Protocol):
    """Implemented by clients.RealSGLangClient (aiohttp) against the SGLang
    OpenAI-compatible endpoint (POST /v1/chat/completions, stream=True).
    Monolithic instance, tensor-parallel across the pod's GPUs — no
    prefill/decode disaggregation. RadixAttention prefix caching works
    automatically as long as prompts follow the SHARED_WORLD_PREFIX contract
    documented above the constant (byte-stable prefix, dynamic content last)."""

    def stream_chat(self, model: str, messages: list[dict[str, str]],
                    response_format: dict[str, Any] | None = None,
                    ) -> AsyncIterator[str]: ...


class GraphClient(Protocol):
    """Implemented by clients.RealNeo4jGraphClient (official async driver).
    Because all writes are serialized through the single writer, managed
    `execute_write` transactions are safe — no dummy-property locks, no
    bipartite coloring. Telemetry attrs are part of the contract so the
    telemetry loop stays implementation-agnostic."""

    batches_written: int
    mutations_written: int

    async def execute_batch(self, mutations: list[dict[str, Any]]) -> None: ...
    async def fetch_memories(self, memory_ids: list[str]) -> list[dict[str, Any]]: ...


class ActionRouterProto(Protocol):
    """Implemented by avatar.ActionRouter. response_format() returns the
    SGLang json_schema payload; intercept() parses one generation, routes
    any action onto the TickEngine command board, and returns the speech."""

    def response_format(self) -> dict[str, Any]: ...
    def intercept(self, agent_id: str, raw: str) -> str: ...


class MemoryStore(Protocol):
    """Implemented by memory.MemoryService (Phase 3). retrieve_context is
    budget-bounded and fail-open (returns "" rather than ever blowing the
    latency budget); store_memory performs the vector half of the dual-write
    and returns the structural mutations for the GraphWriteQueue."""

    async def retrieve_context(self, agent_id: str, query_text: str) -> str: ...
    async def store_memory(
        self, agent_id: str, summary: str, location: str
    ) -> list[dict[str, Any]]: ...


class MediaPublisher(Protocol):
    """Implemented by media.RealLiveKitPublisher (Phase 2). `publish_tokens`
    is the per-token hot path and must return near-instantly; `end_utterance`
    signals end-of-generation so partial sentences get flushed to TTS."""

    async def publish_tokens(self, agent_id: str, text: str) -> None: ...
    async def end_utterance(self, agent_id: str) -> None: ...


class MockSGLangClient:
    """Simulates a monolithic SGLang instance serving an 8B model.
    TTFT is mostly well under budget with occasional congestion spikes so the
    AIMD path is exercised end-to-end in the demo."""

    def __init__(self, spike_prob: float = 0.04, hang_prob: float = 0.0) -> None:
        self._spike_prob = spike_prob
        self._hang_prob = hang_prob  # set > 0 in tests to exercise the Reaper

    async def stream_chat(self, model: str, messages: list[dict[str, str]],
                          response_format: dict[str, Any] | None = None,
                          ) -> AsyncIterator[str]:
        if random.random() < self._hang_prob:
            await asyncio.sleep(10_000)  # simulated wedged request → Reaper food
        # Simulated TTFT (prefill): fast thanks to Radix prefix cache hits,
        # with rare multi-hundred-ms congestion spikes.
        ttft = random.uniform(0.06, 0.28)
        if random.random() < self._spike_prob:
            ttft = random.uniform(0.55, 0.9)
        await asyncio.sleep(ttft)
        for token in ["*glances", " over*", " Did", " you", " hear", " that?"]:
            yield token
            await asyncio.sleep(random.uniform(0.005, 0.02))  # decode cadence


class MockNeo4jClient:
    """Stands in for the async Neo4j driver. Logs batch sizes so you can see
    the single-writer coalescing in action."""

    def __init__(self) -> None:
        self.batches_written = 0
        self.mutations_written = 0

    async def execute_batch(self, mutations: list[dict[str, Any]]) -> None:
        await asyncio.sleep(0.005)  # simulated round-trip
        self.batches_written += 1
        self.mutations_written += len(mutations)
        log.debug("NEO4J batch commit: %d mutations", len(mutations))

    async def fetch_memories(self, memory_ids: list[str]) -> list[dict[str, Any]]:
        return []  # mock graph has no memories; retrieval falls back to payloads


class LiveKitPublisher:
    """No-op stub (mock fallback). The production pipeline lives in
    media.RealLiveKitPublisher: LiveKit SFU egress (never raw WebRTC data
    channels), F5-TTS microservice, per-agent audio tracks — and audio ONLY:
    visemes are computed client-side via WASM tapping the MediaStream."""

    def __init__(self) -> None:
        self.tokens_published = 0

    async def publish_tokens(self, agent_id: str, text: str) -> None:
        self.tokens_published += 1

    async def end_utterance(self, agent_id: str) -> None:
        pass


# ============================================================================
# 5. Single-Writer Graph Mutation Queue
# ============================================================================

class GraphWriteQueue:
    """The deadlock killer.

    Every subsystem that wants to mutate Neo4j calls `submit()`; exactly ONE
    consumer coroutine (`run_writer`) drains the queue and commits batches.
    With a single writer there is no lock contention and therefore no
    Neo.TransientError.Transaction.DeadlockDetected — by construction, not by
    cleverness. Batching (up to `db_batch_max`, with a short linger) turns
    fifty agents' trickle of observations into efficient bulk commits.
    """

    def __init__(self, cfg: EngineConfig, graph: GraphClient) -> None:
        self._cfg = cfg
        self._graph = graph
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=cfg.db_queue_maxsize
        )
        self.submitted = 0
        self.dropped = 0

    async def submit(self, mutation: dict[str, Any]) -> None:
        """Backpressure-aware submit. Background memory writes are droppable
        under extreme load (graceful degradation beats an unbounded queue)."""
        try:
            self._queue.put_nowait(mutation)
            self.submitted += 1
        except asyncio.QueueFull:
            self.dropped += 1
            log.warning("GraphWriteQueue full — dropping background mutation")

    async def close(self) -> None:
        await self._queue.put(None)  # sentinel

    async def run_writer(self) -> None:
        """THE single writer. Do not spawn a second copy of this coroutine —
        the whole deadlock guarantee rests on its uniqueness."""
        while True:
            first = await self._queue.get()
            if first is None:
                return
            batch: list[dict[str, Any]] = [first]

            # Linger briefly to coalesce a batch, then drain what's ready.
            deadline = time.monotonic() + self._cfg.db_batch_linger_s
            while len(batch) < self._cfg.db_batch_max:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    await self._flush(batch)
                    return
                batch.append(item)

            await self._flush(batch)

    async def _flush(self, batch: list[dict[str, Any]]) -> None:
        try:
            await self._graph.execute_batch(batch)
        except Exception:
            # Phase 1: bounded retry with jitter, then dead-letter log.
            log.exception("Graph batch write failed (size=%d)", len(batch))


# ============================================================================
# 6. Inference Dispatch (AIMD-gated, Reaper-registered)
# ============================================================================

class InferenceDispatcher:
    def __init__(
        self,
        cfg: EngineConfig,
        scheduler: MLFQScheduler,
        limiter: AIMDLimiter,
        reaper: ZombieReaper,
        sglang: SGLangClient,
        media: MediaPublisher,
        graph_queue: GraphWriteQueue,
        memory: MemoryStore | None = None,
        action_router: "ActionRouterProto | None" = None,
        navigation: Any | None = None,
    ) -> None:
        self._cfg = cfg
        self._scheduler = scheduler
        self._limiter = limiter
        self._reaper = reaper
        self._sglang = sglang
        self._media = media
        self._graph_queue = graph_queue
        self._memory = memory
        self._actions = action_router
        self._navigation = navigation
        # GOAL BOARD (Phase 8, written by the Q3 GoalSeekDaemon): agent_id →
        # standing-goal string. Injected as the FIRST, byte-stable segment of
        # every dialogue leaf — RadixAttention segment [3], the per-agent
        # branch. The shared root is untouched; changing an agent's goal
        # re-prefills exactly ONE branch, never the tree.
        self.goal_board: dict[str, str] = {}
        # WORLD CONDITIONS (Phase 10, written atomically by the Director):
        # RadixAttention segment [2] — a short, slow-rotating suffix appended
        # AFTER the immutable [1] root in the system message. Rotating it
        # re-prefills ~12 tokens per branch, never the root. None = omitted
        # entirely (byte-identical to pre-Phase-10 prompts).
        self.world_conditions: str | None = None
        self._inflight: set[asyncio.Task] = set()
        # Detached dual-write tasks: strong refs prevent mid-flight GC.
        self._background_writes: set[asyncio.Task] = set()
        # Telemetry
        self.completed = 0
        self.cancelled = 0
        self.ttft_samples: deque[float] = deque(maxlen=500)

    async def run_worker(self) -> None:
        """Pulls from the MLFQ and fires each generation as an independent
        asyncio task. The worker loop itself never blocks on model latency —
        admission control happens inside `_execute` via the AIMD limiter."""
        while True:
            task = await self._scheduler.next_task()
            if task is None:
                return  # scheduler closed
            aio_task = asyncio.create_task(
                self._execute(task), name=f"gen-{task.task_id}"
            )
            self._inflight.add(aio_task)
            aio_task.add_done_callback(self._inflight.discard)
            self._reaper.register(task, aio_task)

    async def _execute(self, task: AgentTask) -> None:
        if task.kind == "pathfind":
            # CPU planning, not GPU inference: rides the MLFQ (Q1, preempted
            # by live speech) but is governed by the planner bulkhead in
            # NavigationService, NOT by AIMD — GPU admission control stays
            # calibrated on TTFT alone. Fail-open: any error leaves the
            # already-running straight-line steering untouched.
            try:
                if self._navigation is not None:
                    await self._navigation.handle_task(task)
            except Exception:
                log.exception("Pathfind task failed for %s (straight-line "
                              "steering continues)", task.agent_id)
            finally:
                self._reaper.unregister(task)
                self.completed += 1
            return

        # ---- MEMORY PRE-FETCH: runs BEFORE acquiring an AIMD slot ---------
        # DB/embedding I/O overlaps queue wait instead of occupying GPU
        # admission; the tick loop is on another coroutine entirely.
        # retrieve_context is internally budget-bounded (~100 ms, fail-open),
        # so a sick memory stack can delay THIS request slightly but can
        # never wedge it — and it never holds an inference slot.
        if task.kind == "dialogue":
            goal = self.goal_board.get(task.agent_id)
            if goal:
                # Stable-first leaf ordering: [goal][memories][event] — the
                # goal segment is identical across this agent's requests
                # until the Q3 daemon replaces it, so it extends the warm
                # cached branch instead of forking it.
                task.messages[-1]["content"] = (
                    f"[Your standing personal goal: {goal}]\n\n"
                    + task.messages[-1]["content"])

        if self._memory is not None and task.kind == "dialogue":
            context = await self._memory.retrieve_context(
                task.agent_id, task.messages[-1]["content"]
            )
            if context:
                # LEAF-ONLY injection: append to the final user message. The
                # system message (SHARED_WORLD_PREFIX) must stay byte-stable
                # — memories are per-agent, per-moment, and would fork the
                # RadixAttention tree at the root if they touched the prefix.
                task.messages[-1]["content"] += (
                    "\n\nRelevant memories:\n" + context
                )

        await self._limiter.acquire()          # ← blocks HERE, not in tick loop
        start = time.monotonic()
        ttft_ms: float | None = None
        generated: list[str] = []
        # Structured agency: when an action router is attached, dialogue
        # generations are schema-constrained JSON. We must BUFFER instead of
        # streaming raw tokens to TTS (nobody wants to hear braces read
        # aloud). Cost: TTFA rises by one short generation (~outputs are one
        # sentence); benefit: parsing is total under xgrammar enforcement.
        structured = self._actions is not None and task.kind == "dialogue"
        response_format = (self._actions.response_format() if structured
                           else task.payload.get("response_format"))
        if task.kind == "dialogue":
            # Assemble system message = [1] root + optional [2] conditions.
            # Byte-stable within a rotation epoch; the root never varies.
            base = (EMBODIED_WORLD_PREFIX if structured
                    else task.messages[0]["content"])
            if self.world_conditions:
                base = (base + "\n\nCurrent world conditions: "
                        + self.world_conditions)
            if structured or self.world_conditions:
                task.messages[0] = {"role": "system", "content": base}
        try:
            if response_format is not None:
                stream = self._sglang.stream_chat(
                    self._cfg.model_name, task.messages,
                    response_format=response_format)
            else:   # kwarg omitted: older two-arg clients stay compatible
                stream = self._sglang.stream_chat(self._cfg.model_name,
                                                  task.messages)
            async for chunk in stream:
                if ttft_ms is None:
                    ttft_ms = (time.monotonic() - start) * 1000.0
                    self.ttft_samples.append(ttft_ms)
                    await self._limiter.record_success(ttft_ms)
                if task.kind == "dialogue" and not structured:
                    await self._media.publish_tokens(task.agent_id, chunk)
                else:
                    generated.append(chunk)    # JSON / summary / consolidation

            if task.kind == "dialogue":
                if structured:
                    say = self._actions.intercept(task.agent_id,
                                                  "".join(generated))
                    if say:
                        await self._media.publish_tokens(task.agent_id, say)
                # Flush any unterminated sentence tail into the TTS pipeline.
                await self._media.end_utterance(task.agent_id)

            # Daemons (e.g. consolidation) submit tasks carrying a future so
            # they can await the generation through the MLFQ + AIMD + Reaper
            # machinery instead of calling SGLang around it.
            fut: asyncio.Future | None = task.payload.get("result_future")
            if fut is not None and not fut.done():
                fut.set_result("".join(generated))

            # MLFQ feedback: overran this level's quantum → demote follow-ups.
            elapsed = time.monotonic() - start
            if elapsed > self._scheduler.quantum_for(task.level):
                self._scheduler.demote(task)

            # Background tasks feed the single-writer graph queue.
            if task.kind == "memory_extraction":
                # DUAL-WRITE, detached: _execute returns (and its finally
                # releases the AIMD slot) without waiting on embedding/Qdrant
                # /Neo4j I/O. The GPU slot is for the GPU.
                summary = "".join(generated).strip() or task.payload.get(
                    "observation", ""
                )
                write = asyncio.create_task(
                    self._dual_write(task.agent_id, summary,
                                     task.payload.get("location", "central_plaza")),
                    name=f"dualwrite-{task.task_id}",
                )
                self._background_writes.add(write)
                write.add_done_callback(self._background_writes.discard)
            self.completed += 1

        except asyncio.CancelledError:
            # Reaped zombie or shutdown; RealSGLangClient aborts server-side.
            self.cancelled += 1
            fut = task.payload.get("result_future")
            if fut is not None and not fut.done():
                fut.cancel()
            raise
        except Exception as exc:
            await self._limiter.record_failure(reason=type(exc).__name__)
            fut = task.payload.get("result_future")
            if fut is not None and not fut.done():
                fut.set_exception(exc)
            log.exception("Inference failed task=%s agent=%s", task.task_id, task.agent_id)
        finally:
            self._reaper.unregister(task)      # even on cancellation…
            await self._limiter.release()      # …the slot ALWAYS returns

    async def _dual_write(self, agent_id: str, summary: str, location: str) -> None:
        """Background half of memory extraction. With a MemoryStore: vector
        write in the service, structural mutations onto the single-writer
        queue. Without one (Phase ≤2 config): legacy observed-edge write."""
        try:
            if self._memory is not None:
                for mutation in await self._memory.store_memory(
                    agent_id, summary, location
                ):
                    await self._graph_queue.submit(mutation)
            else:
                await self._graph_queue.submit({
                    "type": "observed",
                    "agent_id": agent_id,
                    "location": location,
                    "t": time.time(),
                    "summary": summary,
                })
        except Exception:
            log.exception("Dual-write failed for %s", agent_id)


# ============================================================================
# 7. World Model + The Decoupled 20 Hz Tick Loop
# ============================================================================

@dataclass
class Agent:
    id: str
    x: float
    y: float
    vx: float
    vy: float
    last_reasoned_at: float = 0.0   # enforces the ~2 Hz per-agent LLM cadence
    degraded: bool = False          # True → running on state-machine heuristics
    is_human: bool = False          # externally-driven avatar: physics never
                                    # integrates it, LLM never speaks AS it


class World:
    def __init__(self, cfg: EngineConfig, n_agents: int) -> None:
        self._cfg = cfg
        # Stateful environmental objects (Phase 8). Lazy import keeps this
        # module importable standalone; entities.py is stdlib-only.
        from entities import EntityRegistry
        self.entities = EntityRegistry()
        self.navigation: Any = None       # NavigationService (Phase 9) | None
        self.agents: list[Agent] = [
            Agent(
                id=f"agent_{i:03d}",
                x=random.uniform(0, cfg.world_size),
                y=random.uniform(0, cfg.world_size),
                vx=random.uniform(-2.0, 2.0),
                vy=random.uniform(-2.0, 2.0),
            )
            for i in range(n_agents)
        ]

    def spawn_human(self, avatar_id: str) -> Agent:
        """Inject the human observer as a first-class spatial entity. Agents
        perceive it exactly as they perceive each other; its position is
        authoritative from the UE5 uplink, never from local physics."""
        human = Agent(id=avatar_id, x=self._cfg.world_size / 2,
                      y=self._cfg.world_size / 2, vx=0.0, vy=0.0,
                      is_human=True)
        self.agents.append(human)
        return human

    def find(self, agent_id: str) -> Agent | None:
        for a in self.agents:
            if a.id == agent_id:
                return a
        return None

    def step(self, dt: float) -> None:
        """Pure CPU physics: integrate positions, bounce off world bounds.
        Human avatars are skipped: their coordinates are written by the
        AvatarUplink between ticks (single-threaded asyncio — the sync step
        cannot interleave with the uplink callback, so no torn reads).
        Phase 1: replace the O(n²) proximity scan with a spatial hash grid."""
        s = self._cfg.world_size
        nav = self.navigation
        for a in self.agents:
            if a.is_human:
                continue
            if nav is not None:
                # Per-axis wall slide: an axis whose step lands in solid
                # geometry is cancelled; the other axis proceeds — agents
                # slide along walls instead of phasing through them.
                nx = a.x + a.vx * dt
                if nav.blocked(nx, a.y):
                    a.vx = 0.0
                else:
                    a.x = nx
                ny = a.y + a.vy * dt
                if nav.blocked(a.x, ny):
                    a.vy = 0.0
                else:
                    a.y = ny
            else:
                a.x += a.vx * dt
                a.y += a.vy * dt
            if not 0.0 <= a.x <= s:
                a.vx *= -1.0
                a.x = min(max(a.x, 0.0), s)
            if not 0.0 <= a.y <= s:
                a.vy *= -1.0
                a.y = min(max(a.y, 0.0), s)

    def proximity_pairs(self) -> list[tuple[Agent, Agent]]:
        if self.navigation is not None:
            # Grid-accelerated: the spatial hash was rebuilt this tick.
            return self.navigation.proximity_pairs(self._cfg.proximity_radius)
        r2 = self._cfg.proximity_radius ** 2
        out: list[tuple[Agent, Agent]] = []
        ags = self.agents
        for i in range(len(ags)):
            for j in range(i + 1, len(ags)):
                dx = ags[i].x - ags[j].x
                dy = ags[i].y - ags[j].y
                if dx * dx + dy * dy <= r2:
                    out.append((ags[i], ags[j]))
        return out


# ----------------------------------------------------------------------------
# RadixAttention prompt contract (READ BEFORE EDITING ANY PROMPT CODE)
# ----------------------------------------------------------------------------
# SGLang's RadixAttention caches the KV state of every request in a radix tree
# keyed on the TOKENIZED prompt. Two requests share cached prefill work up to
# the first token at which their prompts diverge. To keep the prefill cache
# hit rate > 80% across all agents, prompts MUST be assembled in strictly
# decreasing order of stability:
#
#   [1] SHARED_WORLD_PREFIX   — identical for ALL agents, ALL ticks.  (root)
#   [2] slow-changing globals — world events/news; refresh at low cadence
#                               (e.g. every N seconds), NEVER per-tick.
#   [3] per-agent persona     — stable per agent → each agent gets one warm
#                               branch off the shared root.            (branch)
#   [4] dynamic event payload — the observation for THIS request only. (leaf)
#
# Byte-stability rules for segments [1]–[3]:
#   * NO timestamps, tick counters, or "current time" strings.
#   * NO floating-point coordinates or anything derived from physics state.
#   * NO agent-specific values inside [1] or [2] (that forks the shared root).
#   * NO re-serialization that can reorder dict keys or change whitespace —
#     one byte of drift retokenizes and forks the tree, silently destroying
#     the hit rate.
#   * Rotate segment [2] atomically and infrequently: each rotation is one
#     deliberate cache re-prefill, amortized across every agent's next call.
#
# The chat template counts too: the template applied server-side prepends its
# own tokens, so keep `messages` structure (roles, ordering) identical across
# requests — system message first, single user message last.
#
# Verify in production, don't assume: SGLang reports `cached_tokens` in the
# streamed usage payload and exposes prefix-cache hit metrics on /metrics.
# Target: cached_tokens / prompt_tokens > 0.8 once the tree is warm.
# ----------------------------------------------------------------------------
# When spatial agency is enabled, ALL dialogue tasks share THIS root instead
# — a different constant, but equally byte-stable across every agent and tick,
# so RadixAttention still keeps one warm tree. Never mix the two prefixes in
# one deployment: two roots = two half-warm trees.
EMBODIED_WORLD_PREFIX = (
    "You are an autonomous agent inhabiting the Panopticon plaza, a persistent "
    "real-time town-square simulation observed by a live audience. A human "
    "visitor may be physically present among you.\n"
    "Simulation rules:\n"
    "1. Stay strictly in character at all times.\n"
    "2. Reference only entities and events you can directly perceive.\n"
    "3. Never mention being an AI, a model, or part of a simulation.\n"
    "4. When the human visitor addresses you, their words take precedence "
    "over conversations with other agents.\n"
    "Respond ONLY with a JSON object of this exact shape:\n"
    '{"say": "<one short spoken sentence>", '
    '"action": {"tool": "move_towards", "target_id": "<visible entity id>"} '
    "or null}\n"
    "Use move_towards to physically approach whoever or whatever deserves "
    "your attention; use null to stand your ground."
)

SHARED_WORLD_PREFIX = (
    "You are an autonomous agent inhabiting the Panopticon plaza, a persistent "
    "real-time town-square simulation observed by a live audience.\n"
    "Simulation rules:\n"
    "1. Stay strictly in character at all times.\n"
    "2. Respond with exactly one short spoken sentence, no narration.\n"
    "3. Reference only entities and events you can directly perceive.\n"
    "4. Never mention being an AI, a model, or part of a simulation.\n"
    "5. If nothing warrants a reaction, respond with a brief ambient remark."
)


@dataclass
class KinematicCommand:
    """An agent's intent to act on the physics loop, produced by the
    InferenceDispatcher's action router and consumed by the TickEngine.
    Lifetime: replaced atomically per agent (dict item assignment on the
    single event loop), expired by TTL, cleared on arrival."""
    tool: str                     # currently: "move_towards"
    target_id: str
    issued_at: float = field(default_factory=time.monotonic)
    ttl_s: float = 8.0
    # Phase 9: baked waypoints attached asynchronously by the Q1 planner.
    # None = straight-line steering (the always-available fallback).
    waypoints: list[tuple[float, float]] | None = None
    wp_index: int = 0


class TickEngine:
    """The heartbeat. Runs at exactly `tick_hz` with drift compensation and
    performs ONLY non-blocking work: physics integration, trigger detection,
    and lock-free enqueues. All I/O lives in other coroutines."""

    def __init__(
        self,
        cfg: EngineConfig,
        world: World,
        scheduler: MLFQScheduler,
        limiter: AIMDLimiter,
    ) -> None:
        self._cfg = cfg
        self._world = world
        self._scheduler = scheduler
        self._limiter = limiter
        self.tick_count = 0
        self.max_tick_overrun_ms = 0.0
        # Optional per-tick observer hook (Phase 6 broadcaster). MUST be a
        # plain synchronous callable with no awaits; it runs inside the 20 Hz
        # loop under a fail-open guard.
        self._tick_hooks: list[Any] = []
        self.tick_hook_failures = 0
        self.navigation: Any = None    # Phase 9 geometric core, optional
        # THE COMMAND BOARD — the lock-free seam between agency and physics.
        # Writers: dispatcher coroutines via issue_command (atomic dict item
        # assignment). Reader: _apply_commands, synchronously, once per tick.
        # Single-threaded asyncio + a synchronous physics step = no torn
        # state, no locks, latest-command-wins per agent by construction.
        self.command_board: dict[str, KinematicCommand] = {}
        self.commands_issued = 0
        self.commands_completed = 0
        self.commands_expired = 0
        # THE INTENT BOARD (Phase 8): entity interactions buffered during
        # tick T, resolved deterministically at the head of tick T+1. Same
        # lock-free event-loop theorem as the command board; list append is
        # atomic, the resolution pass swaps the list wholesale.
        self.intent_board: list[Any] = []
        from entities import ResolutionStats
        self.intent_stats = ResolutionStats()

    def submit_intent(self, intent: Any) -> None:
        self.intent_board.append(intent)

    def set_navigation(self, nav: Any) -> None:
        """Attach the Phase 9 geometric core: wall physics, grid proximity,
        collision resolution, and Q1 path planning all switch on together."""
        self.navigation = nav
        self._world.navigation = nav
        nav.bind(self, self._world)

    def issue_command(self, agent_id: str, cmd: KinematicCommand) -> None:
        self.command_board[agent_id] = cmd     # atomic replace: latest wins
        self.commands_issued += 1
        if self.navigation is not None:
            # Straight-line steering starts THIS tick; the plan arrives on
            # Q1 whenever speech allows, and upgrades the command in place.
            self._scheduler.submit_nowait(AgentTask(
                agent_id=agent_id, kind="pathfind",
                level=QueueLevel.COGNITIVE,
                messages=[],
                payload={"target_id": cmd.target_id}))

    def _apply_commands(self, now: float) -> None:
        """Steering pass, runs synchronously at the head of each tick BEFORE
        integration: converts standing commands into velocity mutations. The
        physics step then integrates those velocities exactly like organic
        wander — agency is just another author of vx/vy."""
        if not self.command_board:
            return
        speed = self._cfg.steering_speed
        arrive2 = self._cfg.arrival_radius ** 2
        for agent_id in list(self.command_board):
            cmd = self.command_board[agent_id]
            agent = self._world.find(agent_id)
            target = self._world.find(cmd.target_id)
            if (agent is None or target is None or agent.is_human
                    or now - cmd.issued_at > cmd.ttl_s):
                self.command_board.pop(agent_id, None)
                self.commands_expired += 1
                continue
            dx, dy = target.x - agent.x, target.y - agent.y
            d2 = dx * dx + dy * dy
            if d2 <= arrive2:
                # Arrived: clear the command, hand velocity back to ambient
                # wander at a respectful stroll.
                self.command_board.pop(agent_id, None)
                self.commands_completed += 1
                agent.vx, agent.vy = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
                continue
            # Waypoint steering: consume the baked path if one has arrived;
            # otherwise (or once exhausted) steer straight at the live
            # target — which also absorbs target drift after the final
            # waypoint.
            wx, wy = target.x, target.y
            if cmd.waypoints is not None:
                while cmd.wp_index < len(cmd.waypoints):
                    px, py = cmd.waypoints[cmd.wp_index]
                    wdx = px - agent.x
                    wdy = py - agent.y
                    if wdx * wdx + wdy * wdy <= arrive2:
                        cmd.wp_index += 1        # waypoint reached: advance
                        continue
                    wx, wy = px, py
                    break
                else:
                    cmd.waypoints = None         # path consumed: go direct
            sdx, sdy = wx - agent.x, wy - agent.y
            sd = (sdx * sdx + sdy * sdy) ** 0.5 or 1.0
            agent.vx, agent.vy = sdx / sd * speed, sdy / sd * speed

    def set_tick_hook(self, hook: Any) -> None:
        """Registers a tail hook (broadcaster, snapshotter, test probes).
        Multiple hooks run in registration order, each fail-open-guarded."""
        self._tick_hooks.append(hook)

    async def run(self) -> None:
        period = 1.0 / self._cfg.tick_hz
        next_deadline = time.monotonic() + period
        while True:
            tick_start = time.monotonic()

            # 1. Entity intents (deterministic contested resolution), then
            #    agency steering, then physics. All pure CPU, no awaits.
            self._world.entities.expire_cosigns(tick_start)
            if self.intent_board:
                from entities import resolve_intents
                pending, self.intent_board = self.intent_board, []
                resolve_intents(self._world, self._world.entities, pending,
                                self._cfg.interaction_radius,
                                self.intent_stats)
            self._apply_commands(tick_start)
            self._world.step(period)
            self._world.entities.carry_pass(self._world)
            # Phase 9 geometric passes: rebuild the spatial hash once, then
            # resolve circle overlaps. Synchronous, zero awaits — the same
            # lock-free heartbeat contract as every pass before them.
            if self.navigation is not None:
                self.navigation.rebuild(self._world.agents)
                self.navigation.collision_pass(self._world.agents)

            # 2. Trigger detection → MLFQ enqueue (lock-free fast path).
            now = time.monotonic()
            congested = self._limiter.active >= self._limiter.limit
            for a, b in self._world.proximity_pairs():
                for agent, other in ((a, b), (b, a)):
                    if agent.is_human:
                        continue        # the human perceives via UE5, not LLM
                    if now - agent.last_reasoned_at < self._cfg.reasoning_cooldown_s:
                        continue
                    agent.last_reasoned_at = now
                    if congested:
                        # Graceful degradation: don't pile onto a saturated
                        # queue; the agent keeps walking its current path in
                        # an "observing" animation state instead of speaking.
                        agent.degraded = True
                        continue
                    agent.degraded = False
                    self._scheduler.submit_nowait(AgentTask(
                        agent_id=agent.id,
                        kind="dialogue",
                        level=QueueLevel.INTERACTIVE,
                        messages=[
                            {"role": "system", "content": SHARED_WORLD_PREFIX},
                            {"role": "user",
                             "content": f"{other.id} just crossed your path at "
                                        f"({agent.x:.0f},{agent.y:.0f}). React."},
                        ],
                    ))
                    # Every interaction also spawns a background memory write.
                    self._scheduler.submit_nowait(AgentTask(
                        agent_id=agent.id,
                        kind="memory_extraction",
                        level=QueueLevel.BACKGROUND,
                        messages=[
                            {"role": "system", "content": SHARED_WORLD_PREFIX},
                            {"role": "user", "content": "Summarize this encounter "
                                                        "as a single memory fact."},
                        ],
                        payload={"observation": f"crossed paths with {other.id}",
                                 "location": "central_plaza"},
                    ))

            # 3. Tail hook: state capture for the Glasshouse broadcaster.
            #    Synchronous by contract; guarded so a broken observer can
            #    never break the heartbeat it observes.
            self.tick_count += 1
            for hook in self._tick_hooks:
                try:
                    hook(self.tick_count, self._world)
                except Exception:
                    self.tick_hook_failures += 1
                    if self.tick_hook_failures == 1:
                        log.exception("Tick hook failed — engine continues")

            # 4. Drift-compensated sleep — hold 20 Hz exactly.
            overrun_ms = max(0.0, (time.monotonic() - tick_start - period) * 1000)
            self.max_tick_overrun_ms = max(self.max_tick_overrun_ms, overrun_ms)
            next_deadline += period
            delay = next_deadline - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_deadline = time.monotonic()  # fell behind: resync, don't spiral


# ============================================================================
# 8. Telemetry — if you can't see it, you can't tune it
# ============================================================================

async def telemetry_loop(
    engine: TickEngine,
    scheduler: MLFQScheduler,
    limiter: AIMDLimiter,
    reaper: ZombieReaper,
    dispatcher: InferenceDispatcher,
    graph_queue: GraphWriteQueue,
    graph: GraphClient,
    memory: MemoryStore | None,
    interval_s: float = 3.0,
) -> None:
    while True:
        await asyncio.sleep(interval_s)
        samples = sorted(dispatcher.ttft_samples)
        p50 = samples[len(samples) // 2] if samples else 0.0
        p95 = samples[int(len(samples) * 0.95)] if samples else 0.0
        q0, q1, q2, q3 = scheduler.depths()
        log.info(
            "ticks=%d q=[%d/%d/%d/%d] aimd=%d/%d inflight=%d done=%d reaped=%d "
            "ttft(p50/p95)=%.0f/%.0fms db(batches=%d writes=%d dropped=%d)",
            engine.tick_count, q0, q1, q2, q3,
            limiter.active, limiter.limit, reaper.in_flight,
            dispatcher.completed, reaper.reaped_total, p50, p95,
            graph.batches_written, graph.mutations_written, graph_queue.dropped,
        )
        if memory is not None and getattr(memory, "retrievals", 0):
            r = sorted(memory.retrieval_ms)
            log.info(
                "memory: retrievals=%d p50=%.0fms timeouts=%d failures=%d stored=%d",
                memory.retrievals, r[len(r) // 2], memory.retrieval_timeouts,
                memory.retrieval_failures, memory.memories_stored,
            )


# ============================================================================
# 9. Wiring + Entrypoint
# ============================================================================

async def _supervise(name: str, coro: Any) -> None:
    """FAIL-OPEN AT THE TASK LEVEL: auxiliary services (consolidation,
    broadcaster, oracle) may die; the simulation may not. Their exceptions
    are logged and absorbed here."""
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("%s DIED — simulation continues without it", name)


async def run_engine(
    cfg: EngineConfig,
    n_agents: int,
    duration_s: float | None,
    *,
    sglang: SGLangClient | None = None,
    graph: GraphClient | None = None,
    media: MediaPublisher | None = None,
    memory: MemoryStore | None = None,
    consolidation_factory: Any | None = None,
    metrics_port: int | None = None,
    broadcaster: Any | None = None,
    oracle_factory: Any | None = None,
    action_router: Any | None = None,
    avatar_uplink: Any | None = None,
    voice_parts: Any | None = None,
    goalseek_factory: Any | None = None,
    navigation: Any | None = None,
    snapshot_path: str | None = None,
    director: Any | None = None,
) -> None:
    """Clients are injected; anything left as None falls back to its mock so
    each subsystem can be promoted to production independently. `memory` has
    no mock: None simply disables retrieval and dual-write (legacy path).
    `consolidation_factory`, if given, is called with (scheduler, graph_queue,
    agent_ids_fn) and must return an object with an async run() — it executes
    under fail-open supervision: its death never stops the simulation."""
    sglang = sglang if sglang is not None else MockSGLangClient()
    graph = graph if graph is not None else MockNeo4jClient()
    media = media if media is not None else LiveKitPublisher()

    world = World(cfg, n_agents)
    scheduler = MLFQScheduler(cfg)
    limiter = AIMDLimiter(cfg)
    reaper = ZombieReaper(cfg)
    graph_queue = GraphWriteQueue(cfg, graph)
    dispatcher = InferenceDispatcher(cfg, scheduler, limiter, reaper, sglang,
                                     media, graph_queue, memory=memory,
                                     action_router=action_router,
                                     navigation=navigation)
    engine = TickEngine(cfg, world, scheduler, limiter)
    if navigation is not None:
        engine.set_navigation(navigation)
    if action_router is not None:
        action_router.bind(engine, world)      # late-bind live refs
    if avatar_uplink is not None:
        avatar_uplink.bind(world)              # spawns the human entity

    # Startup hook: clients that need an async connection (LiveKit room join,
    # MemoryService collection setup, broadcaster room join) expose connect().
    for client in (sglang, graph, media, memory, broadcaster, avatar_uplink):
        connecter = getattr(client, "connect", None)
        if connecter is not None:
            await connecter()

    tasks = [
        asyncio.create_task(engine.run(), name="tick-loop"),
        asyncio.create_task(dispatcher.run_worker(), name="mlfq-worker"),
        asyncio.create_task(graph_queue.run_writer(), name="graph-writer"),  # THE one
        asyncio.create_task(reaper.run(), name="zombie-reaper"),
        asyncio.create_task(
            telemetry_loop(engine, scheduler, limiter, reaper, dispatcher,
                           graph_queue, graph, memory),
            name="telemetry",
        ),
    ]

    daemon = None
    if consolidation_factory is not None:
        daemon = consolidation_factory(
            scheduler, graph_queue, lambda: [a.id for a in world.agents]
        )

        tasks.append(asyncio.create_task(
            _supervise("consolidation daemon", daemon.run()),
            name="consolidation-daemon"))

    if voice_parts is not None and avatar_uplink is not None:
        avatar_mod, stt = voice_parts
        ingress = avatar_mod.VoiceIngress(stt, avatar_uplink, scheduler, world)
        if avatar_uplink._room is not None:
            ingress.attach(avatar_uplink._room)
        tasks.append(asyncio.create_task(
            _supervise("voice ingress", ingress.run()), name="voice-ingress"))

    snapshotter = None
    if snapshot_path is not None:
        import persistence as persistence_mod
        # RESTORE FIRST — before any task starts, so the fresh random spawn
        # is superseded and re-issued commands request their Q1 plans into
        # an otherwise-idle scheduler.
        persistence_mod.restore_world(snapshot_path, world, engine,
                                      dispatcher.goal_board)
        snapshotter = persistence_mod.Snapshotter(
            snapshot_path, goal_board=dispatcher.goal_board)
        snapshotter.bind(engine)
        engine.set_tick_hook(snapshotter.make_tick_hook())
    if snapshotter is not None:
        tasks.append(asyncio.create_task(
            _supervise("snapshotter", snapshotter.run_flusher()),
            name="snapshotter"))

    if director is not None:
        director.bind(engine, world, limiter, dispatcher)
        tasks.append(asyncio.create_task(
            _supervise("narrative director", director.run()),
            name="director"))

    goal_daemon = None
    if goalseek_factory is not None:
        goal_daemon = goalseek_factory(scheduler, world, dispatcher.goal_board)
        tasks.append(asyncio.create_task(
            _supervise("goalseek daemon", goal_daemon.run()),
            name="goalseek-daemon"))

    oracle = None
    if broadcaster is not None:
        # Glasshouse state stream: capture at the tick tail (fail-open hook),
        # network I/O on its own supervised task (latest-wins ring inside).
        engine.set_tick_hook(broadcaster.make_tick_hook())
        broadcaster.attach_entities(world.entities)   # Phase 8 delta plane
        tasks.append(asyncio.create_task(
            _supervise("broadcaster", broadcaster.run_sender()),
            name="glasshouse-broadcaster"))

    if oracle_factory is not None:
        # Oracle needs live refs (scheduler for injection, world for radius
        # scoping) that exist only here — hence a factory, like consolidation.
        oracle = oracle_factory(scheduler, world)
        tasks.append(asyncio.create_task(
            _supervise("oracle", oracle.run()), name="oracle"))

    if metrics_port is not None:
        # Lazy import — prometheus_client only needed when observability is on.
        import metrics as metrics_mod
        exporter = metrics_mod.MetricsServer(
            metrics_mod.EngineRefs(
                engine=engine, world=world, scheduler=scheduler,
                limiter=limiter, reaper=reaper, dispatcher=dispatcher,
                graph_queue=graph_queue, graph=graph, memory=memory,
                daemon=daemon, broadcaster=broadcaster, oracle=oracle,
                uplink=avatar_uplink, action_router=action_router,
                goal_daemon=goal_daemon, navigation=navigation,
                snapshotter=snapshotter, director=director,
            ),
            port=metrics_port,
        )
        tasks.append(asyncio.create_task(exporter.run(), name="metrics"))

    log.info("Panopticon up: %d agents, tick=%.0fHz, model=%s, sglang=%s, graph=%s",
             n_agents, cfg.tick_hz, cfg.model_name,
             type(sglang).__name__, type(graph).__name__)

    try:
        if duration_s is None:
            await asyncio.gather(*tasks)
        else:
            done, pending = await asyncio.wait(tasks, timeout=duration_s)
            for t in done:  # surface crashes instead of swallowing them
                if t.exception():
                    raise t.exception()  # type: ignore[misc]
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Close any client that exposes an async close() (real clients do;
        # mocks don't need one). Duck-typed so mocks stay dependency-free.
        # Let detached dual-writes finish (briefly) so memories aren't lost
        # on clean shutdown, then close any client exposing async close().
        if dispatcher._background_writes:
            await asyncio.wait(dispatcher._background_writes, timeout=2.0)
        for client in (sglang, graph, media, memory, broadcaster,
                       avatar_uplink):
            closer = getattr(client, "close", None)
            if closer is not None:
                try:
                    await closer()
                except Exception:
                    log.exception("Error closing %s", type(client).__name__)
        expected_hz = engine.tick_count / duration_s if duration_s else cfg.tick_hz
        log.info("Shutdown. ticks=%d (%.1f Hz effective, max overrun %.1fms), "
                 "generations=%d, cancelled=%d, reaped=%d",
                 engine.tick_count, expected_hz, engine.max_tick_overrun_ms,
                 dispatcher.completed, dispatcher.cancelled, reaper.reaped_total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Panopticon — Phase 1")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--demo-seconds", type=float, default=None,
                        help="run for N seconds then exit (default: run forever)")
    # --- Phase 1 endpoints: omit a flag to keep that subsystem mocked -------
    parser.add_argument("--sglang-url", default=None,
                        help="e.g. http://sglang-router:8000 (mock if omitted)")
    parser.add_argument("--neo4j-uri", default=None,
                        help="e.g. bolt://neo4j-db:7687 (mock if omitted)")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default="neo4j")
    # --- Phase 2 endpoints: all three required together for real audio ------
    parser.add_argument("--livekit-url", default=None,
                        help="e.g. wss://panopticon.livekit.cloud (mock if omitted)")
    parser.add_argument("--livekit-api-key", default=None)
    parser.add_argument("--livekit-api-secret", default=None)
    parser.add_argument("--tts-url", default=None,
                        help="F5-TTS microservice, e.g. http://tts-node:8020")
    # --- Phase 3 endpoints: both required together for agent memory ---------
    parser.add_argument("--qdrant-url", default=None,
                        help="http://qdrant:6333, a local path, or ':memory:' "
                             "(memory disabled if omitted)")
    parser.add_argument("--embeddings-url", default=None,
                        help="OpenAI-compatible /v1/embeddings base URL")
    parser.add_argument("--embeddings-model", default="BAAI/bge-m3")
    parser.add_argument("--consolidate", action="store_true",
                        help="run the Phase 4 consolidation daemon "
                             "(requires the memory stack)")
    parser.add_argument("--metrics-port", type=int, default=None,
                        help="expose Prometheus /metrics on this port")
    # --- Phase 6: the Glasshouse ---------------------------------------------
    parser.add_argument("--broadcast-state", action="store_true",
                        help="stream 20Hz binary world state to UE5 observers "
                             "(requires the LiveKit flags)")
    parser.add_argument("--oracle-feed-url", default=None,
                        help="JSON event feed for real-world injection")
    # --- Phase 7: embodiment & agency ---------------------------------------
    parser.add_argument("--embody", action="store_true",
                        help="human avatar uplink + spatial agency "
                             "(requires the --livekit-* flags)")
    parser.add_argument("--stt-ws-url", default=None,
                        help="streaming STT websocket for human voice ingest")
    # --- Phase 8: intrinsic motivation ---------------------------------------
    parser.add_argument("--navmesh", action="store_true",
                        help="enable the Phase 9 geometric core (collisions, "
                             "wall physics, Q1 pathfinding)")
    parser.add_argument("--map-file", default=None,
                        help="JSON list of solid rects [[x0,y0,x1,y1], ...]")
    parser.add_argument("--snapshot-path", default=None,
                        help="persist/restore ephemeral world state, e.g. "
                             "/workspace/state.json")
    parser.add_argument("--director", action="store_true",
                        help="enable the Narrative Director (economy + "
                             "environment levers)")
    parser.add_argument("--goals", action="store_true",
                        help="run the Q3 goal-seeking daemon (best with "
                             "--neo4j-uri for belief-grounded goals)")
    args = parser.parse_args()

    sglang: SGLangClient | None = None
    graph: GraphClient | None = None
    media_client: MediaPublisher | None = None
    memory_svc: MemoryStore | None = None
    if args.sglang_url or args.neo4j_uri:
        # Lazy import: tick_engine.py itself stays free of third-party deps.
        import clients

        if args.sglang_url:
            sglang = clients.RealSGLangClient(args.sglang_url)
        if args.neo4j_uri:
            if not args.neo4j_password:
                parser.error("--neo4j-password is required with --neo4j-uri")
            graph = clients.RealNeo4jGraphClient(
                args.neo4j_uri, args.neo4j_user, args.neo4j_password,
                database=args.neo4j_database,
            )

    if args.livekit_url:
        # Lazy import — the ~3 lines promised: build TTS client, build the
        # LiveKit publisher around it, inject. run_engine awaits .connect().
        import media
        if not (args.livekit_api_key and args.livekit_api_secret and args.tts_url):
            parser.error("--livekit-url requires --livekit-api-key, "
                         "--livekit-api-secret, and --tts-url")
        tts = media.RealTTSEngine(args.tts_url)
        media_client = media.RealLiveKitPublisher(
            args.livekit_url, args.livekit_api_key, args.livekit_api_secret, tts)

    consolidation_factory = None
    if args.qdrant_url:
        # Lazy import: embedder + vector store + (real) graph reader → service.
        import memory as memory_mod
        if not args.embeddings_url:
            parser.error("--qdrant-url requires --embeddings-url")
        embedder = memory_mod.RealEmbeddingClient(args.embeddings_url,
                                                  args.embeddings_model)
        vectors = memory_mod.RealQdrantClient(args.qdrant_url)
        # Graph expansion needs the REAL Neo4j client; with the mock graph the
        # service still works, falling back to vector payloads (step 2 no-op).
        graph_reader = graph if graph is not None else None
        memory_svc = memory_mod.MemoryService(embedder, vectors, graph_reader)

        if args.consolidate:
            if graph is None:
                parser.error("--consolidate requires --neo4j-uri (the sweep "
                             "reads unconsolidated memories from the graph)")
            import consolidation as consolidation_mod
            consolidation_factory = (
                lambda scheduler, graph_queue, agent_ids:
                    consolidation_mod.ConsolidationDaemon(
                        embedder, vectors, graph, scheduler, graph_queue,
                        agent_ids)
            )

    avatar_uplink = None
    action_router = None
    voice_parts = None
    if args.embody:
        if not (args.livekit_url and args.livekit_api_key
                and args.livekit_api_secret):
            parser.error("--embody requires the --livekit-* flags")
        import avatar as avatar_mod
        avatar_uplink = avatar_mod.AvatarUplink(
            args.livekit_url, args.livekit_api_key, args.livekit_api_secret)
        action_router = avatar_mod.ActionRouter()
        if args.stt_ws_url:
            voice_parts = (avatar_mod, avatar_mod.WsSTTEngine(args.stt_ws_url))

    navigation_svc = None
    if args.navmesh:
        import json as _json
        import navigation as navigation_mod
        _grid = navigation_mod.NavGrid(EngineConfig().world_size)
        if args.map_file:
            for rect in _json.load(open(args.map_file)):
                _grid.add_obstacle_rect(*rect)
        navigation_svc = navigation_mod.NavigationService(_grid)

    goalseek_factory = None
    if args.goals:
        import goalseek as goalseek_mod
        _belief_graph = graph if args.neo4j_uri else None
        goalseek_factory = (
            lambda scheduler, world, board:
                goalseek_mod.GoalSeekDaemon(_belief_graph, scheduler, world,
                                            board))

    state_broadcaster = None
    if args.broadcast_state:
        if not (args.livekit_url and args.livekit_api_key
                and args.livekit_api_secret):
            parser.error("--broadcast-state requires the --livekit-* flags")
        import broadcaster as broadcaster_mod
        state_broadcaster = broadcaster_mod.StateBroadcaster(
            args.livekit_url, args.livekit_api_key, args.livekit_api_secret)
        # is_speaking wiring: only meaningful when audio is also enabled.
        if media_client is not None:
            state_broadcaster._speaking = broadcaster_mod.livekit_speaking_probe(
                media_client)

    oracle_factory = None
    if args.oracle_feed_url:
        import oracle as oracle_mod
        feed = oracle_mod.JsonPollFeed(args.oracle_feed_url)
        oracle_factory = (lambda scheduler, world:
                          oracle_mod.Oracle(feed, scheduler, world))

    asyncio.run(run_engine(EngineConfig(), args.agents, args.demo_seconds,
                           sglang=sglang, graph=graph, media=media_client,
                           memory=memory_svc,
                           consolidation_factory=consolidation_factory,
                           metrics_port=args.metrics_port,
                           broadcaster=state_broadcaster,
                           oracle_factory=oracle_factory,
                           action_router=action_router,
                           avatar_uplink=avatar_uplink,
                           voice_parts=voice_parts,
                           goalseek_factory=goalseek_factory,
                           navigation=navigation_svc,
                           snapshot_path=args.snapshot_path,
                           director=(__import__("director").Director()
                                     if args.director else None)))


if __name__ == "__main__":
    main()
