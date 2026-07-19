"""
Project Panopticon — Phase 5: The 50-Agent Crucible
====================================================
crucible.py

Load harness for the cognitive engine. LiveKit/TTS are BYPASSED (the no-op
publisher stands in) so every joule of stress lands on the MLFQ scheduler,
the AIMD limiter, and the inference path — the systems that decide whether
fifty agents degrade gracefully or whether the GPU server dies.

Three phases:
  WARMUP    organic interactions only; AIMD probes upward (additive increase).
  STORM     a synthetic "market crash": burst Q0/Q1 tasks injected on top of
            maximum organic collision density (50 agents, tight world).
            This is designed to EXCEED server capacity if nothing throttles.
  RECOVERY  injection stops; queues must drain, the limit must regrow.

The mathematical claim under test
---------------------------------
Let W(t) be the AIMD ceiling and A(t) the in-flight requests. The engine
guarantees, by construction of AIMDLimiter.acquire():

    (i)  every ADMISSION satisfies A <= ceil(W) at that instant, and
    (ii) whenever A(t) > W(t) (the drain window after a multiplicative
         decrease — in-flight work is never evicted), acquire() blocks,
         so A(t) is non-increasing until it re-enters the ceiling.

W(t) contracts multiplicatively (×0.75) whenever observed TTFT crosses
the 500 ms budget — which, for any server whose latency grows monotonically
with concurrency, pins W(t) near the concurrency knee K_budget where
TTFT(K_budget) = budget. Provided K_budget < K_OOM (the concurrency at which
the server falls over), the composition yields the safety property:

    max_t A(t) <= max_t ceil(W(t)) < K_OOM               (server never dies)

The simulated backend makes this falsifiable: TTFT(A) = base + (A/36)² ·
0.55 s crosses the 500 ms budget near A≈33, and the server hard-OOMs at
A > 48. If AIMD is broken, the storm WILL kill it. The crucible samples
A(t), W(t), queue depths, and degraded-agent counts at 5 Hz and renders a
verdict on five invariants (safety, admission, backpressure engagement,
liveness, recovery).

Usage:
  python crucible.py --simulate                       # falsifiable local run
  python crucible.py --sglang-url http://...:8000 \\   # production silicon
      [--qdrant-url ... --embeddings-url ... --neo4j-uri ... --neo4j-password ...]
      [--metrics-port 9109]                           # watch it live in Grafana
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import tick_engine as te

logging.getLogger("panopticon").setLevel(logging.WARNING)
log = logging.getLogger("crucible")
logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d CRUCIBLE %(message)s",
                    datefmt="%H:%M:%S")


# ============================================================================
# 1. Crucible configuration — a deliberately hostile world
# ============================================================================

@dataclass(frozen=True)
class CrucibleConfig:
    agents: int = 50
    warmup_s: float = 6.0
    storm_s: float = 12.0
    recovery_s: float = 12.0
    sample_hz: float = 5.0
    # Storm injection: synthetic Q0 reaction bursts on top of organic load.
    storm_inject_hz: float = 20.0        # bursts per second
    storm_burst_size: int = 4            # tasks per burst
    # Simulated backend shape (see module docstring math).
    sim_base_ttft_s: float = 0.06
    sim_knee: float = 36.0
    sim_scale_s: float = 0.55
    sim_oom_capacity: int = 48

    def engine_config(self) -> te.EngineConfig:
        # Tight arena: 50 agents in an 18×18 world with a 5-unit trigger
        # radius ≈ continuous collision saturation. Everything else stays at
        # production defaults — the crucible tests THE system, not a tuned one.
        return te.EngineConfig(world_size=18.0, proximity_radius=5.0)


class ServerOOMError(RuntimeError):
    """The simulated backend 'died'. Any single occurrence fails the run."""


class StressSGLangClient:
    """Concurrency-dependent latency model with a hard death line.

    TTFT(A) = base + (A / knee)² · scale        (queueing-flavored convexity)
    A > oom_capacity  →  ServerOOMError          (KV cache exhaustion)

    The knee is placed so the 500 ms budget is crossed near A≈33 — well below
    the 48-request death line. A correct AIMD therefore stabilizes in the
    low 30s; a broken one lets the storm push A past 48 and the verdict fails.
    """

    def __init__(self, cfg: CrucibleConfig) -> None:
        self._cfg = cfg
        self.active = 0
        self.max_active_seen = 0
        self.oom_events = 0
        self.requests_served = 0

    async def stream_chat(
        self, model: str, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        self.active += 1
        self.max_active_seen = max(self.max_active_seen, self.active)
        try:
            if self.active > self._cfg.sim_oom_capacity:
                self.oom_events += 1
                raise ServerOOMError(
                    f"KV cache exhausted at concurrency {self.active}")
            a = self.active
            ttft = (self._cfg.sim_base_ttft_s
                    + (a / self._cfg.sim_knee) ** 2 * self._cfg.sim_scale_s)
            await asyncio.sleep(ttft)
            for tok in ("*reacts*", " The", " market", " crashed!"):
                yield tok
                await asyncio.sleep(0.004)
            self.requests_served += 1
        finally:
            self.active -= 1


# ============================================================================
# 2. Sampling, storm injection, verdicts
# ============================================================================

@dataclass
class Sample:
    t: float
    phase: str
    active: int
    limit: int
    q0: int
    q1: int
    q2: int
    degraded: int
    ticks: int
    backoffs: int


@dataclass
class Crucible:
    ccfg: CrucibleConfig
    sglang: Any
    graph: Any
    memory: Any = None
    samples: list[Sample] = field(default_factory=list)
    phase: str = "init"

    def __post_init__(self) -> None:
        cfg = self.ccfg.engine_config()
        self.cfg = cfg
        self.world = te.World(cfg, self.ccfg.agents)
        self.scheduler = te.MLFQScheduler(cfg)
        self.limiter = te.AIMDLimiter(cfg)
        self.reaper = te.ZombieReaper(cfg)
        self.graph_queue = te.GraphWriteQueue(cfg, self.graph)
        # LiveKit/TTS bypass: the Phase-0 no-op publisher isolates cognition.
        self.dispatcher = te.InferenceDispatcher(
            cfg, self.scheduler, self.limiter, self.reaper, self.sglang,
            te.LiveKitPublisher(), self.graph_queue, memory=self.memory)
        self.engine = te.TickEngine(cfg, self.world, self.scheduler, self.limiter)

    # ------------------------------------------------------------- helpers

    def _snap(self) -> Sample:
        q0, q1, q2 = self.scheduler.depths()
        return Sample(
            t=time.monotonic(), phase=self.phase,
            active=self.limiter.active, limit=self.limiter.limit,
            q0=q0, q1=q1, q2=q2,
            degraded=sum(1 for a in self.world.agents if a.degraded),
            ticks=self.engine.tick_count,
            backoffs=self.limiter.backoff_events,
        )

    async def _monitor(self) -> None:
        period = 1.0 / self.ccfg.sample_hz
        while True:
            self.samples.append(self._snap())
            await asyncio.sleep(period)

    async def _storm_injector(self) -> None:
        """The market crash: high-priority reaction bursts injected directly
        into Q0/Q1, on top of organic collision load, until the phase ends."""
        period = 1.0 / self.ccfg.storm_inject_hz
        while self.phase == "storm":
            for _ in range(self.ccfg.storm_burst_size):
                agent = random.choice(self.world.agents)
                self.scheduler.submit_nowait(te.AgentTask(
                    agent_id=agent.id, kind="dialogue",
                    level=te.QueueLevel.INTERACTIVE,
                    messages=[
                        {"role": "system", "content": te.SHARED_WORLD_PREFIX},
                        {"role": "user",
                         "content": "The market just crashed! React."},
                    ]))
            # One cognitive task per burst: exercises Q1 demotion under load.
            agent = random.choice(self.world.agents)
            self.scheduler.submit_nowait(te.AgentTask(
                agent_id=agent.id, kind="reasoning",
                level=te.QueueLevel.COGNITIVE,
                messages=[
                    {"role": "system", "content": te.SHARED_WORLD_PREFIX},
                    {"role": "user", "content": "Re-plan your route."},
                ]))
            await asyncio.sleep(period)

    # ----------------------------------------------------------------- run

    async def run(self) -> bool:
        tasks = [
            asyncio.create_task(self.engine.run(), name="tick-loop"),
            asyncio.create_task(self.dispatcher.run_worker(), name="mlfq-worker"),
            asyncio.create_task(self.graph_queue.run_writer(), name="graph-writer"),
            asyncio.create_task(self.reaper.run(), name="zombie-reaper"),
            asyncio.create_task(self._monitor(), name="monitor"),
        ]
        t0 = time.monotonic()
        try:
            self.phase = "warmup"
            log.info("WARMUP   %ds — organic load, AIMD probing upward",
                     int(self.ccfg.warmup_s))
            await asyncio.sleep(self.ccfg.warmup_s)

            self.phase = "storm"
            log.info("STORM    %ds — market crash: +%d tasks/s injected on top "
                     "of organic saturation", int(self.ccfg.storm_s),
                     int(self.ccfg.storm_inject_hz
                         * (self.ccfg.storm_burst_size + 1)))
            storm = asyncio.create_task(self._storm_injector())
            await asyncio.sleep(self.ccfg.storm_s)

            self.phase = "recovery"
            await storm
            log.info("RECOVERY %ds — injection stopped; queues must drain, "
                     "limit must regrow", int(self.ccfg.recovery_s))
            await asyncio.sleep(self.ccfg.recovery_s)
            wall_s = time.monotonic() - t0
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        # Let in-flight generations wind down for the final-state checks.
        for _ in range(100):
            if self.limiter.active == 0:
                break
            await asyncio.sleep(0.05)
        return self.verdict(wall_s)

    # -------------------------------------------------------------- verdict

    def verdict(self, wall_s: float) -> bool:
        s_all = self.samples
        s_storm = [s for s in s_all if s.phase == "storm"]
        s_rec = [s for s in s_all if s.phase == "recovery"]
        checks: list[tuple[str, bool, str]] = []

        # A. SAFETY — the server survived (simulate mode: hard proof).
        if isinstance(self.sglang, StressSGLangClient):
            checks.append((
                "A. SAFETY: server never OOM'd",
                self.sglang.oom_events == 0
                and self.sglang.max_active_seen <= self.ccfg.sim_oom_capacity,
                f"max server concurrency {self.sglang.max_active_seen} "
                f"< death line {self.ccfg.sim_oom_capacity}, "
                f"oom_events={self.sglang.oom_events}",
            ))
        else:
            served = self.dispatcher.completed
            fails = self.limiter.total_failures
            ratio = fails / max(1, served + fails)
            checks.append((
                "A. SAFETY: no failure cascade on real backend",
                ratio < 0.10,
                f"failure ratio {ratio:.1%} (completed={served}, signals={fails})",
            ))

        # B. ADMISSION INVARIANT — stated precisely:
        #   (i)  at every ADMISSION, post-increment A <= ceil(W)   [exact
        #        counter inside AIMDLimiter.acquire — no sampling races]
        #   (ii) A(t) > W(t) is legal ONLY while draining after a
        #        multiplicative decrease (in-flight requests finish naturally;
        #        they are never evicted). While over the ceiling, acquire()
        #        blocks, so A may only fall: every contiguous over-ceiling
        #        run in the samples must end at or below where it began.
        drain_ok = True
        run_start: int | None = None
        for i, s in enumerate(s_all):
            over = s.active > s.limit
            if over and run_start is None:
                run_start = s.active
            elif not over and run_start is not None:
                run_start = None
            elif over and run_start is not None and s.active > run_start:
                drain_ok = False
        over_samples = sum(1 for s in s_all if s.active > s.limit)
        checks.append((
            "B. INVARIANT: admissions bounded by ceiling; overshoot only drains",
            self.limiter.admission_violations == 0 and drain_ok,
            f"admission-point violations={self.limiter.admission_violations} "
            f"(exact), {over_samples} post-backoff drain samples "
            f"(monotone={'yes' if drain_ok else 'NO'})",
        ))

        # C. BACKPRESSURE ENGAGED — the storm actually bent the system.
        storm_backoffs = (s_storm[-1].backoffs - s_storm[0].backoffs) if s_storm else 0
        max_degraded = max((s.degraded for s in s_storm), default=0)
        checks.append((
            "C. BACKPRESSURE: multiplicative decrease + graceful degradation",
            storm_backoffs >= 1 and max_degraded > 0,
            f"storm backoff events={storm_backoffs}, "
            f"peak degraded agents={max_degraded}/{self.ccfg.agents}",
        ))

        # D. LIVENESS — 20 Hz held through the entire ordeal.
        hz = self.engine.tick_count / wall_s
        checks.append((
            "D. LIVENESS: tick loop unbowed",
            hz >= 19.0 and self.engine.max_tick_overrun_ms < 15.0,
            f"{hz:.1f} Hz effective, max overrun "
            f"{self.engine.max_tick_overrun_ms:.1f} ms",
        ))

        # E. RECOVERY — additive regrowth, queues drained, slots returned.
        storm_min_limit = min((s.limit for s in s_storm), default=0)
        end_limit = s_rec[-1].limit if s_rec else 0
        end_q0 = s_rec[-1].q0 if s_rec else -1
        checks.append((
            "E. RECOVERY: limit regrew, Q0 drained, zero leaked slots",
            end_limit > storm_min_limit and end_q0 == 0
            and self.limiter.active == 0,
            f"limit {storm_min_limit}→{end_limit}, final q0={end_q0}, "
            f"active={self.limiter.active}",
        ))

        # ---- Report ---------------------------------------------------------
        print("\n" + "=" * 76)
        print(f" THE CRUCIBLE — {self.ccfg.agents} agents, "
              f"{self.engine.tick_count} ticks, "
              f"{self.dispatcher.completed} generations, "
              f"{self.reaper.reaped_total} reaped")
        print("=" * 76)
        for phase in ("warmup", "storm", "recovery"):
            ph = [s for s in s_all if s.phase == phase]
            if not ph:
                continue
            print(f"  {phase.upper():9s} active mean/max {_mean(ph,'active'):5.1f}/"
                  f"{max(s.active for s in ph):3d}   "
                  f"limit min/max {min(s.limit for s in ph):3d}/"
                  f"{max(s.limit for s in ph):3d}   "
                  f"q0 max {max(s.q0 for s in ph):4d}   "
                  f"degraded max {max(s.degraded for s in ph):3d}")
        ttft = sorted(self.dispatcher.ttft_samples)
        if ttft:
            print(f"  TTFT p50/p95: {ttft[len(ttft)//2]:.0f}/"
                  f"{ttft[int(len(ttft)*0.95)]:.0f} ms (budget 500)")
        print("-" * 76)
        ok = True
        for name, passed, detail in checks:
            ok &= passed
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}\n"
                  f"         {detail}")
        print("=" * 76)
        print(f"  VERDICT: {'SURVIVED THE CRUCIBLE' if ok else 'BROKEN — see failures above'}")
        print("=" * 76 + "\n")
        return ok


def _mean(samples: list[Sample], attr: str) -> float:
    return sum(getattr(s, attr) for s in samples) / max(1, len(samples))


# ============================================================================
# 3. Entrypoint — simulate or production silicon
# ============================================================================

async def _amain(args: argparse.Namespace) -> bool:
    ccfg = CrucibleConfig(agents=args.agents)

    if args.simulate:
        sglang: Any = StressSGLangClient(ccfg)
        graph: Any = te.MockNeo4jClient()
        mem = None
    else:
        import clients
        if not args.sglang_url:
            raise SystemExit("--sglang-url required (or use --simulate)")
        sglang = clients.RealSGLangClient(args.sglang_url)
        graph = (clients.RealNeo4jGraphClient(
                    args.neo4j_uri, args.neo4j_user, args.neo4j_password)
                 if args.neo4j_uri else te.MockNeo4jClient())
        mem = None
        if args.qdrant_url and args.embeddings_url:
            import memory as memory_mod
            mem = memory_mod.MemoryService(
                memory_mod.RealEmbeddingClient(args.embeddings_url,
                                               args.embeddings_model),
                memory_mod.RealQdrantClient(args.qdrant_url),
                graph if args.neo4j_uri else None)
            await mem.connect()

    crucible = Crucible(ccfg, sglang, graph, memory=mem)

    if args.metrics_port:
        import metrics as metrics_mod
        exporter = metrics_mod.MetricsServer(metrics_mod.EngineRefs(
            engine=crucible.engine, world=crucible.world,
            scheduler=crucible.scheduler, limiter=crucible.limiter,
            reaper=crucible.reaper, dispatcher=crucible.dispatcher,
            graph_queue=crucible.graph_queue, graph=graph, memory=mem),
            port=args.metrics_port)
        asyncio.create_task(exporter.run(), name="metrics")

    ok = await crucible.run()
    for c in (sglang, graph, mem):
        closer = getattr(c, "close", None)
        if closer:
            await closer()
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Panopticon Phase 5 stress harness")
    p.add_argument("--agents", type=int, default=50)
    p.add_argument("--simulate", action="store_true",
                   help="falsifiable local run against the queueing-model backend")
    p.add_argument("--sglang-url", default=None)
    p.add_argument("--neo4j-uri", default=None)
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-password", default=None)
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--embeddings-url", default=None)
    p.add_argument("--embeddings-model", default="BAAI/bge-m3")
    p.add_argument("--metrics-port", type=int, default=None)
    args = p.parse_args()
    ok = asyncio.run(_amain(args))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
