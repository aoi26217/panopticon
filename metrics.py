"""
Project Panopticon — Phase 5 Observability
===========================================
metrics.py

Prometheus exporter for the engine's live state. Design choice: a CUSTOM
COLLECTOR rather than instrumented counters sprinkled through the hot paths.
The engine already keeps its own telemetry (plain ints and deques — the
cheapest possible instrumentation for a 20 Hz loop); this module reads that
state AT SCRAPE TIME. Zero overhead between scrapes, zero new locks in the
tick loop, and the /metrics output can never disagree with the engine's own
numbers because they ARE the engine's own numbers.

Attach Grafana to Prometheus and the tuning story for the 50-agent crucible
is four panels:
  1. panopticon_aimd_limit vs panopticon_aimd_active   (the breathing graph)
  2. panopticon_ttft_ms{quantile="0.95"} vs the 500 ms budget line
  3. panopticon_mlfq_depth by queue                    (starvation watch)
  4. panopticon_tick_max_overrun_ms                    (must stay ~0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

from aiohttp import web
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import Collector

log = logging.getLogger("panopticon.metrics")

_QUEUE_LABELS = ("interactive", "cognitive", "background", "goal")


@dataclass
class EngineRefs:
    """Live references into the running engine. Optional subsystems
    (memory, daemon) may be None; their metric families are simply omitted."""
    engine: Any            # TickEngine
    world: Any             # World
    scheduler: Any         # MLFQScheduler
    limiter: Any           # AIMDLimiter
    reaper: Any            # ZombieReaper
    dispatcher: Any        # InferenceDispatcher
    graph_queue: Any       # GraphWriteQueue
    graph: Any             # GraphClient
    memory: Any = None     # MemoryService | None
    daemon: Any = None     # ConsolidationDaemon | None
    broadcaster: Any = None  # StateBroadcaster | None
    oracle: Any = None       # Oracle | None
    uplink: Any = None       # AvatarUplink | None
    action_router: Any = None  # ActionRouter | None
    goal_daemon: Any = None  # GoalSeekDaemon | None
    navigation: Any = None   # NavigationService | None
    snapshotter: Any = None  # Snapshotter | None
    director: Any = None     # Director | None


def _quantile(sorted_samples: list[float], q: float) -> float:
    if not sorted_samples:
        return 0.0
    return sorted_samples[min(len(sorted_samples) - 1, int(len(sorted_samples) * q))]


class PanopticonCollector(Collector):
    def __init__(self, refs: EngineRefs) -> None:
        self._r = refs

    def collect(self) -> Iterable[Any]:  # noqa: C901 - one flat scrape, on purpose
        r = self._r

        # ---- Tick loop -----------------------------------------------------
        yield CounterMetricFamily(
            "panopticon_ticks", "Total spatial simulation ticks",
            value=r.engine.tick_count)
        yield GaugeMetricFamily(
            "panopticon_tick_max_overrun_ms",
            "Worst tick deadline overrun observed (ms); ~0 means 20 Hz held",
            value=r.engine.max_tick_overrun_ms)

        # ---- Agents / degradation -------------------------------------------
        agents = r.world.agents
        yield GaugeMetricFamily(
            "panopticon_agents", "Agents in the world", value=len(agents))
        yield GaugeMetricFamily(
            "panopticon_agents_degraded",
            "Agents currently on state-machine fallback (graceful degradation)",
            value=sum(1 for a in agents if a.degraded))

        # ---- AIMD limiter ----------------------------------------------------
        yield GaugeMetricFamily(
            "panopticon_aimd_limit", "Current AIMD concurrency ceiling",
            value=r.limiter.limit)
        yield GaugeMetricFamily(
            "panopticon_aimd_active", "In-flight inference requests",
            value=r.limiter.active)
        yield CounterMetricFamily(
            "panopticon_aimd_admitted", "Requests admitted past the limiter",
            value=r.limiter.total_admitted)
        yield CounterMetricFamily(
            "panopticon_aimd_failures",
            "Congestion signals (latency spikes, errors, cache saturation)",
            value=r.limiter.total_failures)
        yield CounterMetricFamily(
            "panopticon_aimd_backoff_events",
            "Multiplicative-decrease events", value=r.limiter.backoff_events)
        yield CounterMetricFamily(
            "panopticon_aimd_admission_violations",
            "Exact admission-point invariant breaches (must stay 0)",
            value=r.limiter.admission_violations)

        # ---- MLFQ ------------------------------------------------------------
        depth = GaugeMetricFamily(
            "panopticon_mlfq_depth", "Queued tasks per MLFQ level",
            labels=["queue"])
        for label, d in zip(_QUEUE_LABELS, r.scheduler.depths()):
            depth.add_metric([label], d)
        yield depth

        # ---- Inference / TTFT -------------------------------------------------
        ttft = GaugeMetricFamily(
            "panopticon_ttft_ms", "Time-To-First-Token over a rolling window",
            labels=["quantile"])
        samples = sorted(r.dispatcher.ttft_samples)
        ttft.add_metric(["0.5"], _quantile(samples, 0.5))
        ttft.add_metric(["0.95"], _quantile(samples, 0.95))
        yield ttft
        yield CounterMetricFamily(
            "panopticon_generations_completed", "Completed generations",
            value=r.dispatcher.completed)
        yield CounterMetricFamily(
            "panopticon_generations_cancelled",
            "Cancelled generations (reaper or shutdown)",
            value=r.dispatcher.cancelled)
        yield GaugeMetricFamily(
            "panopticon_inference_inflight",
            "Generations registered with the reaper", value=r.reaper.in_flight)
        yield CounterMetricFamily(
            "panopticon_zombies_reaped", "Tasks killed by the Zombie Reaper",
            value=r.reaper.reaped_total)

        # ---- Graph write path --------------------------------------------------
        yield CounterMetricFamily(
            "panopticon_graph_submitted", "Mutations submitted to the writer queue",
            value=r.graph_queue.submitted)
        yield CounterMetricFamily(
            "panopticon_graph_dropped",
            "Mutations dropped by queue backpressure",
            value=r.graph_queue.dropped)
        yield CounterMetricFamily(
            "panopticon_graph_batches", "Batches committed to Neo4j",
            value=r.graph.batches_written)
        yield CounterMetricFamily(
            "panopticon_graph_mutations", "Mutations committed to Neo4j",
            value=r.graph.mutations_written)
        yield CounterMetricFamily(
            "panopticon_graph_dead_lettered",
            "Mutations persisted to the dead-letter file after retry exhaustion",
            value=getattr(r.graph, "dead_lettered", 0))

        # ---- Memory (optional) ----------------------------------------------
        if r.memory is not None:
            m = r.memory
            yield CounterMetricFamily(
                "panopticon_memory_retrievals", "Hybrid retrievals attempted",
                value=m.retrievals)
            yield CounterMetricFamily(
                "panopticon_memory_retrieval_timeouts",
                "Retrievals that blew the latency budget (failed open)",
                value=m.retrieval_timeouts)
            yield CounterMetricFamily(
                "panopticon_memory_retrieval_failures",
                "Retrievals that errored (failed open)",
                value=m.retrieval_failures)
            yield CounterMetricFamily(
                "panopticon_memories_stored", "Dual-writes completed",
                value=m.memories_stored)
            rq = GaugeMetricFamily(
                "panopticon_memory_retrieval_ms",
                "Retrieval latency over a rolling window", labels=["quantile"])
            rs = sorted(m.retrieval_ms)
            rq.add_metric(["0.5"], _quantile(rs, 0.5))
            rq.add_metric(["0.95"], _quantile(rs, 0.95))
            yield rq

        # ---- Glasshouse (optional) ----------------------------------------
        if r.broadcaster is not None:
            b = r.broadcaster
            yield CounterMetricFamily(
                "panopticon_state_frames_captured",
                "World-state frames packed at the tick tail",
                value=b.frames_captured)
            yield CounterMetricFamily(
                "panopticon_state_frames_sent",
                "Frames delivered to the LiveKit data channel",
                value=b.frames_sent)
            yield CounterMetricFamily(
                "panopticon_state_frames_superseded",
                "Frames coalesced by latest-wins (network slower than 20 Hz)",
                value=b.frames_superseded)
            yield CounterMetricFamily(
                "panopticon_state_send_failures",
                "Frame sends dropped fail-open", value=b.send_failures)
        if r.oracle is not None:
            o = r.oracle
            yield CounterMetricFamily(
                "panopticon_oracle_events", "Feed events ingested",
                value=o.events_ingested)
            yield CounterMetricFamily(
                "panopticon_oracle_malformed", "Feed events rejected",
                value=o.events_malformed)
            yield CounterMetricFamily(
                "panopticon_oracle_agents_notified",
                "Agent perception tasks injected by the Oracle",
                value=o.agents_notified)

        # ---- Embodiment & agency (optional) --------------------------------
        if r.uplink is not None:
            yield CounterMetricFamily(
                "panopticon_avatar_packets_applied",
                "UE5 position packets applied to the human avatar",
                value=r.uplink.packets_applied)
            yield CounterMetricFamily(
                "panopticon_avatar_packets_rejected",
                "Stale or malformed uplink packets dropped",
                value=r.uplink.packets_stale + r.uplink.packets_malformed)
        if r.action_router is not None:
            a = r.action_router
            yield CounterMetricFamily(
                "panopticon_actions_routed",
                "Agent actions accepted onto the command board",
                value=a.actions_routed)
            yield CounterMetricFamily(
                "panopticon_actions_rejected",
                "Actions refused (hallucinated target, bad tool, self)",
                value=a.actions_rejected)
            yield CounterMetricFamily(
                "panopticon_action_parse_fallbacks",
                "Generations degraded to speech-only on parse failure",
                value=a.parse_fallbacks)
        if r.engine is not None and hasattr(r.engine, "intent_stats"):
            s = r.engine.intent_stats
            yield CounterMetricFamily(
                "panopticon_intents_resolved",
                "Entity interactions applied", value=s.resolved)
            yield CounterMetricFamily(
                "panopticon_intents_contested_losses",
                "Eligible actors beaten by a nearer one", value=s.contested_losses)
            yield CounterMetricFamily(
                "panopticon_intents_rejected",
                "Entity interactions refused", value=s.rejected)
        if r.goal_daemon is not None:
            g = r.goal_daemon
            yield CounterMetricFamily(
                "panopticon_goals_set", "Goals written to the board",
                value=g.goals_set)
            yield CounterMetricFamily(
                "panopticon_goals_cleared", "Goals retired", value=g.goals_cleared)
            yield CounterMetricFamily(
                "panopticon_goal_parse_failures",
                "Goal formulations discarded fail-open", value=g.parse_failures)
        if r.broadcaster is not None and hasattr(r.broadcaster,
                                                 "entity_deltas_sent"):
            yield CounterMetricFamily(
                "panopticon_entity_deltas_sent",
                "Entity records shipped on the reliable plane",
                value=r.broadcaster.entity_deltas_sent)
        if r.engine is not None:
            yield CounterMetricFamily(
                "panopticon_commands_completed",
                "Kinematic commands that reached their target",
                value=r.engine.commands_completed)

        # ---- Navigation (optional) ------------------------------------------
        if r.navigation is not None:
            nv = r.navigation
            yield CounterMetricFamily(
                "panopticon_plans_completed",
                "Q1 path plans attached to live commands",
                value=nv.plans_completed)
            yield CounterMetricFamily(
                "panopticon_plans_fallback",
                "Plans degraded to straight-line (bulkhead/deadline/unreachable)",
                value=nv.plans_fallback_saturated
                + nv.plans_fallback_deadline + nv.plans_unreachable)
            yield CounterMetricFamily(
                "panopticon_collisions_resolved",
                "Circle overlaps corrected by the synchronous pass",
                value=nv.collisions_resolved)

        # ---- Persistence & Director (optional) --------------------------------
        if r.snapshotter is not None:
            sn = r.snapshotter
            yield CounterMetricFamily(
                "panopticon_snapshots_flushed",
                "World snapshots durably written", value=sn.snapshots_flushed)
            yield CounterMetricFamily(
                "panopticon_snapshot_failures",
                "Snapshot flushes that failed fail-open",
                value=sn.flush_failures)
            yield GaugeMetricFamily(
                "panopticon_snapshot_last_tick",
                "Tick of the most recent durable snapshot",
                value=max(0, sn.last_flushed_tick))
        if r.director is not None:
            dd = r.director
            yield CounterMetricFamily(
                "panopticon_director_relics_spawned",
                "Scarcity injected by the economy lever",
                value=dd.relics_spawned)
            yield CounterMetricFamily(
                "panopticon_director_rotations",
                "World-condition epoch rotations", value=dd.rotations)
            yield CounterMetricFamily(
                "panopticon_director_rotations_deferred",
                "Rotations deferred by the congestion gate",
                value=dd.rotations_deferred)
        if r.world is not None:
            reg = r.world.entities
            yield CounterMetricFamily(
                "panopticon_cosigns_completed",
                "Multi-signature cooperative actions executed",
                value=reg.cosigns_completed)
            yield CounterMetricFamily(
                "panopticon_cosigns_expired",
                "Cooperative signatures that lapsed unpartnered",
                value=reg.cosigns_expired)

        # ---- Consolidation daemon (optional) -----------------------------------
        if r.daemon is not None:
            d = r.daemon
            yield CounterMetricFamily(
                "panopticon_daemon_sweeps", "Consolidation sweeps executed",
                value=d.sweeps)
            yield CounterMetricFamily(
                "panopticon_daemon_sweep_failures",
                "Sweeps absorbed by fail-open supervision",
                value=d.sweep_failures)
            yield CounterMetricFamily(
                "panopticon_daemon_clumps_consolidated",
                "Episodic clumps compressed to long-term nodes",
                value=d.clumps_consolidated)
            yield CounterMetricFamily(
                "panopticon_daemon_memories_tombstoned",
                "Episodic memories archived out of the hot path",
                value=d.memories_tombstoned)
            yield CounterMetricFamily(
                "panopticon_daemon_llm_parse_failures",
                "Consolidation generations with unusable JSON",
                value=d.llm_parse_failures)


class MetricsServer:
    """Serves /metrics on the engine's own event loop via aiohttp — no
    background scrape thread, no sampling loop; state is read at scrape."""

    def __init__(self, refs: EngineRefs, port: int = 9109) -> None:
        self._registry = CollectorRegistry()
        self._registry.register(PanopticonCollector(refs))
        self._port = port
        self._runner: web.AppRunner | None = None

    async def _handle(self, request: web.Request) -> web.Response:
        body = generate_latest(self._registry)
        return web.Response(body=body,
                            headers={"Content-Type":
                                     "text/plain; version=0.0.4; charset=utf-8"})

    async def run(self) -> None:
        app = web.Application()
        app.router.add_get("/metrics", self._handle)
        app.router.add_get("/healthz", lambda r: web.Response(text="ok"))
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._port)
        await site.start()
        log.info("Prometheus /metrics on :%d", self._port)
        try:
            await asyncio_forever()
        finally:
            await self._runner.cleanup()


async def asyncio_forever() -> None:
    import asyncio
    await asyncio.Event().wait()
