"""
Project Panopticon — Phase 10: Absolute Persistence
====================================================
persistence.py

Survive server death. The durable tier (Neo4j graph, Qdrant vectors) already
survives by construction — every write since Phase 1 has been idempotent and
replayable. What dies with the process is the EPHEMERAL tier: agent
positions and velocities, entity locations and inventory, the command board,
the goal board. This module snapshots that tier and resurrects it.

CAPTURE (tick side — the familiar latest-wins pattern from Phase 6)
  A tail hook fires every `snapshot_every_ticks` ticks and serializes the
  world into a plain dict — synchronous, coherent by the event-loop theorem
  (no coroutine can observe the world mid-capture). The dict lands in a
  1-deep latest-wins slot; if the disk is slower than the capture cadence,
  intermediate snapshots are superseded, never queued. Amortized tick cost:
  one dict build per N ticks, zero I/O ever.

FLUSH (async side)
  A supervised task drains the slot: json.dumps → write to `<path>.tmp` →
  os.replace (POSIX-atomic) → fsync. A crash mid-flush leaves the previous
  snapshot intact; there is no partially-written state file, ever.

RESTORE + RECONCILIATION CONTRACT
  restore_world() rebuilds the ephemeral tier exactly: agents (including the
  human avatar), entities with state/ownership/version, standing goals, and
  kinematic commands — re-issued through engine.issue_command so Q1 path
  plans regenerate naturally. Reconciliation with the durable tier is BY KEY,
  not by copy: agent ids and entity ids are the join keys into Neo4j/Qdrant,
  and because every graph mutation was idempotent (Phase 3/4 design), nothing
  in the durable tier needs rewriting on boot — memories, beliefs, and
  relationships are simply THERE when the resurrected agents next retrieve.
  Deliberately NOT persisted: baked waypoints (stale geometry; replanned on
  re-issue), pending cosigns (sub-second TTL), and in-flight LLM generations
  (their AIMD slots died with the process; the MLFQ refills organically).

Fail-open throughout: a failed capture or flush is counted and skipped; a
corrupt or missing state file boots a fresh world. Persistence is an
upgrade, never a dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("panopticon.persistence")

SNAPSHOT_VERSION = 1


class Snapshotter:
    def __init__(self, path: str | Path, *, every_ticks: int = 100,
                 goal_board: dict[str, str] | None = None) -> None:
        self._path = Path(path)
        self._every = max(1, every_ticks)
        self._goal_board = goal_board if goal_board is not None else {}
        self._engine: Any = None
        self._pending: dict[str, Any] | None = None
        self._ready = asyncio.Event()
        # Telemetry
        self.snapshots_captured = 0
        self.snapshots_flushed = 0
        self.snapshots_superseded = 0
        self.flush_failures = 0
        self.last_flushed_tick = -1

    def bind(self, engine: Any) -> None:
        self._engine = engine

    # ------------------------------------------------- tick side (hot path)

    def make_tick_hook(self) -> Callable[[int, Any], None]:
        def hook(tick: int, world: Any) -> None:
            if tick % self._every:
                return
            self.capture(tick, world)
        return hook

    def capture(self, tick: int, world: Any) -> None:
        """Coherent by construction: runs synchronously at the tick tail."""
        engine = self._engine
        snap: dict[str, Any] = {
            "version": SNAPSHOT_VERSION,
            "tick": tick,
            "saved_at": time.time(),
            "agents": [
                {"id": a.id, "x": a.x, "y": a.y, "vx": a.vx, "vy": a.vy,
                 "is_human": a.is_human, "degraded": a.degraded}
                for a in world.agents
            ],
            "entities": [
                {"id": e.id, "kind": e.kind, "x": e.x, "y": e.y,
                 "state": e.state, "owner_id": e.owner_id,
                 "version": e.version}
                for e in world.entities.all()
            ],
            "commands": {
                aid: {"tool": c.tool, "target_id": c.target_id,
                      "ttl_s": c.ttl_s}
                for aid, c in (engine.command_board.items() if engine else ())
            },
            "goals": dict(self._goal_board),
        }
        if self._pending is not None:
            self.snapshots_superseded += 1
        self._pending = snap
        self.snapshots_captured += 1
        self._ready.set()

    # ---------------------------------------------------- flusher (I/O side)

    async def run_flusher(self) -> None:
        while True:
            await self._ready.wait()
            self._ready.clear()
            snap = self._pending
            self._pending = None
            if snap is None:
                continue
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, self._write, snap)
                self.snapshots_flushed += 1
                self.last_flushed_tick = snap["tick"]
            except asyncio.CancelledError:
                raise
            except Exception:
                self.flush_failures += 1
                log.exception("Snapshot flush failed — simulation unaffected")

    def _write(self, snap: dict[str, Any]) -> None:
        """Atomic: tmp + fsync + os.replace. Runs in the default executor so
        a slow disk never touches the event loop, let alone the tick."""
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(snap, fh, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._path)


# ============================================================================
# Restore
# ============================================================================

def restore_world(path: str | Path, world: Any, engine: Any,
                  goal_board: dict[str, str]) -> int | None:
    """Hydrate the ephemeral tier from a snapshot. Returns the resurrected
    tick number, or None (fail-open: caller proceeds with the fresh world).
    Must run BEFORE the engine tasks start — it rebuilds world.agents
    wholesale and re-issues commands."""
    from tick_engine import Agent, KinematicCommand   # local: avoid cycles

    p = Path(path)
    if not p.exists():
        return None
    try:
        snap = json.loads(p.read_text(encoding="utf-8"))
        if snap.get("version") != SNAPSHOT_VERSION:
            log.warning("Snapshot version %r unsupported — fresh boot",
                        snap.get("version"))
            return None

        # 1) Agents: rebuild the roster exactly as it died (this supersedes
        #    the fresh random spawn — snapshot count wins over --agents).
        world.agents = [
            Agent(id=r["id"], x=r["x"], y=r["y"], vx=r["vx"], vy=r["vy"],
                  is_human=r["is_human"], degraded=r["degraded"])
            for r in snap["agents"]
        ]

        # 2) Entities: spawn-or-overwrite, preserving version watermarks so
        #    the broadcaster's delta plane resumes without a spurious storm.
        reg = world.entities
        for r in snap["entities"]:
            e = reg.find(r["id"]) or reg.spawn(r["id"], r["kind"],
                                               r["x"], r["y"])
            e.kind = r["kind"]
            e.x, e.y = r["x"], r["y"]
            e.state = dict(r["state"])
            e.owner_id = r["owner_id"]
            e.version = r["version"]

        # 3) Goals: the boards are plain dicts — atomic swap of contents.
        goal_board.clear()
        goal_board.update(snap.get("goals", {}))

        # 4) Commands: RE-ISSUED, not copied — issue_command re-requests Q1
        #    path plans against the resurrected (identical) geometry, and
        #    waypoints regenerate rather than replaying stale bakes.
        for aid, c in snap.get("commands", {}).items():
            if world.find(aid) is not None and world.find(c["target_id"]):
                engine.issue_command(aid, KinematicCommand(
                    tool=c["tool"], target_id=c["target_id"],
                    ttl_s=c["ttl_s"]))

        log.info("Resurrected tick %d: %d agents, %d entities, %d goals, "
                 "%d commands (durable tier joins by id — nothing rewritten)",
                 snap["tick"], len(snap["agents"]), len(snap["entities"]),
                 len(snap.get("goals", {})), len(snap.get("commands", {})))
        return int(snap["tick"])
    except Exception:
        log.exception("Snapshot restore failed — booting fresh (fail-open)")
        return None
