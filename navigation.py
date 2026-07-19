"""
Project Panopticon — Phase 9: Navigation, Dynamic NavMesh & Geometric Collisions
=================================================================================
navigation.py   (stdlib only — drop-in, zero dependencies)

Three layers, from coldest to hottest:

  NavGrid          static geometry: a flat bytearray occupancy grid over the
                   world. O(1) solidity lookups, rectangle rasterization for
                   walls/buildings, Bresenham line-of-sight for path
                   simplification and wall-slide checks.
  Pathfinder       8-connected A* over the grid with octile heuristic, no
                   corner cutting, a hard expansion budget, and COOPERATIVE
                   TIME-SLICING: `await asyncio.sleep(0)` every N expansions,
                   so a long search interleaves with the 20 Hz tick instead
                   of starving it (everything shares one event loop).
  Collision layer  a reusable spatial hash + a SYNCHRONOUS zero-await
                   resolution pass run by the TickEngine right after
                   world.step(): overlapping circles (d < 2R) are pushed
                   apart along the collision normal; solid cells eject
                   intruders. Same lock-free theorem as every hot pass since
                   Phase 7: no awaits between first and last instruction.

SCHEDULING (deliberate deviation from the spec, stated plainly)
  Path requests ARE Q1 AgentTasks — they wait behind live speech in the
  MLFQ exactly as specified, and their queue depth is visible in telemetry.
  But they do NOT consume AIMD slots: AIMD is GPU admission control, its
  additive/multiplicative dynamics are calibrated on TTFT, and letting CPU
  planners occupy those slots would both starve inference and corrupt the
  congestion signal the Phase 5 crucible certified. Planners get their own
  bulkhead instead: a fixed semaphore (`max_concurrent_plans`), a per-plan
  deadline, and an expansion budget. Saturation, timeout, budget blowout,
  or an unreachable goal all degrade to the Phase 7 behavior that is ALREADY
  running: straight-line steering. A plan is an upgrade, never a dependency.

ALLOCATION DISCIPLINE (the honest Python version of "zero-alloc")
  CPython cannot do literal zero-allocation loops; what we can do — and
  measure — is keep the synchronous passes free of per-tick data-structure
  churn: the occupancy grid is one bytearray allocated at startup; spatial-
  hash buckets are dict-of-list with lists CLEARED and reused (capacity
  retained) rather than rebuilt; inner loops work on local floats and ints,
  no tuples, no comprehensions. The enforcement mechanism is the same one
  every phase has used: tick_max_overrun_ms stays ~0 under load, or the
  test suite fails.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

log = logging.getLogger("panopticon.navigation")

_SQRT2 = math.sqrt(2.0)


# ============================================================================
# 1. NavGrid — static occupancy
# ============================================================================

class NavGrid:
    def __init__(self, world_size: float, cell_size: float = 1.0) -> None:
        self.cell = cell_size
        self.n = max(1, int(math.ceil(world_size / cell_size)))
        self.world_size = world_size
        self._solid = bytearray(self.n * self.n)      # THE grid: one alloc
        self.obstacle_count = 0

    # ---- authoring ---------------------------------------------------------

    def add_obstacle_rect(self, x0: float, y0: float,
                          x1: float, y1: float) -> None:
        """Rasterize an axis-aligned solid block (wall, building)."""
        cx0 = max(0, int(x0 / self.cell))
        cy0 = max(0, int(y0 / self.cell))
        cx1 = min(self.n - 1, int(x1 / self.cell))
        cy1 = min(self.n - 1, int(y1 / self.cell))
        for cy in range(cy0, cy1 + 1):
            base = cy * self.n
            for cx in range(cx0, cx1 + 1):
                self._solid[base + cx] = 1
        self.obstacle_count += 1

    # ---- O(1) queries (hot path) --------------------------------------------

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return (min(self.n - 1, max(0, int(x / self.cell))),
                min(self.n - 1, max(0, int(y / self.cell))))

    def solid_cell(self, cx: int, cy: int) -> bool:
        if cx < 0 or cy < 0 or cx >= self.n or cy >= self.n:
            return True                                # out of world = solid
        return self._solid[cy * self.n + cx] != 0

    def solid_at(self, x: float, y: float) -> bool:
        return self.solid_cell(int(x / self.cell), int(y / self.cell))

    def line_of_sight(self, ax: float, ay: float,
                      bx: float, by: float) -> bool:
        """Conservative supercover walk from A to B: True iff no solid cell
        is crossed. Used by path simplification and local steering checks."""
        cx, cy = int(ax / self.cell), int(ay / self.cell)
        tx, ty = int(bx / self.cell), int(by / self.cell)
        dx, dy = abs(tx - cx), abs(ty - cy)
        sx = 1 if tx > cx else -1
        sy = 1 if ty > cy else -1
        err = dx - dy
        while True:
            if self.solid_cell(cx, cy):
                return False
            if cx == tx and cy == ty:
                return True
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy


# ============================================================================
# 2. Spatial hash — reusable buckets, O(1) neighborhood
# ============================================================================

class SpatialHash:
    """Flat bucket grid rebuilt in place each tick. Buckets persist across
    rebuilds (list.clear keeps capacity), so steady-state rebuilds allocate
    nothing new. Cell size should be >= the largest query radius so a 3×3
    neighborhood is a complete cover."""

    def __init__(self, world_size: float, cell_size: float) -> None:
        self.cell = cell_size
        self.n = max(1, int(math.ceil(world_size / cell_size)))
        self._buckets: dict[int, list[Any]] = {}

    def rebuild(self, agents: Iterable[Any]) -> None:
        for bucket in self._buckets.values():
            bucket.clear()
        n = self.n
        inv = 1.0 / self.cell
        for a in agents:
            cx = int(a.x * inv)
            cy = int(a.y * inv)
            if cx < 0: cx = 0
            elif cx >= n: cx = n - 1
            if cy < 0: cy = 0
            elif cy >= n: cy = n - 1
            key = cy * n + cx
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = []
                self._buckets[key] = bucket
            bucket.append(a)

    def neighborhood(self, x: float, y: float) -> Iterable[Any]:
        """All agents in the 3×3 cells around (x, y)."""
        n = self.n
        cx = int(x / self.cell)
        cy = int(y / self.cell)
        for oy in (-1, 0, 1):
            ny = cy + oy
            if ny < 0 or ny >= n:
                continue
            base = ny * n
            for ox in (-1, 0, 1):
                nx = cx + ox
                if nx < 0 or nx >= n:
                    continue
                bucket = self._buckets.get(base + nx)
                if bucket:
                    yield from bucket

    def pairs_within(self, radius: float) -> list[tuple[Any, Any]]:
        """All unordered agent pairs with d <= radius. Grid-accelerated
        replacement for the Phase 0 O(n²) proximity scan."""
        r2 = radius * radius
        out: list[tuple[Any, Any]] = []
        seen: set[tuple[int, int]] = set()
        for bucket in self._buckets.values():
            for a in bucket:
                for b in self.neighborhood(a.x, a.y):
                    if b is a:
                        continue
                    pid = (id(a), id(b)) if id(a) < id(b) else (id(b), id(a))
                    if pid in seen:
                        continue
                    seen.add(pid)
                    dx = a.x - b.x
                    dy = a.y - b.y
                    if dx * dx + dy * dy <= r2:
                        out.append((a, b))
        return out


# ============================================================================
# 3. A* — cooperative, budgeted, simplified
# ============================================================================

@dataclass(frozen=True)
class PlannerConfig:
    max_concurrent_plans: int = 4       # the bulkhead (NOT AIMD slots)
    plan_deadline_s: float = 0.5        # late plans lose to straight-line
    max_expansions: int = 6000          # hard A* budget
    yield_every: int = 256              # expansions between event-loop yields
    waypoint_min_gap: float = 0.75      # post-simplification spacing floor


class Pathfinder:
    def __init__(self, grid: NavGrid, cfg: PlannerConfig) -> None:
        self._grid = grid
        self._cfg = cfg

    async def find_path(self, sx: float, sy: float, gx: float, gy: float
                        ) -> list[tuple[float, float]] | None:
        """8-connected A*, octile heuristic, no corner cutting; yields to the
        event loop every `yield_every` expansions so the 20 Hz heartbeat is
        never starved by a long search. Returns world-space waypoints (line-
        of-sight simplified) or None (unreachable / budget exceeded)."""
        grid = self._grid
        n = grid.n
        start = grid.cell_of(sx, sy)
        goal = grid.cell_of(gx, gy)
        if grid.solid_cell(*goal) or grid.solid_cell(*start):
            return None
        if start == goal:
            return [(gx, gy)]

        gxc, gyc = goal
        open_heap: list[tuple[float, int, int, int]] = []
        g_cost: dict[int, float] = {start[1] * n + start[0]: 0.0}
        came: dict[int, int] = {}
        heapq.heappush(open_heap, (0.0, 0, start[0], start[1]))
        counter = 1
        expansions = 0

        while open_heap:
            _f, _c, cx, cy = heapq.heappop(open_heap)
            if (cx, cy) == goal:
                return self._reconstruct(came, n, cx, cy, gx, gy)
            expansions += 1
            if expansions > self._cfg.max_expansions:
                return None                      # budget blown → fallback
            if expansions % self._cfg.yield_every == 0:
                await asyncio.sleep(0)           # let the heartbeat breathe
            here = cy * n + cx
            base_g = g_cost[here]
            for ox, oy, step in ((1, 0, 1.0), (-1, 0, 1.0),
                                 (0, 1, 1.0), (0, -1, 1.0),
                                 (1, 1, _SQRT2), (1, -1, _SQRT2),
                                 (-1, 1, _SQRT2), (-1, -1, _SQRT2)):
                nx2, ny2 = cx + ox, cy + oy
                if grid.solid_cell(nx2, ny2):
                    continue
                if ox and oy and (grid.solid_cell(cx + ox, cy)
                                  or grid.solid_cell(cx, cy + oy)):
                    continue                     # no corner cutting
                key = ny2 * n + nx2
                ng = base_g + step
                if ng < g_cost.get(key, float("inf")):
                    g_cost[key] = ng
                    came[key] = here
                    dx = abs(nx2 - gxc)
                    dy = abs(ny2 - gyc)
                    h = (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)   # octile
                    heapq.heappush(open_heap, (ng + h, counter, nx2, ny2))
                    counter += 1
        return None                              # exhausted: unreachable

    def _reconstruct(self, came: dict[int, int], n: int,
                     cx: int, cy: int, gx: float, gy: float
                     ) -> list[tuple[float, float]]:
        cells: list[tuple[int, int]] = [(cx, cy)]
        key = cy * n + cx
        while key in came:
            key = came[key]
            cells.append((key % n, key // n))
        cells.reverse()
        half = self._grid.cell * 0.5
        pts = [(c[0] * self._grid.cell + half, c[1] * self._grid.cell + half)
               for c in cells]
        pts[-1] = (gx, gy)                       # land exactly on the goal
        return self._simplify(pts)

    def _simplify(self, pts: list[tuple[float, float]]
                  ) -> list[tuple[float, float]]:
        """Greedy line-of-sight string pulling: keep a waypoint only when the
        straight segment to the next candidate crosses geometry."""
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        anchor = 0
        i = 1
        while i < len(pts) - 1:
            if not self._grid.line_of_sight(pts[anchor][0], pts[anchor][1],
                                            pts[i + 1][0], pts[i + 1][1]):
                out.append(pts[i])
                anchor = i
            i += 1
        out.append(pts[-1])
        # Drop near-duplicate points (sub-gap jitter from rasterization).
        gap2 = self._cfg.waypoint_min_gap ** 2
        pruned = [out[0]]
        for p in out[1:]:
            dx = p[0] - pruned[-1][0]
            dy = p[1] - pruned[-1][1]
            if dx * dx + dy * dy >= gap2 or p is out[-1]:
                pruned.append(p)
        return pruned


# ============================================================================
# 4. NavigationService — the Q1 face + the synchronous passes
# ============================================================================

class NavigationService:
    """What the engine and dispatcher talk to.

    Async side (Q1 tasks routed here by the dispatcher): handle_task() runs
    one plan under the bulkhead + deadline and, on success, attaches the
    waypoints to the agent's still-live KinematicCommand.

    Sync side (called by the TickEngine, zero awaits): rebuild() the spatial
    hash, proximity_pairs() for the trigger scan, collision_pass() for the
    geometric push-apart, and blocked()/slide checks for wall physics.
    """

    # CONFIG INVARIANT: EngineConfig.arrival_radius MUST exceed
    # 2 * agent_radius, or the collision pass and the arrival check fight at
    # the boundary — a mover gets pushed out of arrival range every tick it
    # enters it (geometric livelock). Defaults: 1.2 > 2 × 0.4.
    def __init__(self, grid: NavGrid, cfg: PlannerConfig | None = None,
                 agent_radius: float = 0.4) -> None:
        self.grid = grid
        self.cfg = cfg or PlannerConfig()
        self.agent_radius = agent_radius
        self._pathfinder = Pathfinder(grid, self.cfg)
        self._bulkhead = asyncio.Semaphore(self.cfg.max_concurrent_plans)
        self._engine: Any = None
        self._world: Any = None
        self._hash: SpatialHash | None = None
        # Telemetry
        self.plans_completed = 0
        self.plans_fallback_saturated = 0
        self.plans_fallback_deadline = 0
        self.plans_unreachable = 0
        self.collisions_resolved = 0
        self.plan_ms: list[float] = []

    def bind(self, engine: Any, world: Any) -> None:
        self._engine = engine
        self._world = world
        cell = max(world._cfg.proximity_radius, 2 * self.agent_radius, 1.0)
        self._hash = SpatialHash(world._cfg.world_size, cell)

    # ------------------------------------------------------- async (Q1) side

    async def handle_task(self, task: Any) -> None:
        """One Q1 pathfind task. Every failure mode is the same outcome:
        the straight-line steering that began the instant the command was
        issued simply continues. A plan is an upgrade."""
        agent_id = task.agent_id
        target_id = task.payload.get("target_id", "")
        if self._engine is None or self._world is None:
            return
        cmd = self._engine.command_board.get(agent_id)
        if cmd is None or cmd.target_id != target_id:
            return                                # command superseded already
        agent = self._world.find(agent_id)
        target = self._world.find(target_id)
        if agent is None or target is None:
            return

        if self._bulkhead.locked():
            self.plans_fallback_saturated += 1    # bulkhead full: stay linear
            return
        async with self._bulkhead:
            start = time.monotonic()
            try:
                async with asyncio.timeout(self.cfg.plan_deadline_s):
                    path = await self._pathfinder.find_path(
                        agent.x, agent.y, target.x, target.y)
            except (TimeoutError, asyncio.TimeoutError):
                self.plans_fallback_deadline += 1
                return
            finally:
                self.plan_ms.append((time.monotonic() - start) * 1000)
        if path is None:
            self.plans_unreachable += 1
            return
        # Attach only if the SAME command is still standing (latest-wins).
        live = self._engine.command_board.get(agent_id)
        if live is cmd:
            live.waypoints = path
            live.wp_index = 0
            self.plans_completed += 1

    # ------------------------------------------------ sync (tick-loop) side

    def rebuild(self, agents: Iterable[Any]) -> None:
        if self._hash is not None:
            self._hash.rebuild(agents)

    def proximity_pairs(self, radius: float) -> list[tuple[Any, Any]]:
        assert self._hash is not None
        return self._hash.pairs_within(radius)

    def blocked(self, x: float, y: float) -> bool:
        return self.grid.solid_at(x, y)

    def collision_pass(self, agents: list[Any]) -> None:
        """SYNCHRONOUS, zero awaits. Circle-circle: overlapping agents are
        pushed apart along the collision normal, half the penetration each
        (the human avatar is authoritative — its half is donated to the
        other party). Circle-grid: an agent inside a solid cell is ejected
        to the nearest free neighbor center."""
        if self._hash is None:
            return
        two_r = 2.0 * self.agent_radius
        two_r2 = two_r * two_r
        for a in agents:
            for b in self._hash.neighborhood(a.x, a.y):
                if b is a or id(b) < id(a):       # each pair exactly once
                    continue
                dx = b.x - a.x
                dy = b.y - a.y
                d2 = dx * dx + dy * dy
                if d2 >= two_r2 or d2 == 0.0:
                    continue
                d = math.sqrt(d2)
                push = (two_r - d) * 0.5
                nx = dx / d
                ny = dy / d
                if a.is_human:                     # wire is authoritative
                    b.x += nx * push * 2.0
                    b.y += ny * push * 2.0
                elif b.is_human:
                    a.x -= nx * push * 2.0
                    a.y -= ny * push * 2.0
                else:
                    a.x -= nx * push
                    a.y -= ny * push
                    b.x += nx * push
                    b.y += ny * push
                self.collisions_resolved += 1
            # Grid ejection: never end a tick inside a wall.
            if not a.is_human and self.grid.solid_at(a.x, a.y):
                cx, cy = self.grid.cell_of(a.x, a.y)
                for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                               (1, 1), (-1, -1), (1, -1), (-1, 1)):
                    if not self.grid.solid_cell(cx + ox, cy + oy):
                        half = self.grid.cell * 0.5
                        a.x = (cx + ox) * self.grid.cell + half
                        a.y = (cy + oy) * self.grid.cell + half
                        break
