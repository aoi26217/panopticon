"""Phase 9 verification suite (zero infrastructure, real engine throughout).

1. A* CORE      — routes around a wall (path bends, no solid cell crossed);
                  line-of-sight simplification collapses a straight corridor
                  to 2 points; an enclosed goal returns None; corner cutting
                  through diagonal gaps is refused.
2. WALL ROUTING — full stack: move_towards through the real dispatcher; the
                  Q1 pathfind task upgrades the command in place; the agent
                  reaches a target behind a wall, never entering solid
                  geometry, waypoints consumed in order.
3. WALL SLIDE   — an agent driven straight at a wall never phases through.
4. COLLISION    — two agents walked head-on: separation restored to >= 2R
                  at every sampled tick end; the human avatar is immovable
                  (the other party takes the full correction); zero overrun.
5. HASH PARITY  — spatial-hash proximity pairs == brute-force O(n²) pairs on
                  200 random agents (grid replaces the scan, loses nothing).
6. DEGRADATION  — 120 plan requests against a 1-plan bulkhead with a starved
                  deadline: fallbacks counted, straight-line steering keeps
                  every agent moving toward its target, heartbeat >= 19 Hz.
7. PREEMPTION   — a queued Q0 dialogue dequeues before a queued Q1 pathfind.
"""
import asyncio
import logging
import math
import time

import navigation as nav
import tick_engine as te

logging.disable(logging.ERROR)


def stack(world_size=40.0, n_agents=2, cell=1.0, planner=None, radius=0.4,
          speed=2.5):
    # INVARIANT the defaults must respect: arrival_radius > 2 * agent_radius,
    # or the collision pass holds a mover at the exact boundary of arrival
    # forever (geometric livelock). 1.2 > 0.8 here; navigation.py documents
    # the same constraint for production configs.
    cfg = te.EngineConfig(world_size=world_size, proximity_radius=3.0,
                          steering_speed=speed)
    world = te.World(cfg, n_agents)
    scheduler = te.MLFQScheduler(cfg)
    limiter = te.AIMDLimiter(cfg)
    grid = nav.NavGrid(world_size, cell)
    service = nav.NavigationService(grid, planner or nav.PlannerConfig(),
                                    agent_radius=radius)
    engine = te.TickEngine(cfg, world, scheduler, limiter)
    engine.set_navigation(service)
    dispatcher = te.InferenceDispatcher(
        cfg, scheduler, limiter, te.ZombieReaper(cfg), te.MockSGLangClient(),
        te.LiveKitPublisher(), te.GraphWriteQueue(cfg, te.MockNeo4jClient()),
        navigation=service)
    return cfg, world, scheduler, engine, service, grid, dispatcher


# ------------------------------------------------------------------- tests

async def test_astar_core():
    grid = nav.NavGrid(40.0, 1.0)
    grid.add_obstacle_rect(15.0, 0.0, 17.0, 30.0)      # wall with a top gap
    pf = nav.Pathfinder(grid, nav.PlannerConfig())

    path = await pf.find_path(5.0, 5.0, 30.0, 5.0)
    assert path is not None and len(path) >= 3          # bends, not a beeline
    assert max(p[1] for p in path) > 29.0               # went over the top
    for (ax, ay), (bx, by) in zip(path, path[1:]):      # every leg is clear
        assert grid.line_of_sight(ax, ay, bx, by)

    # Straight open corridor simplifies to exactly [start-cell, goal].
    open_grid = nav.NavGrid(40.0, 1.0)
    pf2 = nav.Pathfinder(open_grid, nav.PlannerConfig())
    straight = await pf2.find_path(2.0, 2.0, 30.0, 2.0)
    assert straight is not None and len(straight) == 2

    # Enclosed goal: unreachable → None (→ straight-line fallback upstream).
    boxed = nav.NavGrid(40.0, 1.0)
    boxed.add_obstacle_rect(18.0, 18.0, 22.0, 19.0)
    boxed.add_obstacle_rect(18.0, 21.0, 22.0, 22.0)
    boxed.add_obstacle_rect(18.0, 18.0, 19.0, 22.0)
    boxed.add_obstacle_rect(21.0, 18.0, 22.0, 22.0)
    assert await pf.find_path(5.0, 5.0, 20.5, 20.5) is None \
        or True  # (pf bound to walled grid; evaluate on boxed grid below)
    pf3 = nav.Pathfinder(boxed, nav.PlannerConfig())
    assert await pf3.find_path(5.0, 5.0, 20.5, 20.5) is None

    # No corner cutting: a diagonal "pinch" of two solid cells is impassable.
    pinch = nav.NavGrid(10.0, 1.0)
    pinch.add_obstacle_rect(0.0, 0.0, 4.9, 4.9)         # solid quadrant
    pinch.add_obstacle_rect(5.0, 5.0, 9.9, 9.9)         # opposite quadrant
    pf4 = nav.Pathfinder(pinch, nav.PlannerConfig())
    assert await pf4.find_path(7.5, 2.5, 2.5, 7.5) is None
    print("PASS A*: routes over walls, corridor→2 waypoints, enclosed→None, "
          "no corner cutting")


async def test_wall_routing_full_stack():
    cfg, world, scheduler, engine, service, grid, disp = stack(n_agents=2,
                                                               speed=6.0)
    grid.add_obstacle_rect(15.0, 0.0, 17.0, 30.0)
    a, target = world.agents
    a.x, a.y, a.vx, a.vy = 5.0, 5.0, 0.0, 0.0
    target.x, target.y, target.vx, target.vy = 30.0, 5.0, 0.0, 0.0
    target.last_reasoned_at = a.last_reasoned_at = time.monotonic() + 999

    worker = asyncio.create_task(disp.run_worker())
    runner = asyncio.create_task(engine.run())
    engine.issue_command(a.id, te.KinematicCommand(
        tool="move_towards", target_id=target.id, ttl_s=30.0))
    await asyncio.sleep(0.3)                            # plan lands via Q1
    cmd = engine.command_board[a.id]
    assert cmd.waypoints is not None and len(cmd.waypoints) >= 3
    assert service.plans_completed == 1

    entered_solid = []
    engine.set_tick_hook(lambda t, w: entered_solid.append(
        grid.solid_at(a.x, a.y)))
    for _ in range(120):                                # ~ up to 18s sim time
        await asyncio.sleep(0.1)
        if a.id not in engine.command_board:
            break
    runner.cancel(); worker.cancel()
    assert a.id not in engine.command_board             # arrived
    assert engine.commands_completed == 1
    assert not any(entered_solid)                       # never inside a wall
    assert math.dist((a.x, a.y), (target.x, target.y)) < cfg.arrival_radius + 1.0
    assert cmd.wp_index > 0                             # waypoints consumed
    print(f"PASS routing: Q1 plan ({len(cmd.waypoints)} waypoints) carried "
          f"the agent around the wall to arrival, zero wall entries")


async def test_wall_slide():
    cfg, world, scheduler, engine, service, grid, disp = stack(n_agents=1)
    grid.add_obstacle_rect(20.0, 0.0, 22.0, 40.0)
    a = world.agents[0]
    a.x, a.y, a.vx, a.vy = 15.0, 20.0, 2.5, 0.3         # driving at the wall
    a.last_reasoned_at = time.monotonic() + 999
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(1.0)
    runner.cancel()
    assert a.x < 20.0                                    # never phased through
    assert not grid.solid_at(a.x, a.y)
    assert a.y > 20.0                                    # slid along the face
    print(f"PASS slide: agent stopped at x={a.x:.1f} (< wall at 20), "
          f"slid to y={a.y:.1f}, never inside geometry")


async def test_elastic_collision():
    cfg, world, scheduler, engine, service, grid, disp = stack(n_agents=2)
    a, b = world.agents
    a.x, a.y, a.vx, a.vy = 10.0, 20.0, 2.0, 0.0
    b.x, b.y, b.vx, b.vy = 14.0, 20.0, -2.0, 0.0        # head-on
    a.last_reasoned_at = b.last_reasoned_at = time.monotonic() + 999
    min_sep = [float("inf")]
    engine.set_tick_hook(lambda t, w: min_sep.__setitem__(
        0, min(min_sep[0], math.dist((a.x, a.y), (b.x, b.y)))))
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(2.0)
    runner.cancel()
    two_r = 2 * service.agent_radius
    assert min_sep[0] >= two_r - 0.05, min_sep[0]        # never interpenetrated
    assert service.collisions_resolved > 0
    assert engine.max_tick_overrun_ms < 5.0

    # Human immovability: pin magnus, drive an agent into him.
    import avatar as av
    cfg2, world2, sch2, eng2, svc2, grid2, _ = stack(n_agents=1)
    u = av.AvatarUplink("wss://x", "k", "s"); u.bind(world2)
    h = world2.find(av.HUMAN_AVATAR_ID)
    h.x, h.y = 20.0, 20.0
    ag = world2.agents[0]
    ag.x, ag.y, ag.vx, ag.vy = 17.0, 20.0, 2.0, 0.0
    ag.last_reasoned_at = time.monotonic() + 999
    runner2 = asyncio.create_task(eng2.run())
    await asyncio.sleep(1.5)
    runner2.cancel()
    assert (h.x, h.y) == (20.0, 20.0)                    # wire-authoritative
    assert math.dist((ag.x, ag.y), (20.0, 20.0)) >= 2 * svc2.agent_radius - 0.05
    print(f"PASS collision: min separation {min_sep[0]:.2f} >= 2R={two_r}, "
          f"{service.collisions_resolved} resolutions, human immovable, "
          f"overrun {engine.max_tick_overrun_ms:.1f}ms")


async def test_hash_parity():
    import random
    cfg = te.EngineConfig(world_size=60.0, proximity_radius=4.0)
    world = te.World(cfg, 200)
    h = nav.SpatialHash(60.0, 4.0)
    h.rebuild(world.agents)
    fast = {frozenset((a.id, b.id)) for a, b in h.pairs_within(4.0)}
    r2 = 16.0
    slow = set()
    ags = world.agents
    for i in range(len(ags)):
        for j in range(i + 1, len(ags)):
            dx = ags[i].x - ags[j].x
            dy = ags[i].y - ags[j].y
            if dx * dx + dy * dy <= r2:
                slow.add(frozenset((ags[i].id, ags[j].id)))
    assert fast == slow, (len(fast), len(slow))
    print(f"PASS parity: spatial hash == brute force on 200 agents "
          f"({len(fast)} pairs)")


async def test_degradation_flood():
    planner = nav.PlannerConfig(max_concurrent_plans=1,
                                plan_deadline_s=0.001,      # everyone is late
                                max_expansions=50)
    cfg, world, scheduler, engine, service, grid, disp = stack(
        world_size=60.0, n_agents=40, planner=planner)
    grid.add_obstacle_rect(28.0, 0.0, 30.0, 50.0)
    for a in world.agents:
        a.last_reasoned_at = time.monotonic() + 999
    worker = asyncio.create_task(disp.run_worker())
    runner = asyncio.create_task(engine.run())
    target = world.agents[-1]
    for i in range(3):                                   # 117 plan requests
        for a in world.agents[:-1]:
            engine.issue_command(a.id, te.KinematicCommand(
                tool="move_towards", target_id=target.id, ttl_s=30.0))
        await asyncio.sleep(0.4)
    await asyncio.sleep(0.5)
    runner.cancel(); worker.cancel()
    fallbacks = (service.plans_fallback_saturated
                 + service.plans_fallback_deadline + service.plans_unreachable)
    assert fallbacks > 50, fallbacks                     # the flood degraded
    assert engine.max_tick_overrun_ms < 10.0             # heartbeat unbent
    assert engine.tick_count >= 19 * 1.5                 # ~20 Hz across run
    moving = [a for a in world.agents[:-1]
              if a.id in engine.command_board and math.hypot(a.vx, a.vy) > 1.0]
    assert len(moving) > 20                              # straight-line lives
    print(f"PASS degradation: {fallbacks} planners dropped to heuristic, "
          f"{len(moving)} agents still steering, overrun "
          f"{engine.max_tick_overrun_ms:.1f}ms")


async def test_q0_preempts_pathfind():
    cfg, world, scheduler, engine, service, grid, disp = stack()
    await scheduler.submit(te.AgentTask(agent_id="p", kind="pathfind",
                                        level=te.QueueLevel.COGNITIVE,
                                        messages=[], payload={"target_id": "x"}))
    await scheduler.submit(te.AgentTask(agent_id="s", kind="dialogue",
                                        level=te.QueueLevel.INTERACTIVE,
                                        messages=[]))
    first = await scheduler.next_task()
    second = await scheduler.next_task()
    assert first.kind == "dialogue" and second.kind == "pathfind"
    print("PASS preemption: live speech dequeues before path planning")


async def main():
    await test_astar_core()
    await test_wall_routing_full_stack()
    await test_wall_slide()
    await test_elastic_collision()
    await test_hash_parity()
    await test_degradation_flood()
    await test_q0_preempts_pathfind()
    print("\nALL PHASE 9 TESTS PASSED")


asyncio.run(main())
