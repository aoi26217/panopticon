"""Phase 6 verification suite (no LiveKit server, no UE5 — the wire format,
decoupling, and injection semantics are all provable headlessly).

1. BIT-PERFECT   — every field decodes at its computed struct offset with
                   exact float32/uint values; total size = 6 + 25n.
2. ZERO-ALLOC    — the pack buffer object is reused across captures; id
                   compaction is cached; collisions raise at first sight.
3. DECOUPLING    — a 150 ms-slow network cannot bend the 20 Hz tick: frames
                   coalesce (latest-wins), overrun stays ~0.
4. FAIL-OPEN     — a hard-failing publisher: ticks unaffected, failures
                   counted, capture path never raises.
5. GLOBAL INJECT — a market crash tasks all 50 agents at Q0 with the system
                   message BYTE-IDENTICAL to SHARED_WORLD_PREFIX.
6. LOCAL INJECT  — strict Euclidean scoping incl. the boundary case
                   (3-4-5 triangle, d == r exactly → included).
7. HYGIENE       — malformed feed rows skipped+counted; global rate limit.
"""
import asyncio
import logging
import struct
import time
import types

import tick_engine as te

logging.disable(logging.ERROR)


# ---------------------------------------------------------------- fake rtc

class FakeParticipant:
    def __init__(self, delay_s=0.0, fail=False):
        self.delay_s = delay_s
        self.fail = fail
        self.sent: list[tuple[bytes, bool, str]] = []

    async def publish_data(self, payload, *, reliable=True,
                           destination_identities=(), topic=""):
        if self.fail:
            raise ConnectionError("SFU unreachable")
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        self.sent.append((bytes(payload), reliable, topic))


class FakeRoom:
    def __init__(self, delay_s=0.0, fail=False):
        self.local_participant = FakeParticipant(delay_s, fail)

    async def connect(self, url, token): pass
    async def disconnect(self): pass


def install_fake_rtc(bc_mod, delay_s=0.0, fail=False):
    room = FakeRoom(delay_s, fail)
    bc_mod.rtc = types.SimpleNamespace(Room=lambda: room)
    bc_mod.api = types.SimpleNamespace(
        AccessToken=lambda k, s: types.SimpleNamespace(
            with_identity=lambda i: types.SimpleNamespace(
                with_name=lambda n: types.SimpleNamespace(
                    with_grants=lambda g: types.SimpleNamespace(
                        to_jwt=lambda: "jwt")))),
        VideoGrants=lambda **kw: None)
    return room


import broadcaster as bc          # noqa: E402  (after fake helpers defined)
import oracle as orc              # noqa: E402


def f32(v: float) -> float:
    """Round-trip a python float through float32 — the wire's precision."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


def make_world(n=4, size=100.0):
    cfg = te.EngineConfig(world_size=size)
    w = te.World(cfg, n)
    return cfg, w


# ------------------------------------------------------------------- tests

async def test_bit_perfect():
    room = install_fake_rtc(bc)
    b = bc.StateBroadcaster("wss://x", "k", "s",
                            speaking_probe=lambda aid: aid == "agent_001")
    await b.connect()
    _, world = make_world(3)
    world.agents[0].x, world.agents[0].y = 12.5, -7.25
    world.agents[0].vx, world.agents[0].vy = 1.5, -0.5
    world.agents[1].degraded = True                        # flag bit1
    b.capture(4242, world.agents)
    sender = asyncio.create_task(b.run_sender())
    await asyncio.sleep(0.05)
    sender.cancel()

    (payload, reliable, topic), = room.local_participant.sent
    assert reliable is False and topic == bc.DATA_TOPIC
    assert len(payload) == bc.HEADER_SIZE + 3 * bc.AGENT_SIZE == 81

    tick, count = struct.unpack_from(bc.HEADER_FMT, payload, 0)
    assert (tick, count) == (4242, 3)
    for i, agent in enumerate(world.agents):
        off = bc.HEADER_SIZE + i * bc.AGENT_SIZE
        wid, x, y, vx, vy, flags = struct.unpack_from(bc.AGENT_FMT, payload, off)
        assert wid == bc.compact_agent_id(agent.id)
        assert (x, y, vx, vy) == (f32(agent.x), f32(agent.y),
                                  f32(agent.vx), f32(agent.vy))
        expected_flags = (bc.FLAG_SPEAKING if agent.id == "agent_001" else 0) \
                       | (bc.FLAG_DEGRADED if agent.degraded else 0)
        assert flags == expected_flags, (agent.id, flags, expected_flags)
    assert bc.compact_agent_id("agent_007") == b"agent007"   # 8 exact, no pad
    print("PASS bit-perfect: 81-byte frame, every offset exact, flags correct")


async def test_zero_alloc_and_collisions():
    install_fake_rtc(bc)
    b = bc.StateBroadcaster("wss://x", "k", "s")
    _, world = make_world(4)
    buf_id = id(b._buf)
    for tick in range(200):
        b.capture(tick, world.agents)
    assert id(b._buf) == buf_id                    # same buffer, 200 captures
    assert len(b._id_cache) == 4                   # compaction cached once
    assert b.frames_superseded == 199              # no sender: latest-wins
    try:
        b._wire_id("XXagent_001")                  # compacts to 'agent001' too
        raise AssertionError("collision not detected")
    except bc.AgentIdOverflowError:
        pass
    print("PASS zero-alloc: buffer reused ×200, ids cached, collision raises")


async def test_decoupling_slow_network():
    room = install_fake_rtc(bc, delay_s=0.15)      # 3× slower than tick period
    b = bc.StateBroadcaster("wss://x", "k", "s")
    await b.connect()
    cfg, world = make_world(50, size=18.0)
    engine = te.TickEngine(cfg, world, te.MLFQScheduler(cfg),
                           te.AIMDLimiter(cfg))
    engine.set_tick_hook(b.make_tick_hook())
    tasks = [asyncio.create_task(engine.run()),
             asyncio.create_task(b.run_sender())]
    await asyncio.sleep(2.0)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    assert engine.max_tick_overrun_ms < 5.0, engine.max_tick_overrun_ms
    assert engine.tick_count >= 38                 # ~20 Hz held
    assert b.frames_captured >= 38
    assert b.frames_sent <= 15                     # network-paced
    assert b.frames_superseded >= 20               # coalescing engaged
    last_tick, _ = struct.unpack_from(bc.HEADER_FMT,
                                      room.local_participant.sent[-1][0])
    assert engine.tick_count - last_tick <= 4      # latest-wins freshness
    print(f"PASS decoupling: {engine.tick_count} ticks @ overrun "
          f"{engine.max_tick_overrun_ms:.1f}ms, sent {b.frames_sent}, "
          f"superseded {b.frames_superseded}, freshness gap "
          f"{engine.tick_count - last_tick}")


async def test_fail_open_send():
    install_fake_rtc(bc, fail=True)
    b = bc.StateBroadcaster("wss://x", "k", "s")
    await b.connect()
    cfg, world = make_world(10)
    engine = te.TickEngine(cfg, world, te.MLFQScheduler(cfg),
                           te.AIMDLimiter(cfg))
    engine.set_tick_hook(b.make_tick_hook())
    tasks = [asyncio.create_task(engine.run()),
             asyncio.create_task(b.run_sender())]
    await asyncio.sleep(1.0)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert b.send_failures > 0
    assert engine.tick_hook_failures == 0          # capture never raised
    assert engine.max_tick_overrun_ms < 5.0
    assert engine.tick_count >= 18
    print(f"PASS fail-open: {b.send_failures} send failures absorbed, "
          f"tick loop untouched")


async def test_global_injection():
    cfg, world = make_world(50, size=18.0)
    scheduler = te.MLFQScheduler(cfg)
    o = orc.Oracle(orc.StaticFeed([]), scheduler, world)
    n = o.inject(orc.WorldEvent(scope="global",
                                text="The grain market has crashed"))
    assert n == 50 and o.agents_notified == 50
    q0, q1, q2 = scheduler.depths()
    assert (q0, q1, q2) == (50, 0, 0)              # all INTERACTIVE
    seen = set()
    while True:
        task = None
        if scheduler._queues[0]:
            task = scheduler._queues[0].popleft()
        if task is None:
            break
        assert task.level == te.QueueLevel.INTERACTIVE
        assert task.messages[0]["content"] == te.SHARED_WORLD_PREFIX  # BYTE-identical
        assert "grain market has crashed" in task.messages[-1]["content"]
        assert task.messages[-1]["content"] != te.SHARED_WORLD_PREFIX
        seen.add(task.agent_id)
    assert len(seen) == 50                          # each agent exactly once
    assert all(a.last_reasoned_at > 0 for a in world.agents)  # cooldown reset
    print("PASS global: 50 Q0 tasks, prefix byte-stable, event in leaf only")


async def test_local_injection_euclidean():
    cfg, world = make_world(6, size=100.0)
    ax = [(10.0, 10.0),    # d=0            → in
          (13.0, 14.0),    # d=5.0 exactly  → in (boundary inclusive, 3-4-5)
          (10.0, 14.999),  # d=4.999        → in
          (10.0, 15.001),  # d=5.001        → OUT
          (15.1, 10.0),    # d=5.1          → OUT
          (80.0, 80.0)]    # far            → OUT
    for agent, (x, y) in zip(world.agents, ax):
        agent.x, agent.y = x, y
    scheduler = te.MLFQScheduler(cfg)
    o = orc.Oracle(orc.StaticFeed([]), scheduler, world,
                   cfg=orc.OracleConfig(min_local_interval_s=0.0))
    n = o.inject(orc.WorldEvent(scope="local", text="Lightning strikes!",
                                x=10.0, y=10.0, radius=5.0))
    tasked = {scheduler._queues[0].popleft().agent_id
              for _ in range(len(scheduler._queues[0]))} \
             | ({scheduler._queues[0].popleft().agent_id}
                if scheduler._queues[0] else set())
    expected = {world.agents[0].id, world.agents[1].id, world.agents[2].id}
    assert n == 3 and tasked == expected, (n, tasked, expected)
    print("PASS local: strict Euclidean radius, d==r boundary included, "
          "d=5.001 excluded")


async def test_hygiene():
    cfg, world = make_world(5)
    scheduler = te.MLFQScheduler(cfg)
    rows = [
        {"scope": "global"},                                    # no text
        {"scope": "local", "text": "x", "x": 1, "y": 2},        # no radius
        {"scope": "local", "text": "x", "x": "a", "y": 2, "radius": 3},
        {"scope": "sideways", "text": "x"},                     # bad scope
        {"scope": "local", "text": "x", "x": 1, "y": 2, "radius": -4},
        {"scope": "global", "text": "  Real event  ", "id": 7},  # valid
    ]
    o = orc.Oracle(orc.StaticFeed(rows), scheduler, world)
    runner = asyncio.create_task(o.run())
    await asyncio.sleep(0.1)
    runner.cancel()
    assert o.events_ingested == 6
    assert o.events_malformed == 5
    assert o.global_injections == 1
    # Rate limit: an immediate second global is refused.
    assert o.inject(orc.WorldEvent(scope="global", text="again")) == 0
    assert o.events_rate_limited == 1
    print("PASS hygiene: 5 malformed skipped, 1 injected, rate limit engaged")


async def main():
    await test_bit_perfect()
    await test_zero_alloc_and_collisions()
    await test_decoupling_slow_network()
    await test_fail_open_send()
    await test_global_injection()
    await test_local_injection_euclidean()
    await test_hygiene()
    print("\nALL PHASE 6 TESTS PASSED")


asyncio.run(main())
