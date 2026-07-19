"""Phase 7 verification suite (headless: fake LiveKit packets, scripted
schema-constrained SGLang, real TickEngine/dispatcher/scheduler throughout).

1. UPLINK       — packets move the avatar; stale seq dropped; out-of-bounds
                  clamped; velocity derived; physics never integrates the
                  human; malformed sizes counted.
2. PERCEPTION   — human near an agent: the AGENT gets a proximity task, the
                  human NEVER does (asymmetric triggering).
3. VOICE        — nearest-3 within hearing radius get Q0 tasks quoting the
                  transcript; the 4th-nearest and out-of-range agents get
                  nothing; embodied prefix byte-stable.
4. AGENCY       — full dispatcher path: schema-constrained generation →
                  intercept → say to TTS (no JSON leaked), command on the
                  board → next tick velocity points at magnus → agent
                  closes distance → arrival clears the command.
5. LOCK-FREE    — 200 rapid command replacements during a live 20 Hz run:
                  zero exceptions, zero overrun, latest command wins.
6. REJECTION    — hallucinated target / self-target / bad tool rejected;
                  malformed JSON falls back to speech (agent still talks).
"""
import asyncio
import logging
import math
import struct
import time
import types

import avatar as av
import tick_engine as te

logging.disable(logging.ERROR)


def make_uplink(world):
    u = av.AvatarUplink("wss://x", "k", "s")
    u.bind(world)
    return u


def packet(seq, x, y, topic=av.AVATAR_TOPIC, size_ok=True):
    data = struct.pack(av.UPLINK_FMT, seq, x, y)
    if not size_ok:
        data += b"\x00"
    return types.SimpleNamespace(data=data, topic=topic)


# ------------------------------------------------------------------- tests

async def test_uplink():
    cfg = te.EngineConfig(world_size=100.0)
    world = te.World(cfg, 3)
    u = make_uplink(world)
    human = world.find(av.HUMAN_AVATAR_ID)
    assert human is not None and human.is_human

    u._on_data(packet(1, 20.0, 30.0))
    assert (human.x, human.y) == (20.0, 30.0)
    await asyncio.sleep(0.05)
    u._on_data(packet(2, 21.0, 30.0))          # ~1 unit in ~50ms → vx ≈ 20
    assert human.vx > 5.0 and abs(human.vy) < 1.0
    u._on_data(packet(1, 99.0, 99.0))          # stale seq → dropped
    assert (human.x, human.y) == (21.0, 30.0) and u.packets_stale == 1
    u._on_data(packet(3, -50.0, 500.0))        # hostile input → clamped
    assert (human.x, human.y) == (0.0, 100.0)
    u._on_data(packet(4, 1, 1, size_ok=False))
    assert u.packets_malformed == 1
    u._on_data(packet(5, 1, 1, topic="other.topic"))
    assert u.packets_applied == 3              # foreign topic ignored

    # Physics must NOT integrate or bounce the human.
    human.x, human.y, human.vx, human.vy = 50.0, 50.0, 99.0, 99.0
    world.step(0.05)
    assert (human.x, human.y) == (50.0, 50.0)  # untouched by step()
    print("PASS uplink: apply/stale/clamp/malformed correct, physics hands off")


async def test_perception_asymmetry():
    cfg = te.EngineConfig(world_size=100.0, proximity_radius=5.0)
    world = te.World(cfg, 1)
    world.agents[0].x = world.agents[0].y = 40.0
    u = make_uplink(world)
    human = world.find(av.HUMAN_AVATAR_ID)
    human.x, human.y = 42.0, 40.0              # inside trigger radius
    scheduler = te.MLFQScheduler(cfg)
    engine = te.TickEngine(cfg, world, scheduler, te.AIMDLimiter(cfg))
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(0.3)
    runner.cancel()
    tasks = list(scheduler._queues[0])
    speakers = {t.agent_id for t in tasks}
    assert world.agents[0].id in speakers
    assert av.HUMAN_AVATAR_ID not in speakers  # the LLM never speaks AS Magnus
    assert any(av.HUMAN_AVATAR_ID in t.messages[-1]["content"]
               for t in tasks if t.agent_id == world.agents[0].id)
    print("PASS perception: agent perceives magnus; magnus never gets a task")


async def test_voice_nearest_k():
    cfg = te.EngineConfig(world_size=100.0)
    world = te.World(cfg, 5)
    positions = [(51, 50), (50, 53), (54, 50), (50, 57), (90, 90)]
    for a, (x, y) in zip(world.agents, positions):
        a.x, a.y = float(x), float(y)          # d = 1, 3, 4, 7, far
    u = make_uplink(world)
    human = world.find(av.HUMAN_AVATAR_ID)
    human.x = human.y = 50.0
    scheduler = te.MLFQScheduler(cfg)

    class NullSTT:
        async def push(self, pcm): pass
        async def transcripts(self):
            if False:
                yield ""
        async def close(self): pass

    v = av.VoiceIngress(NullSTT(), u, scheduler, world,
                        cfg=av.VoiceConfig(hearing_radius=8.0, max_listeners=3))
    n = v.inject_speech("Hello there, who runs this market?")
    assert n == 3                               # nearest-3 of the 4 in range
    tasked = set()
    while scheduler._queues[0]:
        t = scheduler._queues[0].popleft()
        assert t.level == te.QueueLevel.INTERACTIVE
        assert t.messages[0]["content"] == te.EMBODIED_WORLD_PREFIX
        assert '"Hello there, who runs this market?"' in t.messages[-1]["content"]
        tasked.add(t.agent_id)
    expected = {world.agents[0].id, world.agents[1].id, world.agents[2].id}
    assert tasked == expected, (tasked, expected)
    print("PASS voice: nearest-3 tasked at Q0, d=7 agent capped out, "
          "far agent silent, prefix byte-stable")


class ScriptedActionLLM:
    """Simulates schema-enforced SGLang: emits {say, action} JSON chosen by
    the leaf content, streamed in chunks. Records the response_format it was
    handed so the schema plumbing is assertable."""

    def __init__(self):
        self.formats = []

    async def stream_chat(self, model, messages, response_format=None):
        self.formats.append(response_format)
        leaf = messages[-1]["content"]
        if "speaks to you directly" in leaf:
            out = ('{"say": "Of course, let me come closer.", '
                   '"action": {"tool": "move_towards", "target_id": "magnus"}}')
        else:
            out = '{"say": "Fine weather today.", "action": null}'
        for i in range(0, len(out), 24):
            yield out[i:i + 24]
            await asyncio.sleep(0.002)


class CapturingMedia(te.LiveKitPublisher):
    def __init__(self):
        super().__init__()
        self.spoken: list[tuple[str, str]] = []

    async def publish_tokens(self, agent_id, text):
        await super().publish_tokens(agent_id, text)
        self.spoken.append((agent_id, text))


async def build_embodied_stack(n_agents=4):
    cfg = te.EngineConfig(world_size=40.0, proximity_radius=3.0)
    world = te.World(cfg, n_agents)
    scheduler = te.MLFQScheduler(cfg)
    limiter = te.AIMDLimiter(cfg)
    router = av.ActionRouter()
    llm = ScriptedActionLLM()
    media = CapturingMedia()
    dispatcher = te.InferenceDispatcher(
        cfg, scheduler, limiter, te.ZombieReaper(cfg), llm, media,
        te.GraphWriteQueue(cfg, te.MockNeo4jClient()), action_router=router)
    engine = te.TickEngine(cfg, world, scheduler, limiter)
    router.bind(engine, world)
    return cfg, world, scheduler, dispatcher, engine, router, llm, media


async def test_agency_full_path():
    cfg, world, scheduler, disp, engine, router, llm, media = \
        await build_embodied_stack()
    u = make_uplink(world)
    human = world.find(av.HUMAN_AVATAR_ID)
    human.x, human.y = 30.0, 30.0
    agent = world.agents[0]
    agent.x, agent.y, agent.vx, agent.vy = 10.0, 10.0, 0.0, 0.0

    task = te.AgentTask(
        agent_id=agent.id, kind="dialogue", level=te.QueueLevel.INTERACTIVE,
        messages=[{"role": "system", "content": te.SHARED_WORLD_PREFIX},
                  {"role": "user",
                   "content": 'magnus speaks to you directly: "Come here."'}])
    await disp._execute(task)

    # Schema plumbing + no JSON leaked to TTS.
    assert llm.formats == [router.response_format()]
    assert task.messages[0]["content"] == te.EMBODIED_WORLD_PREFIX  # root swap
    (spk_id, spoken), = media.spoken
    assert spk_id == agent.id and spoken == "Of course, let me come closer."
    assert "{" not in spoken

    # Command landed on the board; steering converts it next tick.
    assert router.actions_routed == 1
    assert engine.command_board[agent.id].target_id == "magnus"
    d_before = math.dist((agent.x, agent.y), (human.x, human.y))
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(0.12)                  # a couple of ticks
    speed = math.hypot(agent.vx, agent.vy)
    heading = (agent.vx / speed, agent.vy / speed)
    to_h = math.dist((agent.x, agent.y), (human.x, human.y))
    direction = ((human.x - agent.x) / to_h, (human.y - agent.y) / to_h)
    dot = heading[0] * direction[0] + heading[1] * direction[1]
    assert dot > 0.99 and abs(speed - cfg.steering_speed) < 0.01
    await asyncio.sleep(0.5)
    runner.cancel()
    d_after = math.dist((agent.x, agent.y), (human.x, human.y))
    assert d_after < d_before - 0.5            # measurably closer
    # Teleport the human next to the agent → arrival clears the command.
    human.x, human.y = agent.x + 0.5, agent.y
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(0.1)
    runner.cancel()
    assert agent.id not in engine.command_board
    assert engine.commands_completed == 1
    print(f"PASS agency: schema→say→board→velocity (dot={dot:.3f}), "
          f"closed {d_before - d_after:.1f} units, arrival cleared")


async def test_lock_free_under_fire():
    cfg, world, scheduler, disp, engine, router, llm, media = \
        await build_embodied_stack(n_agents=6)
    make_uplink(world)
    runner = asyncio.create_task(engine.run())
    agent = world.agents[0]
    targets = [a.id for a in world.agents[1:]] + [av.HUMAN_AVATAR_ID]
    for i in range(200):                        # storm the board mid-run
        engine.issue_command(agent.id, te.KinematicCommand(
            tool="move_towards", target_id=targets[i % len(targets)]))
        if i % 20 == 0:
            await asyncio.sleep(0.03)
    last_target = targets[199 % len(targets)]
    await asyncio.sleep(0.3)
    runner.cancel()
    assert engine.max_tick_overrun_ms < 5.0
    cmd = engine.command_board.get(agent.id)
    assert cmd is None or cmd.target_id == last_target   # latest wins (or arrived)
    assert engine.commands_issued == 200
    print(f"PASS lock-free: 200 mid-run replacements, overrun "
          f"{engine.max_tick_overrun_ms:.1f}ms, latest command won")


async def test_rejection_and_fallback():
    cfg, world, scheduler, disp, engine, router, llm, media = \
        await build_embodied_stack()
    make_uplink(world)
    aid = world.agents[0].id
    router._route(aid, {"tool": "move_towards", "target_id": "ghost_999"})
    router._route(aid, {"tool": "move_towards", "target_id": aid})
    router._route(aid, {"tool": "teleport", "target_id": "magnus"})
    assert router.actions_rejected == 3 and not engine.command_board

    say = router.intercept(aid, 'the model rambled with no json at all')
    assert say == "the model rambled with no json at all"
    say2 = router.intercept(aid, '{"say": "Hello", "action": {broken')
    assert router.parse_fallbacks == 2 and say2.startswith("{")
    assert router.actions_routed == 0
    print("PASS rejection: ghost/self/bad-tool refused; malformed JSON "
          "degrades to speech")


async def main():
    await test_uplink()
    await test_perception_asymmetry()
    await test_voice_nearest_k()
    await test_agency_full_path()
    await test_lock_free_under_fire()
    await test_rejection_and_fallback()
    print("\nALL PHASE 7 TESTS PASSED")


asyncio.run(main())
