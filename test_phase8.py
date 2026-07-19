"""Phase 8 verification suite (headless, real engine/dispatcher/scheduler).

1. CONTESTED   — A and B pick_up the same key on the same tick: the nearer
                 wins; an EXACT distance tie falls to lexicographic agent_id;
                 the loser is a counted contested_loss; identical inputs →
                 identical winner across 20 replays (determinism), and the
                 resolution pass never bends the 20 Hz tick.
2. INVENTORY   — pick_up sets owner; the carried key is position-slaved to
                 its wandering owner; give transfers within reach and is
                 refused beyond it; you cannot give what you don't hold.
3. USE RULES   — key on door flips locked, bumps version; use requires reach.
4. DELTA PLANE — agent frame stays exactly 6+25n bytes; only dirty entities
                 ship; records round-trip bit-perfectly; a 20-entity mass
                 mutation interlaces across passes at ≤8/flush; keyframe
                 re-heals; reliable=True on the entity topic only.
5. GOAL LOOP   — Q3 task carries the goal grammar, runs through the real
                 dispatcher, lands on the goal board; the next dialogue leaf
                 begins with the byte-stable goal segment (shared root
                 untouched); Q0 strictly preempts Q3; garbage JSON leaves the
                 board unchanged.
"""
import asyncio
import logging
import math
import struct
import types

import avatar as av
import entities as en
import goalseek as gs
import tick_engine as te

logging.disable(logging.ERROR)


def base_world(n=4, size=40.0):
    cfg = te.EngineConfig(world_size=size, proximity_radius=3.0)
    return cfg, te.World(cfg, n)


# ------------------------------------------------------------------- tests

async def test_contested_pickup():
    for trial in range(20):                     # determinism across replays
        cfg, world = base_world(3)
        a, b, c = world.agents
        key = world.entities.spawn("key_01", "key", 10.0, 10.0)
        a.x, a.y = 11.0, 10.0                   # d = 1.0  → wins
        b.x, b.y = 10.0, 11.5                   # d = 1.5  → contested loss
        c.x, c.y = 30.0, 30.0                   # far      → rejected (reach)
        engine = te.TickEngine(cfg, world, te.MLFQScheduler(cfg),
                               te.AIMDLimiter(cfg))
        for agent in (b, a, c):                 # submission order shuffled:
            engine.submit_intent(en.EntityIntent(   # must not matter
                agent_id=agent.id, tool="pick_up", entity_id="key_01"))
        runner = asyncio.create_task(engine.run())
        await asyncio.sleep(0.08)
        runner.cancel()
        assert key.owner_id == a.id, (trial, key.owner_id)
        assert engine.intent_stats.resolved == 1
        assert engine.intent_stats.contested_losses == 1
        assert engine.intent_stats.rejected == 1
        assert engine.max_tick_overrun_ms < 5.0

    # EXACT distance tie → lexicographic agent_id (a 3-4-5 mirror pair).
    cfg, world = base_world(2)
    a, b = world.agents                          # agent_000 < agent_001
    key = world.entities.spawn("key_02", "key", 10.0, 10.0)
    a.x, a.y = 13.0, 14.0                        # d = 5.0
    b.x, b.y = 14.0, 13.0                        # d = 5.0 — identical
    engine = te.TickEngine(te.EngineConfig(world_size=40.0,
                                           interaction_radius=6.0),
                           world, te.MLFQScheduler(cfg), te.AIMDLimiter(cfg))
    engine.submit_intent(en.EntityIntent(agent_id=b.id, tool="pick_up",
                                         entity_id="key_02"))
    engine.submit_intent(en.EntityIntent(agent_id=a.id, tool="pick_up",
                                         entity_id="key_02"))
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(0.08)
    runner.cancel()
    assert key.owner_id == a.id                  # lexicographic invariant
    print("PASS contested: nearest wins ×20 replays, exact tie → lexical id, "
          "physics unbent")


async def test_inventory_and_give():
    cfg, world = base_world(3)
    a, b, c = world.agents
    key = world.entities.spawn("key_01", "key", 10.0, 10.0)
    a.x, a.y, a.vx, a.vy = 10.5, 10.0, 1.5, 0.0
    b.x, b.y = 11.5, 10.0                        # within reach of a
    c.x, c.y = 35.0, 35.0
    engine = te.TickEngine(cfg, world, te.MLFQScheduler(cfg),
                           te.AIMDLimiter(cfg))
    engine.submit_intent(en.EntityIntent(agent_id=a.id, tool="pick_up",
                                         entity_id="key_01"))
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(0.3)                     # let a wander while carrying
    runner.cancel()
    assert key.owner_id == a.id
    assert math.dist((key.x, key.y), (a.x, a.y)) < 0.2   # position-slaved

    # give within reach succeeds; giving what you don't own is refused.
    engine.submit_intent(en.EntityIntent(agent_id=a.id, tool="give",
                                         entity_id="key_01", secondary_id=b.id))
    engine.submit_intent(en.EntityIntent(agent_id=c.id, tool="give",
                                         entity_id="key_01", secondary_id=a.id))
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(0.08)
    runner.cancel()
    assert key.owner_id == b.id
    assert engine.intent_stats.rejected >= 1     # c's fraudulent give

    # give beyond arm's reach is refused (b keeps it, far c gets nothing).
    engine.submit_intent(en.EntityIntent(agent_id=b.id, tool="give",
                                         entity_id="key_01", secondary_id=c.id))
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(0.08)
    runner.cancel()
    assert key.owner_id == b.id
    print("PASS inventory: pick_up→carry slaving→give in reach; fraud and "
          "long-distance gives refused")


async def test_use_rules():
    cfg, world = base_world(1)
    agent = world.agents[0]
    door = world.entities.spawn("door_01", "door", 12.0, 10.0,
                                state={"locked": True, "fixed": True})
    key = world.entities.spawn("key_01", "key", 10.0, 10.0)
    agent.x, agent.y, agent.vx, agent.vy = 10.5, 10.0, 0.0, 0.0
    engine = te.TickEngine(cfg, world, te.MLFQScheduler(cfg),
                           te.AIMDLimiter(cfg))
    v0 = door.version
    engine.submit_intent(en.EntityIntent(agent_id=agent.id, tool="pick_up",
                                         entity_id="key_01"))
    engine.submit_intent(en.EntityIntent(agent_id=agent.id, tool="use",
                                         entity_id="key_01",
                                         secondary_id="door_01"))
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(0.15)
    runner.cancel()
    assert key.owner_id == agent.id
    assert door.state["locked"] is False         # key-on-door rule fired
    assert door.state["last_used_by"] == agent.id
    assert door.version > v0
    # 'fixed' door itself cannot be picked up.
    engine.submit_intent(en.EntityIntent(agent_id=agent.id, tool="pick_up",
                                         entity_id="door_01"))
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(0.08)
    runner.cancel()
    assert door.owner_id is None
    print("PASS use: key unlocks door, version bumped; bolted door "
          "un-pickupable")


async def test_delta_plane():
    import broadcaster as bc
    sent: list[tuple[bytes, bool, str]] = []

    class FP:
        async def publish_data(self, payload, *, reliable=True,
                               destination_identities=(), topic=""):
            sent.append((bytes(payload), reliable, topic))

    class FR:
        local_participant = FP()
        async def connect(self, u, t): pass
        async def disconnect(self): pass

    bc.rtc = types.SimpleNamespace(Room=lambda: FR())
    bc.api = types.SimpleNamespace(
        AccessToken=lambda k, s: types.SimpleNamespace(
            with_identity=lambda i: types.SimpleNamespace(
                with_name=lambda n: types.SimpleNamespace(
                    with_grants=lambda g: types.SimpleNamespace(
                        to_jwt=lambda: "jwt")))),
        VideoGrants=lambda **kw: None)

    cfg, world = base_world(50)
    for i in range(20):
        world.entities.spawn(f"crate_{i:02d}", "crate", float(i), 5.0,
                             state={"sealed": True})
    b = bc.StateBroadcaster("wss://x", "k", "s")
    await b.connect()
    b.attach_entities(world.entities)
    b._last_keyframe = 1e18                      # suppress keyframe for now
    sender = asyncio.create_task(b.run_sender())

    # Pass 1: nothing dirty beyond spawn versions → first flush interlaces.
    b.capture(1, world.agents)
    await asyncio.sleep(0.05)
    agent_frames = [p for p, rel, top in sent if top == bc.DATA_TOPIC]
    entity_frames = [(p, rel) for p, rel, top in sent
                     if top == en.ENTITY_TOPIC]
    assert len(agent_frames[0]) == bc.HEADER_SIZE + 50 * bc.AGENT_SIZE == 1256
    assert len(entity_frames) == 1 and entity_frames[0][1] is True  # reliable
    _, recs = en.decode_entity_deltas(entity_frames[0][0])
    assert len(recs) == 8                        # interlace cap honored

    # Passes 2-3 drain the remaining 12; then a quiet pass ships nothing.
    for tick in (2, 3, 4):
        b.capture(tick, world.agents)
        await asyncio.sleep(0.05)
    entity_frames = [(p, rel) for p, rel, top in sent
                     if top == en.ENTITY_TOPIC]
    assert b.entity_deltas_sent == 20
    assert len(entity_frames) == 3               # 8+8+4; pass 4 sent none

    # A single mutation ships exactly one record, bit-perfectly.
    crate = world.entities.find("crate_07")
    world.entities.mutate(crate, state_update={"sealed": False},
                          owner_id="agent_003")
    b.capture(5, world.agents)
    await asyncio.sleep(0.05)
    payload, _rel = [(p, r) for p, r, t in sent if t == en.ENTITY_TOPIC][-1]
    tick, recs = en.decode_entity_deltas(payload)
    (rec,) = recs
    # Wire ids are the shared compacted forms — same namespace as the
    # agent plane, collision-checked by the broadcaster's registry.
    assert rec == {"id": "crate07", "kind": "crate", "owner_id": "agent003",
                   "x": 7.0, "y": 5.0, "version": crate.version,
                   "state": {"sealed": False}}

    # Keyframe: everything re-marked dirty and re-healed.
    b._last_keyframe = 0.0
    before = b.entity_deltas_sent
    for tick in range(6, 10):
        b.capture(tick, world.agents)
        await asyncio.sleep(0.05)
    assert b.entity_keyframes == 1
    assert b.entity_deltas_sent - before == 20
    sender.cancel()
    print("PASS delta plane: agent frame frozen at 1256B, dirty-only + "
          "8/flush interlace, bit-perfect roundtrip, keyframe heal, reliable")


class GoalLLM:
    """Q3-aware scripted SGLang: goal formulations get a goal JSON; dialogue
    gets plain structured chatter. Records formats + leaves for assertions."""

    def __init__(self, goal_json):
        self.goal_json = goal_json
        self.requests = []

    async def stream_chat(self, model, messages, response_format=None):
        self.requests.append((messages, response_format))
        if messages[0]["content"] == gs.GOALSEEK_PREFIX:
            out = self.goal_json
        else:
            out = '{"say": "As you wish.", "action": null}'
        for i in range(0, len(out), 32):
            yield out[i:i + 32]
            await asyncio.sleep(0.001)


async def build_goal_stack(goal_json):
    cfg, world = base_world(3)
    world.entities.spawn("ledger_01", "ledger", 12.0, 12.0)
    scheduler = te.MLFQScheduler(cfg)
    limiter = te.AIMDLimiter(cfg)
    router = av.ActionRouter()
    llm = GoalLLM(goal_json)
    disp = te.InferenceDispatcher(cfg, scheduler, limiter, te.ZombieReaper(cfg),
                                  llm, te.LiveKitPublisher(),
                                  te.GraphWriteQueue(cfg, te.MockNeo4jClient()),
                                  action_router=router)
    engine = te.TickEngine(cfg, world, scheduler, limiter)
    router.bind(engine, world)

    class FakeGraph:
        async def fetch_beliefs(self, agent_id, limit):
            return [{"text": "Grain deals with agent_002 are unsafe.",
                     "importance": 0.9}]

    daemon = gs.GoalSeekDaemon(FakeGraph(), scheduler, world, disp.goal_board,
                               action_router=router)
    worker = asyncio.create_task(disp.run_worker())
    return cfg, world, scheduler, disp, daemon, llm, worker


async def test_goal_loop():
    goal = "Obtain ledger_01 before agent_002 reaches it."
    cfg, world, scheduler, disp, daemon, llm, worker = await build_goal_stack(
        '{"goal": "' + goal + '"}')
    agent = world.agents[0]
    agent.x, agent.y = 10.0, 10.0        # deterministic: ledger in radius

    changed = await daemon.formulate(agent)
    await asyncio.sleep(0.1)
    assert changed and disp.goal_board[agent.id] == goal

    # The Q3 request carried the goal grammar and the stable Q3 root.
    goal_reqs = [(m, f) for m, f in llm.requests
                 if m[0]["content"] == gs.GOALSEEK_PREFIX]
    (messages, fmt), = goal_reqs
    assert fmt["json_schema"]["schema"] == gs.GOAL_JSON_SCHEMA
    assert "Grain deals with agent_002" in messages[-1]["content"]
    assert "ledger_01 (ledger" in messages[-1]["content"]

    # The next dialogue generation is biased: goal segment leads the leaf,
    # the shared root is byte-identical EMBODIED prefix — never goal-touched.
    task = te.AgentTask(
        agent_id=agent.id, kind="dialogue", level=te.QueueLevel.INTERACTIVE,
        messages=[{"role": "system", "content": te.SHARED_WORLD_PREFIX},
                  {"role": "user", "content": "agent_001 crossed your path. React."}])
    await disp._execute(task)
    dialogue = [(m, f) for m, f in llm.requests
                if m[0]["content"] == te.EMBODIED_WORLD_PREFIX]
    (dmsgs, _), = dialogue
    assert dmsgs[-1]["content"].startswith(
        f"[Your standing personal goal: {goal}]")
    assert goal not in dmsgs[0]["content"]       # root untouched, branch only

    # Identical re-formulation: board unchanged (branch stays warm).
    assert await daemon.formulate(agent) is False
    # Q0 strictly preempts Q3: enqueue one of each; Q0 dequeues first.
    await scheduler.submit(te.AgentTask(agent_id="x", kind="goal_formulation",
                                        level=te.QueueLevel.GOAL, messages=[]))
    await scheduler.submit(te.AgentTask(agent_id="y", kind="dialogue",
                                        level=te.QueueLevel.INTERACTIVE,
                                        messages=[]))
    first = await scheduler.next_task()
    assert first.level == te.QueueLevel.INTERACTIVE
    worker.cancel()
    print("PASS goal loop: Q3 grammar + beliefs in leaf, board set, dialogue "
          "leaf leads with goal, root clean, Q0 preempts Q3")


async def test_goal_fail_open():
    cfg, world, scheduler, disp, daemon, llm, worker = await build_goal_stack(
        'certainly! the goal should be {broken json')
    agent = world.agents[0]
    assert await daemon.formulate(agent) is False
    assert daemon.parse_failures == 1
    assert agent.id not in disp.goal_board       # unchanged: fail-open
    # {"goal": null} clears an existing goal.
    disp.goal_board[agent.id] = "old purpose"
    daemon2_llm_json = '{"goal": null}'
    llm.goal_json = daemon2_llm_json
    assert await daemon.formulate(agent) is True
    assert agent.id not in disp.goal_board and daemon.goals_cleared == 1
    worker.cancel()
    print("PASS goal fail-open: garbage keeps old state; null clears cleanly")


async def main():
    await test_contested_pickup()
    await test_inventory_and_give()
    await test_use_rules()
    await test_delta_plane()
    await test_goal_loop()
    await test_goal_fail_open()
    print("\nALL PHASE 8 TESTS PASSED")


asyncio.run(main())
