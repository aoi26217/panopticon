"""Phase 10 verification suite (zero infrastructure, real engine throughout).

1. RESURRECTION — a running world (moved agents, human avatar, carried key,
                  mutated door, live command, standing goals) is snapshotted,
                  the process 'dies', a FRESH world restores: every float
                  exact, entity versions preserved, goals identical, the
                  command re-issued — then physics continues on tick one.
2. ATOMICITY    — flush leaves no .tmp; a corrupt file boots fresh
                  (fail-open) leaving the world untouched.
3. DECOUPLING   — a 200 ms-slow disk cannot bend the tick: snapshots
                  coalesce latest-wins, overrun stays ~0.
4. ROTATION     — the [2] conditions segment appends AFTER a byte-identical
                  [1] root; two prompts in one epoch are byte-identical;
                  a saturated limiter DEFERS rotation (congestion gate).
5. ECONOMY      — low contention + a cluster → relic at the centroid; high
                  contention → silence; the relic cap holds.
6. COOPERATION  — a heavy_vault (quorum 2): two same-tick signers open it;
                  a staggered pair within the TTL opens it (the buffer);
                  a lone signer's signature EXPIRES and the vault stays
                  locked; an out-of-reach second signer never counts.
7. HEARTBEAT    — snapshotter + director + cosigns all live: overrun ~0.
"""
import asyncio
import json
import logging
import math
import time
from pathlib import Path

import avatar as av
import director as dr
import entities as en
import persistence as ps
import tick_engine as te

logging.disable(logging.ERROR)
STATE = Path("/tmp/panopticon_state.json")


def build(n=4, size=40.0):
    cfg = te.EngineConfig(world_size=size, proximity_radius=3.0)
    world = te.World(cfg, n)
    scheduler = te.MLFQScheduler(cfg)
    limiter = te.AIMDLimiter(cfg)
    engine = te.TickEngine(cfg, world, scheduler, limiter)
    return cfg, world, scheduler, limiter, engine


# ------------------------------------------------------------------- tests

async def test_resurrection():
    STATE.unlink(missing_ok=True)
    cfg, world, sched, lim, engine = build()
    u = av.AvatarUplink("wss://x", "k", "s"); u.bind(world)
    human = world.find(av.HUMAN_AVATAR_ID)
    human.x, human.y = 33.0, 7.5

    key = world.entities.spawn("key_01", "key", 10.0, 10.0)
    door = world.entities.spawn("door_01", "door", 20.0, 20.0,
                                state={"locked": True, "fixed": True})
    world.entities.mutate(key, owner_id="agent_000")           # carried
    world.entities.mutate(door, state_update={"locked": False,
                                              "last_used_by": "agent_000"})
    goal_board = {"agent_000": "Deliver key_01 to magnus.",
                  "agent_002": "Guard door_01."}
    snap = ps.Snapshotter(STATE, every_ticks=5, goal_board=goal_board)
    snap.bind(engine)
    engine.set_tick_hook(snap.make_tick_hook())
    engine.issue_command("agent_001", te.KinematicCommand(
        tool="move_towards", target_id="agent_000", ttl_s=30.0))
    for a in world.agents:
        a.last_reasoned_at = time.monotonic() + 999

    runner = asyncio.create_task(engine.run())
    flusher = asyncio.create_task(snap.run_flusher())
    await asyncio.sleep(0.6)
    runner.cancel(); flusher.cancel()
    assert snap.snapshots_flushed >= 1
    assert not STATE.with_suffix(".json.tmp").exists()          # atomic
    # The contract: resurrection matches the LAST FLUSHED SNAPSHOT bit-for-
    # bit; the loss window is bounded at <= every_ticks (5 ticks = 250 ms
    # here). The world moved past the capture before death — that delta is
    # the accepted price of asynchronous durability, so the fingerprint is
    # the file, not the corpse.
    on_disk = json.loads(STATE.read_text())
    assert engine.tick_count - on_disk["tick"] <= 5             # bounded loss
    died = {r["id"]: (r["x"], r["y"], r["vx"], r["vy"])
            for r in on_disk["agents"]}
    died_versions = {r["id"]: r["version"] for r in on_disk["entities"]}

    # ---- the process is dead; a fresh boot begins --------------------------
    cfg2, world2, sched2, lim2, engine2 = build(n=2)   # wrong --agents count:
    goal_board2: dict[str, str] = {}                   # snapshot must win
    tick = ps.restore_world(STATE, world2, engine2, goal_board2)
    assert tick is not None and tick >= 5

    assert {a.id for a in world2.agents} == set(died)           # roster exact
    for a in world2.agents:
        assert (a.x, a.y, a.vx, a.vy) == died[a.id]             # floats exact
    h2 = world2.find(av.HUMAN_AVATAR_ID)
    assert h2 is not None and h2.is_human
    key2, door2 = world2.entities.find("key_01"), world2.entities.find("door_01")
    assert key2.owner_id == "agent_000"
    assert door2.state == {"locked": False, "fixed": True,
                           "last_used_by": "agent_000"}
    assert {e.id: e.version for e in world2.entities.all()} == died_versions
    assert goal_board2 == goal_board
    assert engine2.command_board["agent_001"].target_id == "agent_000"

    # ...and life continues on the very next tick.
    for a in world2.agents:
        a.last_reasoned_at = time.monotonic() + 999
    before = {a.id: (a.x, a.y) for a in world2.agents if not a.is_human
              and (abs(a.vx) > 0.01 or abs(a.vy) > 0.01)}
    runner2 = asyncio.create_task(engine2.run())
    await asyncio.sleep(0.12)
    runner2.cancel()
    moved = sum(1 for aid, (x, y) in before.items()
                if math.dist((x, y), (world2.find(aid).x,
                                      world2.find(aid).y)) > 0.01)
    assert moved == len(before) and engine2.tick_count >= 1
    # Carried key still slaved to its resurrected owner.
    owner = world2.find("agent_000")
    assert math.dist((key2.x, key2.y), (owner.x, owner.y)) < 0.2
    print(f"PASS resurrection: tick {tick} restored exactly "
          f"({len(died)} agents, {len(died_versions)} entities, "
          f"{len(goal_board)} goals, 1 command) and physics resumed")


async def test_atomicity_and_fail_open():
    STATE.write_text('{"version": 1, "tick": 40, "agents": [BROKEN')
    cfg, world, sched, lim, engine = build(n=3)
    fingerprint = [(a.id, a.x, a.y) for a in world.agents]
    assert ps.restore_world(STATE, world, engine, {}) is None
    assert [(a.id, a.x, a.y) for a in world.agents] == fingerprint
    assert ps.restore_world("/tmp/does_not_exist_anywhere.json",
                            world, engine, {}) is None
    print("PASS fail-open: corrupt and missing snapshots boot fresh, "
          "world untouched")


async def test_slow_disk_decoupled():
    STATE.unlink(missing_ok=True)
    cfg, world, sched, lim, engine = build(n=20)
    snap = ps.Snapshotter(STATE, every_ticks=2)
    snap.bind(engine)
    real_write = snap._write
    def slow_write(s):
        time.sleep(0.2)                    # 4× the tick period, in-executor
        real_write(s)
    snap._write = slow_write
    engine.set_tick_hook(snap.make_tick_hook())
    for a in world.agents:
        a.last_reasoned_at = time.monotonic() + 999
    runner = asyncio.create_task(engine.run())
    flusher = asyncio.create_task(snap.run_flusher())
    await asyncio.sleep(2.0)
    runner.cancel(); flusher.cancel()
    assert engine.max_tick_overrun_ms < 5.0
    assert snap.snapshots_captured >= 15
    assert snap.snapshots_flushed <= 10            # disk-paced
    assert snap.snapshots_superseded >= 5          # latest-wins coalescing
    print(f"PASS decoupling: 200ms disk, {snap.snapshots_captured} captured "
          f"→ {snap.snapshots_flushed} flushed ({snap.snapshots_superseded} "
          f"superseded), overrun {engine.max_tick_overrun_ms:.1f}ms")


class EpochLLM:
    def __init__(self):
        self.systems = []
    async def stream_chat(self, model, messages, response_format=None):
        self.systems.append(messages[0]["content"])
        yield '{"say": "Noted.", "action": null}'


async def test_prefix_rotation():
    cfg, world, sched, lim, engine = build()
    router = av.ActionRouter()
    llm = EpochLLM()
    disp = te.InferenceDispatcher(cfg, sched, lim, te.ZombieReaper(cfg), llm,
                                  te.LiveKitPublisher(),
                                  te.GraphWriteQueue(cfg, te.MockNeo4jClient()),
                                  action_router=router)
    router.bind(engine, world)
    d = dr.Director(dr.DirectorConfig(min_rotation_interval_s=0.0))
    d.bind(engine, world, lim, disp)

    def dialogue():
        return te.AgentTask(agent_id="agent_000", kind="dialogue",
                            level=te.QueueLevel.INTERACTIVE,
                            messages=[{"role": "system",
                                       "content": te.SHARED_WORLD_PREFIX},
                                      {"role": "user", "content": "React."}])

    await disp._execute(dialogue())                    # epoch 0: no segment
    d.rotate_conditions("steady rain; awnings drum")
    await disp._execute(dialogue())                    # epoch 1
    await disp._execute(dialogue())                    # epoch 1 again
    e0, e1a, e1b = llm.systems
    assert e0 == te.EMBODIED_WORLD_PREFIX              # pre-rotation: pristine
    assert e1a.startswith(te.EMBODIED_WORLD_PREFIX)    # [1] root byte-intact
    assert e1a.endswith("Current world conditions: steady rain; awnings drum")
    assert e1a == e1b                                  # epoch-stable: cached
    # Congestion gate: saturate the limiter → the cycle defers rotation.
    lim._active = lim.limit
    before = d.rotations
    d._lever_environment()
    assert d.rotations == before and d.rotations_deferred == 1
    lim._active = 0
    print("PASS rotation: root byte-identical, segment [2] appended, "
          "epoch-stable, congestion-gated")


async def test_economy_lever():
    cfg, world, sched, lim, engine = build(n=12, size=60.0)
    for i, a in enumerate(world.agents[:8]):           # a tight plaza huddle
        a.x, a.y = 9.0 + (i % 3) * 0.8, 9.0 + (i // 3) * 0.8
    disp = object()
    d = dr.Director(dr.DirectorConfig(relic_cap=2))
    d.bind(engine, world, lim, disp)

    d._lever_economy()                                 # calm plaza → relic
    relics = [e for e in world.entities.all() if e.kind == "relic"]
    assert len(relics) == 1
    r = relics[0]
    assert math.dist((r.x, r.y), (9.8, 9.8)) < 2.0     # at the huddle
    # High contention → the thermostat goes quiet.
    engine.intent_stats.contested_losses += 10
    d._lever_economy()
    assert sum(1 for e in world.entities.all() if e.kind == "relic") == 1
    # Calm again → second relic; then the cap holds forever.
    d._lever_economy(); d._lever_economy(); d._lever_economy()
    assert sum(1 for e in world.entities.all() if e.kind == "relic") == 2
    print("PASS economy: relic at the cluster centroid, thermostat "
          "silent under contention, cap held at 2")


def vault_stack(n=3):
    cfg, world, sched, lim, engine = build(n=n)
    vault = world.entities.spawn("vault_01", "heavy_vault", 20.0, 20.0,
                                 state={"locked": True, "fixed": True,
                                        "cosigners_required": 2})
    for a in world.agents:
        a.vx = a.vy = 0.0
        a.last_reasoned_at = time.monotonic() + 999
    return cfg, world, engine, vault


def cosign(engine, agent_id):
    engine.submit_intent(en.EntityIntent(
        agent_id=agent_id, tool="use",
        entity_id="vault_01", secondary_id="vault_01"))


async def test_cooperative_vault():
    # (a) two signers, same tick → open, credited to both.
    cfg, world, engine, vault = vault_stack()
    a, b, c = world.agents
    a.x, a.y = 20.8, 20.0
    b.x, b.y = 20.0, 20.9
    c.x, c.y = 35.0, 35.0
    cosign(engine, a.id); cosign(engine, b.id)
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(0.1)
    runner.cancel()
    assert vault.state["locked"] is False
    assert vault.state["opened_by"] == sorted([a.id, b.id])
    assert world.entities.cosigns_completed == 1
    assert engine.intent_stats.resolved == 1           # ONE action, two hands

    # (b) a lone signer: the signature expires, the vault holds.
    cfg, world, engine, vault = vault_stack()
    world.agents[0].x, world.agents[0].y = 20.8, 20.0
    cosign(engine, world.agents[0].id)
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(0.2)
    assert vault.state["locked"] is True               # pending, not open
    assert "vault_01" in world.entities.pending_cosigns
    await asyncio.sleep(world.entities.cosign_ttl_s + 0.3)
    runner.cancel()
    assert vault.state["locked"] is True
    assert world.entities.cosigns_expired >= 1
    assert "vault_01" not in world.entities.pending_cosigns
    print("PASS vault(a,b): duo opened it in one resolved action; the loner's "
          "signature lapsed to silence")

    # (c) staggered within the TTL: the buffer is the whole point.
    cfg, world, engine, vault = vault_stack()
    a, b, _ = world.agents
    a.x, a.y = 20.8, 20.0
    b.x, b.y = 20.0, 20.9
    runner = asyncio.create_task(engine.run())
    cosign(engine, a.id)
    await asyncio.sleep(0.4)                           # 8 ticks apart
    assert vault.state["locked"] is True
    cosign(engine, b.id)
    await asyncio.sleep(0.1)
    runner.cancel()
    assert vault.state["locked"] is False
    assert vault.state["opened_by"] == sorted([a.id, b.id])

    # (d) an out-of-reach 'second' signer never counts.
    cfg, world, engine, vault = vault_stack()
    a, b, _ = world.agents
    a.x, a.y = 20.8, 20.0
    b.x, b.y = 35.0, 35.0                              # yelling from afar
    cosign(engine, a.id); cosign(engine, b.id)
    runner = asyncio.create_task(engine.run())
    await asyncio.sleep(0.1)
    runner.cancel()
    assert vault.state["locked"] is True
    assert len(world.entities.pending_cosigns.get("vault_01", {})) == 1
    assert engine.intent_stats.rejected >= 1           # b: out of reach
    print("PASS vault(c,d): staggered cosign within TTL opened it; a distant "
          "signature was refused, quorum unmet")


async def test_heartbeat_full_stack():
    STATE.unlink(missing_ok=True)
    cfg, world, engine, vault = vault_stack(n=20)
    snap = ps.Snapshotter(STATE, every_ticks=4)
    snap.bind(engine)
    engine.set_tick_hook(snap.make_tick_hook())
    d = dr.Director(dr.DirectorConfig(min_rotation_interval_s=0.0))
    class MiniDisp: world_conditions = None
    d.bind(engine, world, te.AIMDLimiter(cfg), MiniDisp())
    runner = asyncio.create_task(engine.run())
    flusher = asyncio.create_task(snap.run_flusher())
    for i in range(10):
        cosign(engine, world.agents[i % 20].id)
        d.cycle()
        await asyncio.sleep(0.2)
    runner.cancel(); flusher.cancel()
    assert engine.max_tick_overrun_ms < 5.0
    assert engine.tick_count >= 35
    print(f"PASS heartbeat: snapshots + director + cosigns concurrent, "
          f"{engine.tick_count} ticks, overrun "
          f"{engine.max_tick_overrun_ms:.1f}ms")


async def main():
    await test_resurrection()
    await test_atomicity_and_fail_open()
    await test_slow_disk_decoupled()
    await test_prefix_rotation()
    await test_economy_lever()
    await test_cooperative_vault()
    await test_heartbeat_full_stack()
    print("\nALL PHASE 10 TESTS PASSED")


asyncio.run(main())
