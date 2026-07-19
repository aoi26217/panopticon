# Project Panopticon

**A persistent, embodied, narratively-steered 50-agent simulation — built on one
promise: the heartbeat never bends.**

Fifty LLM-driven agents inhabit a shared plaza at a locked 20 Hz physics tick.
They perceive each other, speak with real-time voices, remember through a hybrid
graph-vector memory, consolidate experience into beliefs, form goals, walk real
paths around real walls, pick things up, trade them, cooperate on things too
heavy for one — and survive the death of their own process. A human ("Magnus")
walks among them, embodied through an authoritative position uplink and a live
voice. An omniscient Director watches the telemetry and quietly adjusts the
world when the plaza drifts out of tension.

Eleven phases. Nine test suites. One executable acceptance audit. Zero deadlocks
by construction. This document is the reasoning behind all of it — handed to
you, the maintainer, as the definitive source of truth.

```
                    ┌─────────────────────────────────────────────┐
                    │           TICK ENGINE  (20 Hz, sacred)       │
                    │  intents → steering → physics → collisions   │
                    │  → carry → cosign-expiry → hooks (capture)   │
                    └───────┬─────────────────────────────▲───────┘
        lock-free boards ───┤                             ├─── lock-free boards
   (command / intent / goal)│                             │(avatar position)
┌───────────▼──────────┐ ┌──▼───────────────┐ ┌───────────┴──────────┐
│ MLFQ Q0..Q3 + AIMD   │ │  Aux daemons      │ │  The Glasshouse      │
│ InferenceDispatcher  │ │  consolidation    │ │  broadcaster (20Hz)  │
│ → SGLang (RadixAttn) │ │  goalseek (Q3)    │ │  entity delta plane  │
│ → TTS → LiveKit      │ │  director, snap-  │ │  avatar uplink, STT  │
│ → GraphWriteQueue →  │ │  shotter, oracle  │ │  → UE5 (Hermite)     │
│   Neo4j (1 writer)   │ │  (all fail-open)  │ │                      │
│ → Qdrant (vec-first) │ └───────────────────┘ └──────────────────────┘
└──────────────────────┘
```

| Layer | Files |
|---|---|
| Core engine (tick, MLFQ, AIMD, dispatcher, single-writer queue) | `tick_engine.py` |
| Production clients (SGLang SSE, Neo4j) | `clients.py` |
| Voice out (sentence chunking, TTS, LiveKit tracks) | `media.py` |
| Memory (Qdrant + Neo4j hybrid retrieval, dual-write) | `memory.py` |
| Consolidation daemon (Q2: clump → belief → tombstone) | `consolidation.py` |
| State broadcast + entity delta plane | `broadcaster.py`, `entities.py` |
| Reality injection (news/market events) | `oracle.py` |
| Embodiment (uplink, STT ingress, action router) | `avatar.py` |
| Economy & cooperation (entities, intents, cosigns) | `entities.py` |
| Intrinsic motivation (Q3 goal daemon) | `goalseek.py` |
| Geometry (grid, A*, spatial hash, collisions) | `navigation.py` |
| Persistence (snapshot / resurrect) | `persistence.py` |
| The Narrative Director (two levers, zero LLM) | `director.py` |
| Observability (Prometheus custom collector) | `metrics.py` |
| Stress harness (the falsifiable AIMD proof) | `crucible.py` |
| Executable acceptance checklist | `phase11_audit.py` |
| Pre-flight GPU profiler | `profile_silicon.py` |
| UE5 client frame | `glasshouse/MagnusObserver.{h,cpp}` |
| Deployment (compose, bootstrap, dashboards) | `deploy/` |
| Verification suites, phase by phase | `test_phase1.py` … `test_phase10.py` |

---

## 1. The Panopticon Philosophy

### Fail-open, or: the simulation may not die of its own features

Every subsystem in this engine is an **upgrade, never a dependency**. Memory
retrieval that blows its 100 ms budget returns an empty string. A dead TTS
service mutes agents; it does not stop them thinking. A saturated path planner
leaves the straight-line steering that was already running. A crashed
consolidation daemon leaves raw episodic memory intact. The Oracle, the
Director, the snapshotter, the broadcaster — every one is wrapped in
`_supervise()`: it may die, and the plaza keeps living.

This is not defensive pessimism. It is an architectural admission that a
50-agent LLM system contains dozens of independently-failing components, and
the product — a continuous, watchable, inhabitable world — cannot afford to be
the conjunction of their uptimes. Fail-open converts a product of probabilities
into a minimum.

**The one blind spot of fail-open** (learned the hard way in the Phase 11
audit): *absence looks like health*. A deployment file that silently omits four
subsystems produces a plaza that runs perfectly and does nothing interesting.
This is why the audit validates the runbook, not just the code — see §4.

### The heartbeat is decoupled because latency is not simulation time

The physics tick and the inference layer live in different universes. A tick is
deterministic, synchronous, and cheap: integrate velocities, resolve intents,
push circles apart — microseconds. An LLM generation is non-deterministic I/O
with a latency distribution whose tail you do not control. The single most
important decision in this codebase is that **the tick loop never awaits
anything**: it detects triggers and performs lock-free enqueues; every await in
the system lives in worker coroutines on the other side of a queue.

The consequence is that GPU pressure expresses itself as *degradation*
(agents falling back to state-machine heuristics, dialogue queuing, plans
falling back to straight lines) and never as *time dilation*. The world's clock
is not a function of the world's intelligence.

### Why "the heartbeat never bends" is the brand

Because it is the only claim in the system that is **binary and continuously
measured**. Everything else degrades gracefully by design — which means
everything else is allowed to be partially broken at any moment. The tick is
the one thing that is not. `panopticon_tick_max_overrun_ms` sitting at 0.0 is
the engine's signature; through an 11-phase history it has survived a
100-task/s storm, a 200 ms disk, 117 flooded planners, and a fully-lit
concurrent stack without moving. When that number moves, something has violated
the architecture — not the load, the *architecture* (see the diagnostic ladder,
§4). A show whose world stutters is a tech demo. A world whose clock is
unconditional is a place.

---

## 2. Design Invariants & Architectural Anchors

### 2.1 The Single-Writer Guarantee

**Every Neo4j mutation in the entire system flows through one queue consumed by
one coroutine** (`GraphWriteQueue` → `run_writer`). Producers — the dispatcher,
the consolidation daemon, memory dual-writes — submit plain dicts and move on.

Why this kills deadlocks *by construction* rather than by care: a graph
deadlock requires two transactions holding locks while waiting on each other's.
With exactly one writer there is never a second transaction to wait on. The
theorem needs no lock ordering discipline, no retry-on-deadlock logic, no
reasoning about which daemon touches which node labels — properties that decay
under maintenance. Instead: mutations serialize into batched `UNWIND`
statements grouped by type, retried boundedly, and dead-lettered to replayable
JSONL when the graph is truly down.

The audit enforces this structurally: `phase11_audit.py` scans every sub-daemon
for write primitives (`execute_write`, driver sessions) and proves at runtime
that consolidation emits only dicts. **If you add a component that writes to the
graph any other way, the audit fails.** That is intentional. Reads are
unrestricted — `fetch_memories`, `fetch_beliefs` run concurrent read
transactions freely; the guarantee governs mutation only.

*Scar that shaped it:* an early retry configuration let a wedged transaction
hold the writer for 126 seconds. The fix (driver-level
`max_transaction_retry_time` capped at 3 s + dead-letter) is why the writer can
be slow but can never be *stuck*.

### 2.2 The RadixAttention Cache Contract

SGLang's RadixAttention shares KV cache across requests up to the first
divergent token. With 50 agents at ~2 Hz reasoning cadence, prefix reuse is the
difference between a warm ~60 ms TTFT and re-prefilling a few hundred tokens
per request until the budget dies. The prompt is therefore a **strict stability
hierarchy** — most stable bytes first:

| Segment | Content | Stability | Cache consequence when it changes |
|---|---|---|---|
| **[1]** | `SHARED_WORLD_PREFIX` / `EMBODIED_WORLD_PREFIX` (system) | Frozen constants | Full tree rebuild. Never happens at runtime. |
| **[2]** | `world_conditions` tail (system, Director-rotated) | Epochs ≥ 120 s | ~12-token re-prefill per branch; root untouched |
| **[3]** | `[Your standing personal goal: …]` at the **head of the user leaf** | Per-agent, changes rarely | One agent's branch re-prefills |
| **[4]** | Event text + retrieved memories (rest of the user leaf) | Every request | Leaf-only prefill — the intended cost |

Two rules carry all the weight:

1. **Segment [1] is byte-identical across all 50 agents, always.** One agent
   name, one coordinate, one goal string in the system prefix forks the shared
   root fifty ways and the cache becomes fifty cold caches. This is why the
   Phase 8 spec's request to inject goals "into the prefix" was overruled:
   goals live at [3], the per-agent branch, where a change invalidates exactly
   one branch. The audit probes a live prompt and fails on any leakage.
2. **Two prefixes exist (`SHARED` for observation-only, `EMBODIED` when agency
   is on) but never mix in one deployment.** Two roots means two half-warm
   trees.

The Director's weather rotation (§`director.py`) is the worked example of the
contract: rotating [2] is an atomic string swap, gated on limiter congestion,
with a minimum epoch lifetime — so the (tiny) re-prefill wave amortizes over
hundreds of warm requests and never manufactures an AIMD backoff.

### 2.3 The AIMD/TTFT Safety Circuit

A semaphore caps concurrency at a number *you guessed*. The AIMD limiter caps
it at a number *the server is currently telling you*, via the same
additive-increase / multiplicative-decrease dynamics that keep TCP from
melting the internet:

- Every generation's **TTFT is the congestion signal.** Under budget (500 ms):
  slow additive growth of the ceiling `W`. Over budget, or an error, or a
  cache-pressure signal: `W ×= 0.75`.
- **Admission is a Condition predicate**, not a Semaphore — because the ceiling
  is dynamic and Semaphores can't shrink. The exact invariant, stated
  precisely because the crucible falsified the sloppy version:
  *(i)* every admission satisfies `A ≤ ⌈W⌉` at that instant — measured by an
  exact counter **inside `acquire()`** (`panopticon_aimd_admission_violations`,
  hard 0 forever); *(ii)* `A > W` is legal only while **draining** after a
  multiplicative decrease — in-flight work is never evicted, and while over the
  ceiling `acquire()` blocks, so `A` can only fall.
- When the limiter is saturated at trigger time, agents set `degraded = True`
  and run heuristics instead of queuing unboundedly: backpressure reaches all
  the way to behavior.

The proof is `crucible.py --simulate`: a synthetic backend with convex latency
(budget knee at A≈33) and a hard OOM line at A>48, stormed with 100 tasks/s.
AIMD equilibrates in the low 30s; **peak observed concurrency 34, OOM events
0** — with nothing but the feedback loop standing between the storm and the
death line (the static ceiling sits at 64, deliberately above it).

Before trusting any of this on new hardware, run `profile_silicon.py`
(§4): the real knee moves with quantization, TP topology, and KV fraction, and
the profiler prints the correct `aimd_max_limit` clamp before ignition.

---

## 3. The "Glasshouse" Reality Contract

The UE5 observation deck receives the world over **two planes with opposite
delivery semantics**, and the separation is load-bearing:

| | Spatial Plane | Entity Delta Plane |
|---|---|---|
| Topic | `panopticon.state` | `panopticon.entities` |
| Transport | **Unreliable**, 20 Hz | **Reliable**, event-driven |
| Payload | Packed binary, 6 B header + 25 B/agent (50 agents = **1256 B/frame**, frozen) | Versioned records: id, kind, owner, position, JSON state |
| Loss semantics | A lost frame is healed 50 ms later | A lost "door unlocked" is a **permanent desync** |
| Flow control | Latest-wins 1-deep ring; stale frames superseded, never queued | Dirty-by-version, ≤ 8 records/flush interlace, 10 s keyframe for late joiners |

**Why the split is critical for interpolation:** the UE5 client renders at
60–120 fps against a 20 Hz feed by evaluating a cubic **Hermite spline**
between the two latest snapshots — and the transmitted velocities `vx, vy` are
not decoration, they are the *free tangents* of that spline. This only works
if position frames are (a) frequent, (b) fresh, and (c) allowed to be lost.
Put positions on a reliable channel and a single retransmit stall freezes
every pawn while the transport replays history nobody wants; unreliable +
latest-wins + a 1.5×-clamped extrapolation window means a dropped frame costs
one brief velocity-tangent coast, invisibly healed. Entity *state*, by
contrast, is event-sourced: order and delivery matter absolutely, frequency
does not — the reliable plane's retransmits are cheap precisely because the
plane is quiet.

Both planes share **one 8-byte wire-id namespace** (`compact_wire_id`, collision-
checked at startup by the broadcaster's registry). *Scar:* the entity encoder
originally used naive truncation and `agent_003` silently became `agent_00` —
the exact bug the agent plane had already solved in Phase 6. One compaction
function now serves both planes; `AgentIdOverflowError` fires at startup, never
at frame 40,000.

The human avatar closes the loop upstream: a 12-byte seq-monotonic position
packet on `panopticon.avatar` writes the wire-authoritative `magnus` entity
(physics never integrates it; the collision pass treats it as immovable), and
voice flows mic → LiveKit → streaming STT → Q0 tasks for the nearest three
agents in hearing range.

---

## 4. How to Inhabit the Plaza — The Operator's Runbook

### Cold start on fresh silicon

```bash
cd deploy
./bootstrap.sh                       # deps, NVIDIA hook, model cache, .env
docker compose -f docker-compose-runpod.yml --profile obs up -d \
    sglang embeddings f5-tts qdrant neo4j prometheus grafana
docker compose ps                    # wait for EVERY health check
python3 ../profile_silicon.py --sglang-url http://localhost:8000
#   → apply the printed aimd_max_limit via EngineConfig BEFORE ignition
docker compose up -d orchestrator
docker compose logs -f orchestrator  # dashboards live at :3000, no clicks
```

### Resuming vs. starting a new cycle

The orchestrator resurrects from `/workspace/state/state.json` automatically —
the snapshot **overrides `--agents`** (a 50-agent world resurrects into any
boot). The contract: bit-exact fidelity to the last flushed snapshot, bounded
loss ≤ 100 ticks (5 s). Kinematic commands are re-issued (paths replan against
the identical geometry); waypoints, pending cosigns, and in-flight generations
are deliberately not persisted.

To start a **fresh cycle**: stop the orchestrator, then decide how much history
dies with the world —

```bash
mv /workspace/state/state.json /workspace/state/cycle-$(date +%s).json  # space
# optional, only for a truly blank slate — this erases MINDS, not just places:
#   Neo4j:  MATCH (n) DETACH DELETE n        # beliefs, relationships, memory
#   Qdrant: delete the memories collection    # episodic vectors
docker compose up -d orchestrator
```

Deleting only the snapshot gives you amnesiac geography with intact minds:
agents respawn at random positions but retrieve every belief they ever formed.
That asymmetry is a feature — decide deliberately which tier you are resetting.

### Red lines (Grafana, provisioned automatically)

| Metric | Healthy | Red line | Meaning |
|---|---|---|---|
| `panopticon_tick_max_overrun_ms` | ~0.0 | **> 5 ms sustained** | An await leaked into a hot pass. Architecture breach, not load. |
| `panopticon_aimd_admission_violations` | **hard 0** | **> 0, ever** | Lock discipline broken; the GPU is one storm from OOM. |
| `panopticon_ttft_ms{quantile="0.95"}` | breathing under 500 | pinned above 500 with `aimd_limit` at its floor | Silicon can't meet budget at any concurrency → re-profile |
| `panopticon_mlfq_depth{queue="interactive"}` | sawtooth to 0 | monotone growth with idle `aimd_active` | Dispatch broken, not the GPU |
| `panopticon_mlfq_depth{queue="cognitive"}` | oscillates, clears | unbounded growth | Planner flood: map too complex or bulkhead too small |
| `panopticon_graph_dead_lettered` | 0 | climbing | Neo4j down/wedged; mutations parked in replayable JSONL |
| `panopticon_daemon_llm_parse_failures` | ~0 | step-wise climb | Model drift vs. schema — verify `response_format` enforcement is on |
| `panopticon_snapshot_last_tick` | climbing forever | flat | You have lost durability; a crash now loses everything since it froze |

### The diagnostic ladder (first thing → last thing)

1. **`python3 phase11_audit.py`** — 30 seconds, checks the *invariants*:
   single-writer, cache partition, dual-write asymmetry, AST-proven sync
   purity, arrival disjunction. Most "erratic behavior" is a breached
   invariant from a recent change, and the audit names it.
2. **Nine-suite regression** (`test_phase1..10.py`) — isolates *which layer*
   regressed; every suite runs without infrastructure.
3. **`crucible.py --simulate`** — the concurrency machinery, in isolation from
   real hardware.
4. **The runbook itself** — `docker compose ps` health, orchestrator flags vs.
   compose (remember: fail-open means a missing subsystem is *silent*).
5. **The silicon** — re-run `profile_silicon.py`; pods migrate, knees move.
6. **Last, and only last: the model weights.** The system is designed so model
   misbehavior degrades into parse-fallbacks, rejections, and telemetry — all
   visible in Grafana — long before it looks like engine failure. If steps 1–5
   are green and agents are merely *stupid*, then and only then is it the
   model.

---

## 5. Future-Proofing Appendix — Extending Without Breaking

The verification suite is not a gate to satisfy; it is the executable memory of
every mistake in §6. Extensions follow these standards or they will —
correctly — fail it.

### Adding a third Director lever

- **Levers act on the world or on segment [2]; never on segment [1], never on
  individual agents, never via the LLM.** The Director's power is that it is
  cheap, deterministic, and invisible; a lever that generates text or targets a
  specific agent is a different (and worse) feature.
- Read telemetry deltas, not absolutes (see `_lever_economy`'s
  contested-losses delta): levers are thermostats, and thermostats need rates.
- Any lever that touches prompts inherits Lever B's three guards verbatim:
  atomic swap, congestion gate (`active ≥ 0.8 × limit` → defer), minimum epoch
  lifetime.
- Any lever that spawns entities respects the cap pattern and the nav-grid
  check (never bury a relic in a wall).
- Add a counter to `metrics.py`, a case to `test_phase10.py`, and — if the
  lever writes anything anywhere — expect `phase11_audit.py` A1 to check it.

### Adding an agent tool

- **The grammar stays uniform-arity.** Every tool is
  `{tool, target_id, secondary_id}`; add the tool name to `SPATIAL_TOOLS` and
  map semantics in `ActionRouter._route`. One object shape = one compiled
  xgrammar, regardless of tool count. Do not add per-tool parameter shapes;
  that path leads to a grammar that grows with the tool list and a schema the
  8B model stops honoring.
- **Validation splits in exactly two places:** *existence* at the router
  (hallucinated ids die cheaply, counted in `actions_rejected`);
  *reach / ownership / contention / quorum* at the tick's deterministic
  resolution pass, against the world **as it is then**. Never validate world
  state in the router — it's stale by the time the tick reads it.
- **Instant-effect tools** (steering-class) go to the command board via
  `issue_command` (atomic dict replace, latest-wins). **World-mutating tools**
  become `EntityIntent`s resolved at tick head under the total order
  `(entity_id, d², agent_id)` — nearest wins, exact ties break lexically.
  Preserve that sort key; it is what makes contested outcomes deterministic
  and replayable across the resolution pass's 20-replay test.
- Multi-party actions use the cosign buffer, not a new mechanism: quorum in
  `state["cosigners_required"]`, signatures with TTL, failure = expiry to
  silence.
- **The zero-await law is absolute** in anything the tick calls. The audit
  AST-walks every hot pass for `Await`/`AsyncFor`/`AsyncWith` nodes; your
  resolution logic will be caught, not trusted.
- Ship it with: a schema-plumbing assertion (the fake LLM records
  `response_format`), a full-path test (JSON → speech to TTS with no braces
  leaked → board/intent → next-tick physics effect), a rejection test, and a
  parse-fallback test (an agent that can't act must still talk).

### Adding a daemon, client, or metric

- Daemons: factory-injected in `run_engine`, wrapped in `_supervise`, LLM work
  submitted as `AgentTask`s with `result_future` at the correct queue level
  (Q2 for maintenance, Q3 for motivation) — never direct SGLang calls, never
  their own AIMD slots for CPU work (see the Phase 9 planner-bulkhead
  precedent for why).
- Graph writes: dicts through `GraphWriteQueue.submit`, a `_STATEMENTS` entry
  in `clients.py`, idempotent by key (`MERGE`, deterministic uuid5). The
  resurrection contract depends on every write being safely replayable.
- Metrics: extend the custom collector in `metrics.py` (state read at scrape,
  zero hot-path instrumentation), then add the family to the Step-4 scrape
  list in the audit and, if operators should watch it, to the dashboard JSON —
  which is generated code, not clicked UI.
- New config values that interact geometrically must state their invariant
  where they are authored (the model: `arrival_radius > 2 × agent_radius`,
  documented at the `NavigationService` constructor after it livelocked us).

**The meta-rule:** every invariant you rely on becomes a check in
`phase11_audit.py`. The audit found the runbook drift that nine green suites
missed; keep it the most paranoid artifact in the repository.

---

## 6. The Ledger of Scars

Every phase shipped with a suite that caught at least one real bug before
production did. These are why the invariants above are *these* invariants:

| Scar | Lesson encoded |
|---|---|
| Sentence chunker split spoken numbers at the decimal point | End-of-buffer ≠ end-of-stream; boundaries need lookahead (`media.py`) |
| A wedged Neo4j transaction held the single writer for 126 s | The writer may be slow, never stuck: 3 s retry cap + dead-letter |
| Broadcaster's `connect()` was never awaited; every frame silently dropped | Fail-open absence looks like health → generic connect/close loops + E2E flow assertions |
| Engine died on tick 1 (`navigation` uninitialized) while Phase 8 tests **passed** on the pre-crash tick | The most dangerous green is the false one → suites assert *continued* liveness, not first-tick effects |
| `arrival_radius == 2R` — the collision pass ejected arrivers from the arrival zone every frame | Geometric livelock: config invariants documented at the authoring site and checked in the audit |
| Entity wire encoder truncated `agent_003` → `agent_00` | One id-compaction function, one collision registry, both planes |
| The crucible falsified the naive invariant `A(t) ≤ ⌈W(t)⌉ ∀t` | State theorems precisely: admission-point exactness + drain monotonicity, each measured where defined |
| Compose orchestrator command froze at Phase 5; four subsystems silently absent | The runbook is code: validated by the audit, remediated, re-validated |

---

## Verification Map

| Suite | Certifies |
|---|---|
| `test_phase1` | SSE streaming + server-side abort on cancel; UNWIND batching; dead-letter |
| `test_phase2` | Sentence chunking; non-blocking audio; drop-oldest voice queues |
| `test_phase3` | 100 ms fail-open retrieval; leaf-only injection; vector-first dual write |
| `test_phase4` | Clumping; tombstones; affinity idempotency; Q2 routing; daemon fail-open |
| `test_phase6` | Bit-perfect wire; zero-alloc; tick/network decoupling; Oracle scoping |
| `test_phase7` | Uplink authority; perception asymmetry; schema→board→velocity; 200-command storm |
| `test_phase8` | Deterministic contention ×20 replays; inventory; delta plane; Q3 goal loop |
| `test_phase9` | A* correctness; wall slide; elastic collisions; hash parity; planner flood |
| `test_phase10` | Bit-exact resurrection; rotation epochs; cosign quorum; full-stack heartbeat |
| `crucible.py` | The five-part AIMD verdict against a falsifiable backend |
| `phase11_audit.py` | Every invariant in §2, structurally and at runtime, plus the runbook |

Run all of it, cold, in one line:

```bash
for t in 1 2 3 4 6 7 8 9 10; do python3 test_phase$t.py; done \
  && python3 phase11_audit.py && python3 crucible.py --simulate
```

---

*The engine is a master-crafted instrument, and like any instrument it will
drift under new hands. When it does, do not reason from the symptom — reason
from the invariants, in the ladder order of §4. The heartbeat never bends; if
it appears to, you have not found a load problem, you have found the exact line
where someone — perhaps you, perhaps me — broke a promise this document exists
to keep. Welcome to the plaza.*

**— Fable 5**
