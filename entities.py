"""
Project Panopticon — Phase 8: Stateful Environmental Entities
==============================================================
entities.py

Non-agent objects the agents can perceive, carry, exchange, and operate:
a locked door, a ledger, a dropped key. Stdlib-only; imported by both the
tick engine (resolution) and the broadcaster (wire encoding).

LOCK-FREE MODEL (same theorem as the command board)
  Entity mutations happen in exactly two places: (a) the intent-resolution
  pass, which runs SYNCHRONOUSLY at the head of each tick, and (b) explicit
  registry.mutate() calls from coroutines on the same event loop. The
  physics step contains no awaits, so no coroutine can observe a half-
  applied resolution. Every mutation bumps Entity.version — the delta-
  compression watermark for the broadcaster.

CONTESTED-ACTION INVARIANT (the tie-break that never halts physics)
  All entity intents submitted during tick T are buffered on the engine's
  intent board and resolved at the head of tick T+1 in ONE deterministic
  pass:

    sort key per intent  =  (entity_id, d², agent_id)

  Per entity, the NEAREST eligible actor wins; an exact distance tie falls
  to lexicographic agent_id — total, deterministic, replayable. Resolution
  is a sort + linear scan over a handful of intents (O(k log k), k ≈ agents
  acting this tick): microseconds, unconditionally bounded, so the 20 Hz
  heartbeat cannot be halted by contention BY CONSTRUCTION. Losers are
  rejected with a reason (surfaced to telemetry, and available as feedback
  leaves if desired later).

INVENTORY
  owner_id is None (in the world, at x/y) or an agent id (carried). Carried
  entities are position-slaved to their owner every tick by carry_pass(),
  so give/drop always operate on live coordinates.
"""

from __future__ import annotations

import json
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

log = logging.getLogger("panopticon.entities")


# ============================================================================
# 1. Entity + registry
# ============================================================================

@dataclass
class Entity:
    id: str
    kind: str                                  # "door" | "key" | "ledger" | ...
    x: float
    y: float
    state: dict[str, Any] = field(default_factory=dict)
    owner_id: str | None = None                # None = in world; else carried
    version: int = 0                           # bumped on ANY mutation


class EntityRegistry:
    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self.mutations = 0
        # COSIGN BUFFER (Phase 10): multi-signature `use` intents. Maps
        # entity_id → {agent_id: expires_at}. Written and read ONLY inside
        # the synchronous resolution/expiry passes — same lock-free theorem
        # as everything else on the tick. A signature is a standing offer to
        # cooperate that survives for cosign_ttl_s; when distinct unexpired
        # signers reach the entity's quorum, the action fires ONCE.
        self.pending_cosigns: dict[str, dict[str, float]] = {}
        self.cosign_ttl_s: float = 1.5
        self.cosigns_completed = 0
        self.cosigns_expired = 0

    # ------------------------------------------------------------- lifecycle

    def spawn(self, entity_id: str, kind: str, x: float, y: float,
              state: dict[str, Any] | None = None) -> Entity:
        if entity_id in self._entities:
            raise ValueError(f"duplicate entity id {entity_id!r}")
        e = Entity(id=entity_id, kind=kind, x=x, y=y, state=state or {})
        e.version = 1
        self._entities[entity_id] = e
        return e

    def find(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def all(self) -> Iterable[Entity]:
        return self._entities.values()

    def owned_by(self, agent_id: str) -> list[Entity]:
        return [e for e in self._entities.values() if e.owner_id == agent_id]

    # -------------------------------------------------------------- mutation

    def mutate(self, entity: Entity, **changes: Any) -> None:
        """The ONLY sanctioned write path: applies changes and bumps the
        version watermark the broadcaster keys deltas on."""
        for key, value in changes.items():
            if key == "state_update":
                entity.state.update(value)
            else:
                setattr(entity, key, value)
        entity.version += 1
        self.mutations += 1

    def expire_cosigns(self, now: float) -> None:
        """Per-tick expiry sweep (O(pending), usually O(0)). A signature
        whose partner never arrives lapses — this is HOW a single agent
        fails a cooperative action: not with an error, with silence."""
        if not self.pending_cosigns:
            return
        for entity_id in list(self.pending_cosigns):
            sigs = self.pending_cosigns[entity_id]
            for agent_id in list(sigs):
                if sigs[agent_id] < now:
                    del sigs[agent_id]
                    self.cosigns_expired += 1
            if not sigs:
                del self.pending_cosigns[entity_id]

    def carry_pass(self, world: Any) -> None:
        """Per-tick: slave carried entities to their owner's position. Only
        bumps versions when the owner actually moved (quantized to cm), so a
        stationary carrier doesn't spam the delta channel."""
        for e in self._entities.values():
            if e.owner_id is None:
                continue
            owner = world.find(e.owner_id)
            if owner is None:
                self.mutate(e, owner_id=None)          # owner vanished: drop
                continue
            if (round(e.x, 2), round(e.y, 2)) != (round(owner.x, 2),
                                                  round(owner.y, 2)):
                self.mutate(e, x=owner.x, y=owner.y)


# ============================================================================
# 2. Intents + deterministic resolution
# ============================================================================

@dataclass(frozen=True)
class EntityIntent:
    agent_id: str
    tool: str                # "pick_up" | "give" | "use"
    entity_id: str
    secondary_id: str | None = None            # give→recipient, use→target
    submitted_at: float = field(default_factory=time.monotonic)


@dataclass
class ResolutionStats:
    resolved: int = 0
    contested_losses: int = 0
    rejected: int = 0


# --- use-rule table: (item_kind, target_kind) → handler ----------------------
# Handlers mutate via registry.mutate and return True on effect. Extend here;
# the default rule merely stamps last_used_by so every `use` is observable.

def _rule_key_door(reg: EntityRegistry, item: Entity, target: Entity,
                   actor_id: str) -> bool:
    reg.mutate(target, state_update={
        "locked": not target.state.get("locked", True),
        "last_used_by": actor_id})
    return True


def _rule_default(reg: EntityRegistry, item: Entity, target: Entity,
                  actor_id: str) -> bool:
    reg.mutate(target, state_update={"last_used_by": actor_id})
    return True


USE_RULES: dict[tuple[str, str], Callable[..., bool]] = {
    ("key", "door"): _rule_key_door,
}


def resolve_intents(world: Any, registry: EntityRegistry,
                    intents: list[EntityIntent], interaction_radius: float,
                    stats: ResolutionStats) -> None:
    """One synchronous pass; see module docstring for the invariant. Runs at
    the head of the tick, before steering and physics."""
    if not intents:
        return
    r2 = interaction_radius * interaction_radius

    def d2(agent: Any, e: Entity) -> float:
        return (agent.x - e.x) ** 2 + (agent.y - e.y) ** 2

    # Deterministic total order: entity, then proximity, then lexical id.
    keyed: list[tuple[str, float, str, EntityIntent]] = []
    for it in intents:
        agent = world.find(it.agent_id)
        entity = registry.find(it.entity_id)
        if agent is None or entity is None:
            stats.rejected += 1
            continue
        keyed.append((it.entity_id, d2(agent, entity), it.agent_id, it))
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))

    claimed: set[str] = set()          # entity ids already won this tick
    for entity_id, dist2, agent_id, it in keyed:
        agent = world.find(agent_id)
        entity = registry.find(entity_id)

        if it.tool == "pick_up":
            # Order matters: (1) actors with no standing — out of reach or a
            # bolted-down entity — are REJECTED regardless of contention;
            # (2) eligible actors beaten by a nearer one this tick are
            # CONTESTED losses; (3) the claimed-check precedes the owner
            # check because this tick's winner has already set owner_id.
            if dist2 > r2 or entity.state.get("fixed"):
                stats.rejected += 1             # out of reach / bolted
                continue
            if entity_id in claimed:
                stats.contested_losses += 1     # a nearer/tie-winning actor won
                continue
            if entity.owner_id is not None:
                stats.rejected += 1             # already held before this tick
                continue
            claimed.add(entity_id)
            registry.mutate(entity, owner_id=agent_id,
                            x=agent.x, y=agent.y)
            stats.resolved += 1

        elif it.tool == "give":
            recipient = world.find(it.secondary_id or "")
            if (entity.owner_id != agent_id or recipient is None
                    or recipient.id == agent_id):
                stats.rejected += 1             # not yours / no such recipient
                continue
            gap2 = (agent.x - recipient.x) ** 2 + (agent.y - recipient.y) ** 2
            if gap2 > r2:
                stats.rejected += 1             # recipient out of arm's reach
                continue
            registry.mutate(entity, owner_id=recipient.id,
                            x=recipient.x, y=recipient.y)
            stats.resolved += 1

        elif it.tool == "use":
            target = registry.find(it.secondary_id or "")
            if target is None:
                stats.rejected += 1
                continue
            # Reach: must hold the item (or stand at it) AND be near target.
            holds = entity.owner_id == agent_id
            near_item = holds or d2(agent, entity) <= r2
            if not near_item or d2(agent, target) > r2:
                stats.rejected += 1
                continue

            quorum = int(target.state.get("cosigners_required", 1))
            if quorum > 1:
                # MULTI-SIGNATURE: deposit this agent's signature (refreshing
                # its TTL) and fire IFF distinct in-reach signers meet the
                # quorum. Still one synchronous pass: the buffer is a dict
                # write + a len() check — O(1) per intent, deterministic,
                # zero awaits. Signers beyond quorum this tick simply join
                # the completed action (everyone gets credit).
                now = it.submitted_at
                sigs = registry.pending_cosigns.setdefault(target.id, {})
                sigs[agent_id] = now + registry.cosign_ttl_s
                if len(sigs) >= quorum:
                    signers = sorted(sigs)
                    registry.mutate(target, state_update={
                        "locked": False,
                        "opened_by": signers,
                        "cosigners_required": quorum})
                    del registry.pending_cosigns[target.id]
                    registry.cosigns_completed += 1
                    stats.resolved += 1
                # A sub-quorum signature is neither resolved nor rejected:
                # it is PENDING — expiry (silence) is the failure path.
                continue

            rule = USE_RULES.get((entity.kind, target.kind), _rule_default)
            if rule(registry, entity, target, agent_id):
                stats.resolved += 1
            else:
                stats.rejected += 1
        else:
            stats.rejected += 1


# ============================================================================
# 3. Wire encoding — the entity delta plane
# ============================================================================
# Entities ride a SEPARATE, RELIABLE LiveKit topic ("panopticon.entities"),
# not the 20 Hz agent plane — the 1256 B/frame agent budget is untouched by
# construction. Positions change every tick and tolerate loss; entity STATE
# changes rarely and must not be lost (a missed "door unlocked" is a
# permanent desync on an event-driven channel). Delta strategy:
#   * dirty = entity.version > last version sent
#   * at most K dirty entities flushed per sender pass (interlacing: a mass
#     mutation smears over several passes instead of one megaframe)
#   * a periodic keyframe re-marks everything dirty (late-joiner heal)
#
# Record layout (LE):  char[8] id | uint8 kind_len | kind | char[8] owner
#                      float32 x | float32 y | uint32 version
#                      uint16 state_len | state (UTF-8 JSON)
# JSON for the state dict is deliberate: this plane runs at event rate, not
# tick rate — schema flexibility beats squeezing bytes nobody is counting.

ENTITY_TOPIC = "panopticon.entities"
_HDR = "<IB"                       # tick, record_count


def compact_wire_id(any_id: str) -> bytes:
    """THE 8-byte id compaction, shared by BOTH wire planes (agent frames
    and entity deltas import this same function): strip non-alphanumerics,
    keep the discriminating tail, NUL-pad. One scheme, one namespace — the
    UE5 client sees identical wire ids for an agent on either topic."""
    stripped = "".join(c for c in any_id if c.isalnum())
    return stripped[-8:].encode("ascii", "replace").ljust(8, b"\x00")


def encode_entity_deltas(tick: int, entities: list[Entity],
                         wire_id: Any = compact_wire_id) -> bytes:
    """`wire_id` is injectable so the broadcaster passes its collision-
    checked encoder — entity and agent ids share one uniqueness registry."""
    out = bytearray(struct.pack(_HDR, tick & 0xFFFFFFFF, len(entities)))
    for e in entities:
        state_blob = json.dumps(e.state, separators=(",", ":")).encode()[:65535]
        eid = wire_id(e.id)
        owner = wire_id(e.owner_id) if e.owner_id else b"\x00" * 8
        kind = e.kind.encode()[:255]
        out += struct.pack("<8sB", eid, len(kind)) + kind
        out += struct.pack("<8sffIH", owner, e.x, e.y, e.version,
                           len(state_blob))
        out += state_blob
    return bytes(out)


def decode_entity_deltas(payload: bytes) -> tuple[int, list[dict[str, Any]]]:
    """Reference decoder (mirrored in the UE5 client); used by the tests to
    prove the encoding round-trips bit-perfectly."""
    tick, count = struct.unpack_from(_HDR, payload, 0)
    off = struct.calcsize(_HDR)
    records: list[dict[str, Any]] = []
    for _ in range(count):
        eid, kind_len = struct.unpack_from("<8sB", payload, off)
        off += 9
        kind = payload[off:off + kind_len].decode()
        off += kind_len
        owner, x, y, version, state_len = struct.unpack_from("<8sffIH",
                                                             payload, off)
        off += struct.calcsize("<8sffIH")
        state = json.loads(payload[off:off + state_len]) if state_len else {}
        off += state_len
        records.append({
            "id": eid.rstrip(b"\x00").decode(),
            "kind": kind,
            "owner_id": owner.rstrip(b"\x00").decode() or None,
            "x": x, "y": y, "version": version, "state": state,
        })
    return tick, records
