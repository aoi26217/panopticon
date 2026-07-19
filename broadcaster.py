"""
Project Panopticon — Phase 6: The Glasshouse (server half, stream out)
=======================================================================
broadcaster.py

Streams the 20 Hz spatial state to every connected UE5 observer over a
LiveKit WebRTC data channel — binary, unreliable, latest-wins.

WIRE FORMAT (little-endian, packed, no padding) — keep in exact sync with
FGlasshousePacket in the UE5 client (AMagnusObserver.h):

    HEADER  (6 bytes)
        uint32  tick_count
        uint16  agent_count
    PER AGENT (25 bytes, repeated agent_count times)
        char[8] agent_id        # deterministic 8-byte compaction, see below
        float32 x
        float32 y
        float32 vx
        float32 vy
        uint8   flags           # bit0 = is_speaking, bit1 = is_degraded

    total = 6 + 25 * agent_count      (50 agents → 1256 bytes/frame,
                                       ~25 KB/s — trivially inside any
                                       data-channel budget)

THE THREE DESIGN RULES

1. Zero allocation on the tick path. One bytearray is allocated at startup,
   sized for max_agents; every capture is struct.pack_into over that same
   buffer. The ONLY per-frame allocation is the final bytes() copy at the
   SDK boundary (LiveKit requires an immutable payload) — and that happens
   on the SENDER task, never inside the tick.

2. Latest-wins decoupling. The tick hook writes into the pack buffer,
   snapshots it into a 1-deep ring, and sets an event — microseconds, no
   awaits. A separate sender coroutine drains the ring at whatever rate the
   network sustains. If the network is slower than 20 Hz, intermediate
   frames are silently superseded (a stale position frame has negative
   value to an interpolating observer). Network latency therefore CANNOT
   produce a tick overrun, by construction.

3. Unreliable transport. reliable=False: a dropped frame is healed by the
   next one 50 ms later, whereas a TCP-style retransmit stall would freeze
   every pawn in the glasshouse. The UE5 client's Hermite interpolation is
   specifically there to ride through single-frame loss.

Fail-open: a send failure is counted and logged; frames drop, the
simulation never notices. The broadcaster observes the engine — it is
never allowed to become load-bearing.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Any, Callable, Iterable

from livekit import api, rtc

from entities import ENTITY_TOPIC, compact_wire_id, encode_entity_deltas

log = logging.getLogger("panopticon.broadcaster")

# --- Wire format constants (single source of truth for the test suite) -----
HEADER_FMT = "<IH"                      # tick_count, agent_count
AGENT_FMT = "<8sffffB"                  # id, x, y, vx, vy, flags
HEADER_SIZE = struct.calcsize(HEADER_FMT)    # 6
AGENT_SIZE = struct.calcsize(AGENT_FMT)      # 25
FLAG_SPEAKING = 0b0000_0001
FLAG_DEGRADED = 0b0000_0010
DATA_TOPIC = "panopticon.state"


class AgentIdOverflowError(ValueError):
    """Two agent ids compacted to the same 8-byte wire id. The wire format
    is char[8]; ids must be uniquely representable. Fail at startup, loudly —
    never at frame 40,000, silently."""


# Single compaction scheme for the whole wire (agents AND entities) — the
# implementation lives in entities.py; this alias preserves the public name.
compact_agent_id = compact_wire_id


class StateBroadcaster:
    """Owns its own LiveKit participant (separate from the audio publisher,
    so observers can subscribe to state and voice independently)."""

    def __init__(
        self,
        url: str,
        api_key: str,
        api_secret: str,
        *,
        room_name: str = "panopticon",
        identity: str = "panopticon-broadcaster",
        max_agents: int = 256,
        speaking_probe: Callable[[str], bool] | None = None,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._api_secret = api_secret
        self._room_name = room_name
        self._identity = identity
        self._speaking = speaking_probe or (lambda _aid: False)

        # THE buffer — allocated once, packed into forever (rule 1).
        self._buf = bytearray(HEADER_SIZE + AGENT_SIZE * max_agents)
        self._max_agents = max_agents
        self._id_cache: dict[str, bytes] = {}
        self._wire_ids: dict[bytes, str] = {}

        # Latest-wins ring (rule 2): depth 1, newest frame replaces unsent.
        self._pending: bytes | None = None
        self._frame_ready = asyncio.Event()

        self._room: rtc.Room | None = None
        # Entity delta plane (Phase 8): rides the SAME sender task on a
        # separate RELIABLE topic. The 20 Hz agent frame stays byte-for-byte
        # identical — the 1256 B/frame budget is untouched by construction.
        self._registry: Any = None
        self._sent_versions: dict[str, int] = {}
        self._last_keyframe = 0.0
        self.entity_flush_max = 8          # interlace cap per sender pass
        self.entity_keyframe_s = 10.0      # late-joiner full-state heal
        # Telemetry
        self.frames_captured = 0
        self.frames_sent = 0
        self.frames_superseded = 0
        self.send_failures = 0
        self.last_sent_tick = -1
        self.entity_deltas_sent = 0
        self.entity_keyframes = 0

    # ------------------------------------------------------------ lifecycle

    async def connect(self) -> None:
        token = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(self._identity)
            .with_name("Panopticon State Broadcaster")
            .with_grants(api.VideoGrants(
                room_join=True, room=self._room_name,
                can_publish=True, can_publish_data=True, can_subscribe=False,
            ))
            .to_jwt()
        )
        self._room = rtc.Room()
        await self._room.connect(self._url, token)
        log.info("Broadcaster connected: room=%s topic=%s",
                 self._room_name, DATA_TOPIC)

    async def close(self) -> None:
        if self._room is not None:
            await self._room.disconnect()

    # ------------------------------------------------- tick-side (hot path)

    def _wire_id(self, agent_id: str) -> bytes:
        wid = self._id_cache.get(agent_id)
        if wid is None:
            wid = compact_agent_id(agent_id)
            owner = self._wire_ids.get(wid)
            if owner is not None and owner != agent_id:
                raise AgentIdOverflowError(
                    f"{agent_id!r} and {owner!r} both compact to {wid!r}")
            self._id_cache[agent_id] = wid
            self._wire_ids[wid] = agent_id
        return wid

    def capture(self, tick_count: int, agents: Iterable[Any]) -> None:
        """Called from the TickEngine tail. Synchronous, no awaits, no heap
        churn beyond the single snapshot handed to the sender. Any exception
        is swallowed by the engine's hook guard (fail-open), but this method
        is written not to raise in steady state."""
        buf = self._buf
        offset = HEADER_SIZE
        n = 0
        for agent in agents:
            if n >= self._max_agents:
                break
            flags = 0
            if self._speaking(agent.id):
                flags |= FLAG_SPEAKING
            if agent.degraded:
                flags |= FLAG_DEGRADED
            struct.pack_into(AGENT_FMT, buf, offset,
                             self._wire_id(agent.id),
                             agent.x, agent.y, agent.vx, agent.vy, flags)
            offset += AGENT_SIZE
            n += 1
        struct.pack_into(HEADER_FMT, buf, 0, tick_count & 0xFFFFFFFF, n)

        # SDK boundary copy (immutable payload) + latest-wins publish.
        if self._pending is not None:
            self.frames_superseded += 1          # network lagging: coalesce
        self._pending = bytes(memoryview(buf)[:offset])
        self.frames_captured += 1
        self._frame_ready.set()

    def attach_entities(self, registry: Any) -> None:
        """Enable the entity plane. Dirty detection is the version watermark:
        anything whose Entity.version exceeds what we last shipped."""
        self._registry = registry

    def make_tick_hook(self) -> Callable[[int, Any], None]:
        """Adapter for TickEngine.set_tick_hook."""
        return lambda tick, world: self.capture(tick, world.agents)

    # ------------------------------------------------------- sender (I/O)

    async def run_sender(self) -> None:
        """Drains the ring at network pace. Rule 2 lives here: this loop can
        be arbitrarily slow without ever touching tick timing."""
        while True:
            await self._frame_ready.wait()
            self._frame_ready.clear()
            frame = self._pending
            self._pending = None
            if frame is None or self._room is None:
                continue
            try:
                await self._room.local_participant.publish_data(
                    frame, reliable=False, topic=DATA_TOPIC)
                self.frames_sent += 1
                self.last_sent_tick = struct.unpack_from(HEADER_FMT, frame)[0]
                await self._flush_entity_deltas(self.last_sent_tick)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.send_failures += 1
                if self.send_failures in (1, 10, 100) or \
                        self.send_failures % 1000 == 0:
                    log.warning("State frame send failed (x%d) — dropping, "
                                "simulation unaffected", self.send_failures,
                                exc_info=True)


    async def _flush_entity_deltas(self, tick: int) -> None:
        """Interlaced delta flush: at most `entity_flush_max` dirty entities
        per pass, oldest-id-first for determinism, so a mass mutation (fifty
        doors slam) smears over several passes instead of one megaframe.
        Worst case per pass: 8 records × ~90 B ≈ 720 B on the RELIABLE plane
        — a different budget from the 20 Hz agent frame entirely. Reliable
        because entity STATE is event-sourced: a lost "door unlocked" would
        be a permanent desync, unlike a lost position healed 50 ms later."""
        if self._registry is None:
            return
        now = time.monotonic()
        if now - self._last_keyframe > self.entity_keyframe_s:
            self._sent_versions.clear()            # everything dirty → heal
            self._last_keyframe = now
            self.entity_keyframes += 1
        dirty = sorted(
            (e for e in self._registry.all()
             if e.version > self._sent_versions.get(e.id, 0)),
            key=lambda e: e.id)[: self.entity_flush_max]
        if not dirty:
            return
        payload = encode_entity_deltas(tick, dirty, wire_id=self._wire_id)
        try:
            await self._room.local_participant.publish_data(
                payload, reliable=True, topic=ENTITY_TOPIC)
            for e in dirty:
                self._sent_versions[e.id] = e.version
            self.entity_deltas_sent += len(dirty)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.send_failures += 1                # unsent stays dirty: retry


def livekit_speaking_probe(publisher: Any) -> Callable[[str], bool]:
    """Wire the broadcaster's is_speaking flag to the Phase 2 audio
    publisher: an agent is 'speaking' while its speaker channel has queued
    or in-flight sentences. Cheap dict lookups only — tick-path safe."""

    def probe(agent_id: str) -> bool:
        channel = getattr(publisher, "_channels", {}).get(agent_id)
        return bool(channel) and not channel.sentences.empty()

    return probe
