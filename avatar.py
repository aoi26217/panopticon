"""
Project Panopticon — Phase 7: The Architect's Embodiment & Spatial Agency
==========================================================================
avatar.py

The ghost becomes flesh, in three couplings:

  UPLINK   UE5 → Python : the human's (x, y) streamed over a LiveKit data
           channel into a World agent (`magnus`) that other agents perceive
           exactly as they perceive each other. Physics never integrates it;
           the wire is authoritative.
  VOICE    UE5 mic → LiveKit audio track → streaming STT → Q0 tasks for the
           nearest agents, framed as the human's direct real-time speech.
  AGENCY   agents → physics : schema-enforced JSON generations carrying
           {say, action}; actions become KinematicCommands on the
           TickEngine's lock-free command board and turn into velocity the
           very next tick.

CONCURRENCY MODEL (why there are no locks in this file)
  Every writer here — the uplink data callback, the STT transcript handler,
  the dispatcher's intercept() — runs on the SAME asyncio event loop as the
  TickEngine. The physics step is a synchronous function: between its first
  and last instruction, no callback can run. Position writes and command-
  board assignments are therefore atomic with respect to physics BY THE
  EVENT LOOP'S OWN SCHEDULING. Locks would add latency to defend against an
  interleaving that cannot occur.

UPLINK WIRE FORMAT (little-endian, packed; UE5 side sends at 20–30 Hz):
    uint32  seq          # client-monotonic; stale/reordered packets dropped
    float32 x
    float32 y
Total 12 bytes on topic "panopticon.avatar". Send unreliable from UE5:
latest-wins is the correct semantic for a position, and a lost packet is
healed 33 ms later.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

import aiohttp
from livekit import api, rtc

from entities import EntityIntent
from tick_engine import AgentTask, KinematicCommand, QueueLevel, EMBODIED_WORLD_PREFIX

log = logging.getLogger("panopticon.avatar")

HUMAN_AVATAR_ID = "magnus"          # compacts cleanly to b"magnus\x00\x00"
AVATAR_TOPIC = "panopticon.avatar"
UPLINK_FMT = "<Iff"                 # seq, x, y
UPLINK_SIZE = struct.calcsize(UPLINK_FMT)   # 12


# ============================================================================
# 1. The Uplink — UE5 position → World
# ============================================================================

class AvatarUplink:
    """Owns a subscribe-capable LiveKit participant; consumes avatar packets
    and writes them into the human's World agent. Velocity is derived from
    consecutive positions so the downstream broadcaster echoes proper Hermite
    tangents and OTHER observers see the human glide, not teleport."""

    def __init__(
        self,
        url: str,
        api_key: str,
        api_secret: str,
        *,
        room_name: str = "panopticon",
        identity: str = "panopticon-avatar-uplink",
        avatar_id: str = HUMAN_AVATAR_ID,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._api_secret = api_secret
        self._room_name = room_name
        self._identity = identity
        self._avatar_id = avatar_id
        self._room: rtc.Room | None = None
        self._world: Any = None
        self._human: Any = None
        self._last_seq = -1
        self._last_recv: float | None = None
        # Telemetry
        self.packets_applied = 0
        self.packets_stale = 0
        self.packets_malformed = 0

    def bind(self, world: Any) -> None:
        """Late-bind the live World and materialize the human entity."""
        self._world = world
        self._human = world.find(self._avatar_id) or world.spawn_human(
            self._avatar_id)
        log.info("Human avatar %r embodied at (%.1f, %.1f)",
                 self._avatar_id, self._human.x, self._human.y)

    async def connect(self) -> None:
        token = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(self._identity)
            .with_grants(api.VideoGrants(
                room_join=True, room=self._room_name,
                can_subscribe=True, can_publish=False,
                can_publish_data=False,
            ))
            .to_jwt()
        )
        self._room = rtc.Room()
        self._room.on("data_received", self._on_data)
        await self._room.connect(self._url, token)
        log.info("Avatar uplink listening on topic %r", AVATAR_TOPIC)

    async def close(self) -> None:
        if self._room is not None:
            await self._room.disconnect()

    # ---- hot path: sync callback on the event loop -------------------------

    def _on_data(self, packet: Any) -> None:
        """rtc data callback. Synchronous, allocation-light, fail-open."""
        try:
            if getattr(packet, "topic", "") != AVATAR_TOPIC:
                return
            data = packet.data
            if len(data) != UPLINK_SIZE:
                self.packets_malformed += 1
                return
            seq, x, y = struct.unpack(UPLINK_FMT, data)
            if seq <= self._last_seq:
                self.packets_stale += 1          # reordered/duplicate: drop
                return
            self._last_seq = seq
            self._apply(x, y)
        except Exception:
            self.packets_malformed += 1

    def _apply(self, x: float, y: float) -> None:
        if self._human is None or self._world is None:
            return
        size = self._world._cfg.world_size
        x = min(max(x, 0.0), size)               # clamp: the client is
        y = min(max(y, 0.0), size)               # untrusted input, always
        now = time.monotonic()
        h = self._human
        if self._last_recv is not None:
            dt = now - self._last_recv
            if 0.001 < dt < 1.0:                  # sane interval → velocity
                h.vx = (x - h.x) / dt             # (Hermite tangents for the
                h.vy = (y - h.y) / dt             # downstream broadcaster)
            else:
                h.vx = h.vy = 0.0
        h.x, h.y = x, y
        self._last_recv = now
        self.packets_applied += 1


# ============================================================================
# 2. Voice Ingress — human speech → nearest agents, at Q0
# ============================================================================

class STTEngine(Protocol):
    """Streaming speech-to-text. push() receives 48 kHz mono PCM s16le frames
    from the LiveKit audio stream; transcripts() yields FINAL utterance
    strings (partials are the STT service's internal business)."""

    async def push(self, pcm: bytes) -> None: ...
    def transcripts(self) -> AsyncIterator[str]: ...
    async def close(self) -> None: ...


class WsSTTEngine:
    """Websocket client for a streaming STT microservice (a faster-whisper /
    conformer wrapper on the multimodal GPU, or a hosted provider gateway).
    Contract: binary PCM frames up, {"text": str, "final": bool} JSON down.
    Fail-open: a dead STT service mutes the human; it never touches the sim."""

    def __init__(self, ws_url: str) -> None:
        self._url = ws_url
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._finals: asyncio.Queue[str] = asyncio.Queue(maxsize=16)
        self._reader: asyncio.Task | None = None
        self.push_failures = 0

    async def _ensure(self) -> None:
        if self._ws is None or self._ws.closed:
            self._session = self._session or aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(self._url, heartbeat=15)
            self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self._ws is not None
        async for msg in self._ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                obj = json.loads(msg.data)
                if obj.get("final") and obj.get("text", "").strip():
                    if self._finals.full():
                        self._finals.get_nowait()     # latest speech wins
                    self._finals.put_nowait(obj["text"].strip())
            except (json.JSONDecodeError, KeyError):
                continue

    async def push(self, pcm: bytes) -> None:
        try:
            await self._ensure()
            assert self._ws is not None
            await self._ws.send_bytes(pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.push_failures += 1               # fail-open: frame dropped

    async def transcripts(self) -> AsyncIterator[str]:
        while True:
            yield await self._finals.get()

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()


@dataclass(frozen=True)
class VoiceConfig:
    hearing_radius: float = 8.0       # world units within which speech lands
    max_listeners: int = 3            # nearest-k cap: a whisper is not a PA
    human_identity: str = "magnus-ue5"  # LiveKit identity of the UE5 client


class VoiceIngress:
    """Attaches to the human participant's LiveKit audio track, pumps PCM
    into the STT engine, and converts each final transcript into INTERACTIVE
    tasks for the nearest agents. Q0 + the embodied prefix's rule 4 ("the
    human's words take precedence") together give human speech priority over
    inter-agent chatter at BOTH the scheduling and the reasoning layer."""

    def __init__(
        self,
        stt: STTEngine,
        uplink: AvatarUplink,
        scheduler: Any,
        world: Any,
        cfg: VoiceConfig | None = None,
    ) -> None:
        self._stt = stt
        self._uplink = uplink
        self._scheduler = scheduler
        self._world = world
        self._cfg = cfg or VoiceConfig()
        self._pump: asyncio.Task | None = None
        # Telemetry
        self.utterances = 0
        self.agents_addressed = 0

    def attach(self, room: rtc.Room) -> None:
        """Hook track subscription on an existing room (the uplink's)."""
        room.on("track_subscribed", self._on_track)

    def _on_track(self, track: Any, publication: Any, participant: Any) -> None:
        if (getattr(participant, "identity", "") != self._cfg.human_identity
                or getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO):
            return
        log.info("Human voice track subscribed — STT pump starting")
        self._pump = asyncio.create_task(self._pump_audio(track),
                                         name="stt-pump")

    async def _pump_audio(self, track: Any) -> None:
        stream = rtc.AudioStream(track, sample_rate=48_000, num_channels=1)
        try:
            async for event in stream:
                await self._stt.push(bytes(event.frame.data))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("STT audio pump died — human is muted (fail-open)")

    async def run(self) -> None:
        async for text in self._stt.transcripts():
            self.inject_speech(text)

    # ---- injection (sync; tests drive it directly) --------------------------

    def inject_speech(self, text: str) -> int:
        """Final transcript → Q0 tasks for the nearest-k agents inside the
        hearing radius, measured from the avatar's live position."""
        human = self._world.find(HUMAN_AVATAR_ID)
        if human is None or not text.strip():
            return 0
        text = text.strip()[:400]
        r2 = self._cfg.hearing_radius ** 2
        in_range = sorted(
            ((a, (a.x - human.x) ** 2 + (a.y - human.y) ** 2)
             for a in self._world.agents if not a.is_human),
            key=lambda t: t[1],
        )
        targets = [a for a, d2 in in_range if d2 <= r2][: self._cfg.max_listeners]
        now = time.monotonic()
        leaf = (f"{HUMAN_AVATAR_ID}, the human visitor standing before you, "
                f'speaks to you directly, right now: "{text}" '
                f"Respond to them.")
        for agent in targets:
            self._scheduler.submit_nowait(AgentTask(
                agent_id=agent.id, kind="dialogue",
                level=QueueLevel.INTERACTIVE,
                messages=[
                    {"role": "system", "content": EMBODIED_WORLD_PREFIX},
                    {"role": "user", "content": leaf},
                ]))
            agent.last_reasoned_at = now
        self.utterances += 1
        self.agents_addressed += len(targets)
        log.info("Human spoke to %d agent(s): %.60r", len(targets), text)
        return len(targets)


# ============================================================================
# 3. Spatial agency — schema-enforced actions → the command board
# ============================================================================

# The grammar SGLang enforces (xgrammar): the model is physically unable to
# emit tokens outside this shape, so intercept() parsing is total in
# production. The lenient fallback below exists for degraded deployments
# (schema enforcement disabled) — fail-open means the agent still SPEAKS.
# UNIFORM ARITY keeps the xgrammar small and strictly bounded: every tool
# takes (target_id, secondary_id) and the router maps semantics per tool —
#   move_towards(target_id=entity|agent,   secondary_id=null)
#   pick_up     (target_id=entity,         secondary_id=null)
#   give        (target_id=entity,         secondary_id=recipient agent)
#   use         (target_id=item entity,    secondary_id=target entity)
# One object shape → one compiled grammar, regardless of tool count.
SPATIAL_TOOLS = ("move_towards", "pick_up", "give", "use")

ACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "say": {"type": "string", "maxLength": 300},
        "action": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "enum": list(SPATIAL_TOOLS)},
                        "target_id": {"type": "string", "maxLength": 32},
                        "secondary_id": {
                            "anyOf": [{"type": "string", "maxLength": 32},
                                      {"type": "null"}]},
                    },
                    "required": ["tool", "target_id", "secondary_id"],
                    "additionalProperties": False,
                },
            ]
        },
    },
    "required": ["say", "action"],
    "additionalProperties": False,
}


class ActionRouter:
    """The seam between language and physics. The dispatcher hands every
    structured dialogue generation to intercept(); speech goes back to the
    caller (→ TTS), actions become KinematicCommands via
    TickEngine.issue_command — an atomic dict assignment on the shared event
    loop, applied by the steering pass at the head of the next tick. Worst-
    case language-to-locomotion latency: one tick, 50 ms."""

    def __init__(self, command_ttl_s: float = 8.0) -> None:
        self._ttl = command_ttl_s
        self._engine: Any = None
        self._world: Any = None
        # Telemetry
        self.actions_routed = 0
        self.actions_rejected = 0
        self.parse_fallbacks = 0

    def bind(self, engine: Any, world: Any) -> None:
        self._engine = engine
        self._world = world

    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {"name": "panopticon_action", "strict": True,
                            "schema": ACTION_JSON_SCHEMA},
        }

    def intercept(self, agent_id: str, raw: str) -> str:
        """Parse one generation. Returns the speech (possibly the raw text on
        parse fallback — an agent that can't act can still talk)."""
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            self.parse_fallbacks += 1
            return raw.strip()
        try:
            obj = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            self.parse_fallbacks += 1
            return raw.strip()
        say = obj.get("say")
        say = say.strip() if isinstance(say, str) else ""
        action = obj.get("action")
        if isinstance(action, dict):
            self._route(agent_id, action)
        return say

    def _route(self, agent_id: str, action: dict[str, Any]) -> None:
        tool = action.get("tool")
        target_id = action.get("target_id")
        secondary = action.get("secondary_id")
        if (tool not in SPATIAL_TOOLS or not isinstance(target_id, str)
                or self._engine is None or self._world is None):
            self.actions_rejected += 1
            return

        if tool == "move_towards":
            if self._world.find(target_id) is None or target_id == agent_id:
                self.actions_rejected += 1     # hallucinated / self-chase
                return
            self._engine.issue_command(agent_id, KinematicCommand(
                tool="move_towards", target_id=target_id, ttl_s=self._ttl))
            self.actions_routed += 1
            return

        # Entity tools: EXISTENCE validated here (hallucinated ids die at the
        # router, cheaply); RANGE/OWNERSHIP/CONTENTION validated at the tick's
        # deterministic resolution pass, against the world as it IS then.
        registry = self._world.entities
        if registry.find(target_id) is None:
            self.actions_rejected += 1
            return
        if tool == "give" and (not isinstance(secondary, str)
                               or self._world.find(secondary) is None):
            self.actions_rejected += 1
            return
        if tool == "use" and (not isinstance(secondary, str)
                              or registry.find(secondary) is None):
            self.actions_rejected += 1
            return
        self._engine.submit_intent(EntityIntent(
            agent_id=agent_id, tool=tool, entity_id=target_id,
            secondary_id=secondary if isinstance(secondary, str) else None))
        self.actions_routed += 1
