"""
Project Panopticon — Phase 2 Media Pipeline
============================================
media.py

Real-time audio egress: SGLang token streams → sentence chunks → F5-TTS →
per-agent LiveKit audio tracks.

Data flow (all stages decoupled by queues; the LLM dispatcher never waits):

  InferenceDispatcher ──publish_tokens()──►  SentenceChunker (per agent)
        │  returns immediately                     │ complete sentence
        ▼  AIMD slot freed                         ▼
   (next generation)                    bounded sentence queue (drop-oldest)
                                                   │
                                       _speaker_loop background task
                                                   │
                                        RealTTSEngine (F5-TTS HTTP, PCM)
                                                   │ 10 ms frames
                                        rtc.AudioSource.capture_frame
                                                   │ (realtime pacing here)
                                            LiveKit SFU → observers

╔══════════════════════════════════════════════════════════════════════════╗
║ VISEME ANTI-PATTERN ENFORCEMENT — DO NOT ADD LIP-SYNC LOGIC TO THIS FILE  ║
║                                                                            ║
║ The server ships AUDIO ONLY. Phoneme/viseme extraction is computed        ║
║ CLIENT-SIDE by a WASM model tapping the incoming WebRTC MediaStream       ║
║ (~10 ms cadence, <10 ms overhead). Server-side viseme generation wastes   ║
║ bandwidth and guarantees audio/animation desync under network jitter,     ║
║ because animation data and audio would ride different clocks. Any PR      ║
║ adding viseme, phoneme, or timing-metadata emission here is wrong by      ║
║ architecture, not by implementation.                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

import aiohttp
from livekit import api, rtc

log = logging.getLogger("panopticon.media")


# ============================================================================
# 1. Sentence Chunker
# ============================================================================

class SentenceChunker:
    """Buffers an incoming token stream and yields complete sentences.

    Flow-matching TTS (F5-TTS) is non-autoregressive: it denoises a whole
    utterance in parallel, so it needs complete clauses to plan prosody —
    feeding it token fragments produces robotic joins, and feeding it entire
    multi-sentence monologues delays Time-To-First-Audio. Sentence
    granularity is the latency/prosody sweet spot.

    Design notes:
      * Pure CPU, single compiled regex, no awaits — safe to call from the
        dispatcher's hot path.
      * A boundary is terminal punctuation (. ! ? … and CJK equivalents),
        optionally followed by closing quotes/brackets, then whitespace.
      * Common abbreviations (Mr., Dr., St., ...) and decimal points are
        guarded so "Dr. Ives owes 3.50 dollars." stays one sentence.
      * `flush()` drains any unterminated tail when the LLM stream ends —
        models sometimes stop without punctuation and that speech must not
        be swallowed.
    """

    _BOUNDARY = re.compile(
        r"""
        ([.!?…。！？]+["'”’)\]]*)               # terminal punct (+ closing quotes)
        (?=\s)                                 # a FOLLOWING char must exist:
        """,                                   # end-of-buffer ≠ end-of-stream
        re.VERBOSE,                            # (a token like "3." may just be
    )                                          # awaiting its "50"). Stream end
                                               # is signalled via flush() only.
    _WORD_BEFORE = re.compile(r"([A-Za-z]+|\d+)$")
    # Boundary veto list: "Dr. Ives" and "3. 50" style false splits.
    _ABBREV = frozenset({
        "mr", "ms", "mrs", "dr", "st", "jr", "sr", "vs",
        "prof", "etc", "ave", "blvd", "no", "vol",
    })
    _MIN_CHARS = 2  # ignore degenerate "fragments" like a bare "!"

    def __init__(self) -> None:
        self._buf: str = ""

    def _is_false_boundary(self, m: re.Match[str]) -> bool:
        if not m.group(1).startswith("."):
            return False
        wm = self._WORD_BEFORE.search(self._buf, 0, m.start())
        if wm is None:
            return False
        # Decimals ("3.50") never match at all: '.' must be followed by
        # whitespace/end. Only abbreviation words need the veto — sentences
        # may legitimately end in bare numbers ("I am 42.").
        return wm.group(1).lower() in self._ABBREV

    def feed(self, token: str) -> list[str]:
        """Append a token; return zero or more COMPLETE sentences."""
        self._buf += token
        out: list[str] = []
        search_from = 0
        while True:
            m = self._BOUNDARY.search(self._buf, search_from)
            if not m:
                break
            if self._is_false_boundary(m):
                search_from = m.end()          # skip past "Dr." / "3." etc.
                continue
            sentence = self._buf[: m.end()].strip()
            self._buf = self._buf[m.end():]
            search_from = 0
            if len(sentence) >= self._MIN_CHARS:
                out.append(sentence)
        return out

    def flush(self) -> str | None:
        """End of stream: return the unterminated remainder, if any."""
        tail = self._buf.strip()
        self._buf = ""
        return tail if len(tail) >= self._MIN_CHARS else None


# ============================================================================
# 2. TTS Engine (F5-TTS as an HTTP microservice)
# ============================================================================

class TTSEngine(Protocol):
    """Async text → raw PCM audio (s16le, mono, `sample_rate` Hz).

    F5-TTS runs as a SEPARATE HTTP microservice on the multimodal GPU (2-3 GB
    at FP16), keeping the orchestrator process free of torch/CUDA imports.
    Flow-matching generates faster than realtime, so streaming chunked PCM
    back over HTTP keeps Time-To-First-Audio low without websockets."""

    sample_rate: int

    def synthesize(self, text: str, voice: str) -> AsyncIterator[bytes]: ...


class TTSRequestError(RuntimeError):
    """Non-200 from the TTS service; the speaker loop logs and skips the
    sentence — a dropped line of ambient dialogue beats a stalled track."""


class RealTTSEngine:
    """aiohttp client for the F5-TTS microservice.

    Expected endpoint contract (implement in the F5-TTS wrapper service):
        POST {base_url}/synthesize
        body: {"text": str, "voice": str, "format": "pcm_s16le",
               "sample_rate": int}
        response: chunked stream of raw PCM s16le mono bytes.

    Cancellation mirrors RealSGLangClient: a hard resp.close() on
    CancelledError so an aborted utterance frees the TTS GPU immediately.
    """

    def __init__(
        self,
        base_url: str,
        *,
        sample_rate: int = 24_000,
        connect_timeout_s: float = 5.0,
        sock_read_timeout_s: float = 15.0,
    ) -> None:
        self._url = base_url.rstrip("/") + "/synthesize"
        self.sample_rate = sample_rate
        self._timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=connect_timeout_s, sock_read=sock_read_timeout_s
        )
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                connector=aiohttp.TCPConnector(limit=64),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def synthesize(self, text: str, voice: str) -> AsyncIterator[bytes]:
        session = await self._ensure_session()
        payload = {
            "text": text,
            "voice": voice,
            "format": "pcm_s16le",
            "sample_rate": self.sample_rate,
        }
        resp = await session.post(self._url, json=payload)
        try:
            if resp.status != 200:
                body = (await resp.text())[:300]
                raise TTSRequestError(f"TTS HTTP {resp.status}: {body}")
            async for chunk in resp.content.iter_chunked(4096):
                if chunk:
                    yield chunk
        except asyncio.CancelledError:
            resp.close()  # abort synthesis server-side, free the TTS GPU
            raise
        finally:
            if not resp.closed:
                resp.release()


# ============================================================================
# 3. LiveKit Publisher
# ============================================================================

# Deterministic per-agent voice assignment (F5-TTS reference voices).
_VOICES = ("plaza_m1", "plaza_m2", "plaza_f1", "plaza_f2", "plaza_n1", "plaza_n2")

_FRAME_MS = 10                      # LiveKit-idiomatic frame duration
_BYTES_PER_SAMPLE = 2               # PCM s16le mono


@dataclass
class _AgentChannel:
    """Everything owned by one speaking agent."""
    chunker: SentenceChunker
    sentences: asyncio.Queue[str]
    source: rtc.AudioSource
    track: rtc.LocalAudioTrack
    speaker_task: asyncio.Task
    voice: str
    dropped_sentences: int = 0


class RealLiveKitPublisher:
    """Implements tick_engine.MediaPublisher over a LiveKit SFU.

    Topology: ONE orchestrator participant publishing a POOL of audio tracks,
    one `rtc.AudioSource`/`LocalAudioTrack` per speaking agent, named
    `agent-{id}` so the WebGL/UE5 client can spatialize each stream by the
    agent's world position. (If per-track subscription control ever matters —
    e.g. observers only hearing nearby agents — split into one Room
    participant per agent; the SFU handles selective forwarding either way.)

    The non-blocking contract with the dispatcher:
      * publish_tokens() does regex work + put_nowait, then returns. The
        first call for a new agent awaits one-time track publication (~ms);
        every subsequent call is effectively synchronous.
      * end_utterance() flushes the chunker tail — models often stop
        without terminal punctuation.
      * All TTS + frame-pacing latency lives in per-agent _speaker_loop
        tasks. AudioSource.capture_frame() awaits when its internal jitter
        buffer (queue_size_ms) is full: THAT is the realtime pacing point,
        and it back-pressures only the speaker task.
      * Sentence queues are bounded with DROP-OLDEST overflow: ambient
        dialogue is ephemeral, and fresh speech beats a growing backlog of
        stale speech (graceful degradation, consistent with the tick loop).
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        api_secret: str,
        tts: TTSEngine,
        *,
        room_name: str = "panopticon",
        identity: str = "panopticon-orchestrator",
        sentence_queue_size: int = 8,
        max_tracks: int = 64,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._api_secret = api_secret
        self._tts = tts
        self._room_name = room_name
        self._identity = identity
        self._sentence_queue_size = sentence_queue_size
        self._max_tracks = max_tracks

        self._room: rtc.Room | None = None
        self._channels: dict[str, _AgentChannel] = {}
        self._channel_locks: dict[str, asyncio.Lock] = {}
        # Telemetry (tokens_published kept for parity with the Phase 0 stub)
        self.tokens_published = 0
        self.sentences_spoken = 0
        self.sentences_dropped = 0
        self.tts_failures = 0

    # ------------------------------------------------------------------ setup

    async def connect(self) -> None:
        token = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(self._identity)
            .with_name("Panopticon Orchestrator")
            .with_grants(api.VideoGrants(
                room_join=True, room=self._room_name,
                can_publish=True, can_subscribe=False,
            ))
            .to_jwt()
        )
        self._room = rtc.Room()
        await self._room.connect(self._url, token)
        log.info("LiveKit connected: room=%s identity=%s", self._room_name, self._identity)

    async def close(self) -> None:
        # Cancel speakers outright — pending ambient speech has no value at
        # shutdown, and a sentinel push would raise QueueFull on a saturated
        # queue. Speaker loops re-raise CancelledError cleanly.
        for channel in self._channels.values():
            channel.speaker_task.cancel()
        await asyncio.gather(
            *(c.speaker_task for c in self._channels.values()),
            return_exceptions=True,
        )
        if self._room is not None:
            await self._room.disconnect()
        if hasattr(self._tts, "close"):
            await self._tts.close()  # type: ignore[attr-defined]
        self._channels.clear()

    # -------------------------------------------------------- MediaPublisher

    async def publish_tokens(self, agent_id: str, text: str) -> None:
        """Hot path — called per streamed LLM token by the dispatcher."""
        self.tokens_published += 1
        channel = self._channels.get(agent_id)
        if channel is None:
            channel = await self._create_channel(agent_id)
            if channel is None:
                return  # track pool exhausted; agent is silently text-only
        for sentence in channel.chunker.feed(text):
            self._enqueue_sentence(channel, sentence)

    async def end_utterance(self, agent_id: str) -> None:
        """Generation finished: flush any unterminated tail."""
        channel = self._channels.get(agent_id)
        if channel is None:
            return
        tail = channel.chunker.flush()
        if tail:
            self._enqueue_sentence(channel, tail)

    # ---------------------------------------------------------------- internal

    def _enqueue_sentence(self, channel: _AgentChannel, sentence: str) -> None:
        try:
            channel.sentences.put_nowait(sentence)
        except asyncio.QueueFull:
            # Drop-OLDEST: discard the stalest queued sentence, keep the new
            # one. The agent skips a line rather than falling behind reality.
            try:
                channel.sentences.get_nowait()
                channel.dropped_sentences += 1
                self.sentences_dropped += 1
            except asyncio.QueueEmpty:
                pass
            channel.sentences.put_nowait(sentence)

    async def _create_channel(self, agent_id: str) -> _AgentChannel | None:
        assert self._room is not None, "publisher used before connect()"
        # Per-agent lock: two concurrent generations for the same agent must
        # not race to publish two tracks.
        lock = self._channel_locks.setdefault(agent_id, asyncio.Lock())
        async with lock:
            if agent_id in self._channels:                # lost the race: fine
                return self._channels[agent_id]
            if len(self._channels) >= self._max_tracks:
                log.warning("Track pool exhausted (%d) — %s gets no audio",
                            self._max_tracks, agent_id)
                return None

            source = rtc.AudioSource(self._tts.sample_rate, num_channels=1)
            track = rtc.LocalAudioTrack.create_audio_track(f"agent-{agent_id}", source)
            await self._room.local_participant.publish_track(
                track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )

            channel = _AgentChannel(
                chunker=SentenceChunker(),
                sentences=asyncio.Queue(maxsize=self._sentence_queue_size),
                source=source,
                track=track,
                speaker_task=None,  # type: ignore[arg-type]  # set just below
                voice=_VOICES[hash(agent_id) % len(_VOICES)],
            )
            channel.speaker_task = asyncio.create_task(
                self._speaker_loop(agent_id, channel), name=f"speaker-{agent_id}"
            )
            self._channels[agent_id] = channel
            log.info("Published audio track for %s (voice=%s, pool=%d/%d)",
                     agent_id, channel.voice, len(self._channels), self._max_tracks)
            return channel

    async def _speaker_loop(self, agent_id: str, channel: _AgentChannel) -> None:
        """One long-lived task per agent: sentence → TTS → 10 ms PCM frames →
        AudioSource. Everything slow happens here, far from the dispatcher."""
        frame_samples = self._tts.sample_rate * _FRAME_MS // 1000
        frame_bytes = frame_samples * _BYTES_PER_SAMPLE

        while True:
            sentence = await channel.sentences.get()
            try:
                remainder = b""
                async for chunk in self._tts.synthesize(sentence, channel.voice):
                    data = remainder + chunk
                    offset = 0
                    while len(data) - offset >= frame_bytes:
                        frame = rtc.AudioFrame(
                            data=data[offset:offset + frame_bytes],
                            sample_rate=self._tts.sample_rate,
                            num_channels=1,
                            samples_per_channel=frame_samples,
                        )
                        # Realtime pacing: awaits iff the source's internal
                        # buffer (queue_size_ms) is full. Back-pressures THIS
                        # task only — TTS reads pause, dispatcher unaffected.
                        await channel.source.capture_frame(frame)
                        offset += frame_bytes
                    remainder = data[offset:]
                if remainder:                              # zero-pad final frame
                    frame = rtc.AudioFrame(
                        data=remainder.ljust(frame_bytes, b"\x00"),
                        sample_rate=self._tts.sample_rate,
                        num_channels=1,
                        samples_per_channel=frame_samples,
                    )
                    await channel.source.capture_frame(frame)
                self.sentences_spoken += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self.tts_failures += 1
                log.exception("TTS/publish failed for %s — sentence skipped: %.60r",
                              agent_id, sentence)
