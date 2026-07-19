"""Phase 2 verification suite (no LiveKit server / no TTS GPU required).

1. SentenceChunker: boundaries, abbreviations, decimals, quotes, flush().
2. RealTTSEngine streams PCM from a fake F5-TTS HTTP service; errors raise.
3. Full publisher pipeline with a fake rtc layer: tokens → sentences → TTS →
   correctly-sized 10 ms frames captured on the right agent's AudioSource.
4. THE contract: publish_tokens returns fast even when TTS is slow (the
   dispatcher / AIMD slot is never held hostage by audio synthesis).
5. Drop-oldest overflow on the bounded sentence queue.
"""
import asyncio
import logging
import struct
import time
import types

from aiohttp import web

import media

logging.disable(logging.ERROR)
SR = 24_000


# ---------------------------------------------------------------- fake rtc

class FakeAudioSource:
    def __init__(self, sample_rate, num_channels, queue_size_ms=1000, loop=None):
        self.sample_rate = sample_rate
        self.frames = []

    async def capture_frame(self, frame):
        self.frames.append(frame)
        await asyncio.sleep(0)  # yield like the real pacing point


class FakeAudioFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel, **kw):
        self.data = bytes(data)
        self.sample_rate = sample_rate
        self.samples_per_channel = samples_per_channel


class FakeTrack:
    @staticmethod
    def create_audio_track(name, source):
        t = types.SimpleNamespace(name=name, source=source)
        return t


class FakeParticipant:
    def __init__(self):
        self.published = []

    async def publish_track(self, track, options):
        self.published.append(track.name)


class FakeRoom:
    def __init__(self):
        self.local_participant = FakeParticipant()
        self.connected_to = None

    async def connect(self, url, token):
        self.connected_to = url

    async def disconnect(self):
        pass


def install_fake_rtc():
    fake = types.SimpleNamespace(
        AudioSource=FakeAudioSource,
        AudioFrame=FakeAudioFrame,
        LocalAudioTrack=FakeTrack,
        Room=FakeRoom,
        TrackPublishOptions=lambda **kw: None,
        TrackSource=types.SimpleNamespace(SOURCE_MICROPHONE=2),
    )
    media.rtc = fake
    media.api = types.SimpleNamespace(  # AccessToken builder chain
        AccessToken=lambda k, s: types.SimpleNamespace(
            with_identity=lambda self_i: types.SimpleNamespace(
                with_name=lambda n: types.SimpleNamespace(
                    with_grants=lambda g: types.SimpleNamespace(to_jwt=lambda: "jwt")))),
        VideoGrants=lambda **kw: None,
    )


class InstantTTS:
    sample_rate = SR

    async def synthesize(self, text, voice):
        # 100 ms of audio per sentence: 10 exact frames after slicing.
        yield b"\x01\x00" * (SR // 10)


class SlowTTS:
    sample_rate = SR

    def __init__(self, delay_s):
        self.delay_s = delay_s

    async def synthesize(self, text, voice):
        await asyncio.sleep(self.delay_s)
        yield b"\x01\x00" * (SR // 100)


# ------------------------------------------------------------------- tests

def test_chunker():
    c = media.SentenceChunker()
    out = []
    for tok in ["Hel", "lo the", "re! How are", " you? I am", " Dr. Ives, worth 3.", "50 dollars."]:
        out += c.feed(tok)
    # "3." mid-stream must NOT split (end-of-buffer != end-of-stream)
    assert out == ["Hello there!", "How are you?"], out
    assert c.flush() == "I am Dr. Ives, worth 3.50 dollars."

    c = media.SentenceChunker()
    out = c.feed('She said "Stop!" and left. Then silence')
    assert out == ['She said "Stop!"', "and left."], out
    assert c.flush() == "Then silence"
    c2 = media.SentenceChunker()
    assert c2.feed("I am 42. And proud. ") == ["I am 42.", "And proud."]
    assert c.flush() is None
    print("PASS chunker: boundaries, abbreviations, decimals, quotes, flush")


async def test_tts_client():
    async def handler(request):
        body = await request.json()
        assert body["format"] == "pcm_s16le"
        resp = web.StreamResponse()
        await resp.prepare(request)
        for _ in range(3):
            await resp.write(struct.pack("<480h", *([100] * 480)))
        return resp

    async def err_handler(request):
        return web.json_response({"error": "voice not found"}, status=422)

    app = web.Application()
    app.router.add_post("/synthesize", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18441).start()
    err_app = web.Application()
    err_app.router.add_post("/synthesize", err_handler)
    err_runner = web.AppRunner(err_app)
    await err_runner.setup()
    await web.TCPSite(err_runner, "127.0.0.1", 18442).start()

    tts = media.RealTTSEngine("http://127.0.0.1:18441", sample_rate=SR)
    total = b"".join([c async for c in tts.synthesize("Hello there!", "plaza_m1")])
    assert len(total) == 3 * 480 * 2, len(total)
    await tts.close()

    bad = media.RealTTSEngine("http://127.0.0.1:18442", sample_rate=SR)
    try:
        async for _ in bad.synthesize("x", "nope"):
            pass
        raise AssertionError("expected TTSRequestError")
    except media.TTSRequestError as exc:
        assert "422" in str(exc)
    await bad.close()
    await runner.cleanup()
    await err_runner.cleanup()
    print("PASS tts client: streamed PCM intact, 422 → TTSRequestError")


async def make_publisher(tts):
    pub = media.RealLiveKitPublisher("wss://fake", "key", "secret", tts,
                                     sentence_queue_size=3)
    await pub.connect()
    return pub


async def test_pipeline():
    pub = await make_publisher(InstantTTS())
    for tok in ["*nods*", " Busy plaza", " today.", " Very busy!"]:
        await pub.publish_tokens("agent_001", tok)
    await pub.publish_tokens("agent_002", "Indeed it is.")
    await pub.end_utterance("agent_001")   # as the dispatcher does
    await pub.end_utterance("agent_002")
    await asyncio.sleep(0.1)  # let speaker loops drain

    room = pub._room
    assert room.local_participant.published == ["agent-agent_001", "agent-agent_002"]
    ch1 = pub._channels["agent_001"]
    frame_samples = SR * 10 // 1000
    # 2 sentences × 100 ms audio = 20 frames of exactly 10 ms each
    assert len(ch1.source.frames) == 20, len(ch1.source.frames)
    assert all(f.samples_per_channel == frame_samples for f in ch1.source.frames)
    assert pub.sentences_spoken == 3
    await pub.close()
    print("PASS pipeline: per-agent tracks, 10ms frame slicing, sentence routing")


async def test_flush_via_end_utterance():
    pub = await make_publisher(InstantTTS())
    await pub.publish_tokens("agent_003", "I never finish my")
    await asyncio.sleep(0.05)
    assert pub.sentences_spoken == 0          # no boundary yet
    await pub.end_utterance("agent_003")      # ← dispatcher's flush call
    await asyncio.sleep(0.05)
    assert pub.sentences_spoken == 1
    await pub.close()
    print("PASS flush: unterminated tail spoken after end_utterance")


async def test_nonblocking_contract():
    pub = await make_publisher(SlowTTS(delay_s=1.0))
    await pub.publish_tokens("agent_004", "warmup.")   # channel exists now
    t0 = time.perf_counter()
    for _ in range(50):
        await pub.publish_tokens("agent_004", "More words arrive quickly. ")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50, f"publish_tokens blocked: {elapsed_ms:.1f}ms"
    await pub.close()
    print(f"PASS non-blocking: 50 sentence-producing calls in {elapsed_ms:.1f}ms "
          f"with a 1000ms-slow TTS (AIMD slot never hostage)")


async def test_drop_oldest():
    pub = await make_publisher(SlowTTS(delay_s=5.0))   # speaker never drains
    for i in range(10):
        await pub.publish_tokens("agent_005", f"Sentence number {i}. ")
    ch = pub._channels["agent_005"]
    queued = list(ch.sentences._queue)                 # type: ignore[attr-defined]
    assert len(queued) == 3                            # bounded
    assert queued[-1] == "Sentence number 9."          # newest kept
    assert pub.sentences_dropped > 0
    await pub.close()
    print(f"PASS drop-oldest: queue bounded at 3, newest speech kept, "
          f"dropped={pub.sentences_dropped}")


async def main():
    test_chunker()
    await test_tts_client()
    install_fake_rtc()
    await test_pipeline()
    await test_flush_via_end_utterance()
    await test_nonblocking_contract()
    await test_drop_oldest()
    print("\nALL PHASE 2 TESTS PASSED")


asyncio.run(main())
