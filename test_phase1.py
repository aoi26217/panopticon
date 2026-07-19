"""Phase 1 verification suite (no GPU / no Neo4j server required).

1. RealSGLangClient streams tokens from a fake SGLang SSE server; TTFT is
   measured on the first delta.
2. Cancelling the consuming task hard-closes the HTTP connection and the
   SERVER observes the disconnect (the mechanism SGLang uses to abort the
   request and free the KV slot).
3. Non-200 responses raise SGLangRequestError (→ AIMD record_failure path).
4. RealNeo4jGraphClient.group_rows maps mixed batches to per-type UNWIND rows.
5. An unreachable Neo4j endpoint exhausts retries and dead-letters the batch
   as replayable JSONL.
"""
import asyncio
import json
import logging
import time
from pathlib import Path

from aiohttp import web

import clients

logging.disable(logging.ERROR)

SERVER_STATE = {"aborted": False, "chunks_sent": 0}


def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


async def fake_sglang_handler(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    mode = body["messages"][-1]["content"]

    try:
        if mode == "HANG_AFTER_ONE":
            await resp.write(sse({"choices": [{"delta": {"content": "first"}}]}))
            SERVER_STATE["chunks_sent"] += 1
            # Wedged generation: keep trying to write; a hard client close
            # surfaces here as ConnectionResetError → SGLang's abort path.
            for _ in range(200):
                await asyncio.sleep(0.05)
                await resp.write(b": keep-alive\n\n")
        else:
            for tok in ["Hello", " from", " fake", " SGLang"]:
                await resp.write(sse({"choices": [{"delta": {"content": tok}}]}))
                SERVER_STATE["chunks_sent"] += 1
                await asyncio.sleep(0.01)
            await resp.write(b"data: [DONE]\n\n")
    except (ConnectionResetError, ConnectionError, asyncio.CancelledError):
        SERVER_STATE["aborted"] = True
        raise
    return resp


async def fake_error_handler(request: web.Request) -> web.Response:
    return web.json_response({"error": "model not loaded"}, status=503)


async def main() -> None:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", fake_sglang_handler)
    err_app = web.Application()
    err_app.router.add_post("/v1/chat/completions", fake_error_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 18321)
    await site.start()
    err_runner = web.AppRunner(err_app)
    await err_runner.setup()
    await web.TCPSite(err_runner, "127.0.0.1", 18322).start()

    client = clients.RealSGLangClient("http://127.0.0.1:18321")

    # ---- Test 1: happy-path streaming + TTFT ------------------------------
    tokens, start, ttft_ms = [], time.monotonic(), None
    async for tok in client.stream_chat("test-model", [{"role": "user", "content": "hi"}]):
        if ttft_ms is None:
            ttft_ms = (time.monotonic() - start) * 1000
        tokens.append(tok)
    assert tokens == ["Hello", " from", " fake", " SGLang"], tokens
    assert ttft_ms is not None and ttft_ms < 200
    print(f"PASS stream: {len(tokens)} tokens, ttft={ttft_ms:.1f}ms")

    # ---- Test 2: cancellation → server-side abort -------------------------
    async def consume_hanging() -> None:
        async for _ in client.stream_chat(
            "test-model", [{"role": "user", "content": "HANG_AFTER_ONE"}]
        ):
            pass

    task = asyncio.create_task(consume_hanging())
    await asyncio.sleep(0.3)          # let the first chunk arrive, then wedge
    task.cancel()                     # ← the ZombieReaper's move
    try:
        await task
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.3)          # give the server a write cycle to notice
    assert SERVER_STATE["aborted"], "server never observed the disconnect!"
    print("PASS cancel: hard close propagated — server aborted the generation")

    # ---- Test 3: HTTP error → SGLangRequestError --------------------------
    err_client = clients.RealSGLangClient("http://127.0.0.1:18322")
    try:
        async for _ in err_client.stream_chat("m", [{"role": "user", "content": "x"}]):
            pass
        raise AssertionError("expected SGLangRequestError")
    except clients.SGLangRequestError as exc:
        assert "503" in str(exc)
        print("PASS error: 503 surfaced as SGLangRequestError (AIMD failure path)")

    # ---- Test 4: batch grouping for UNWIND --------------------------------
    grouped = clients.RealNeo4jGraphClient.group_rows([
        {"type": "observed", "agent_id": "a1", "location": "plaza", "t": 1.0, "summary": "s1"},
        {"type": "observed", "agent_id": "a2", "location": "plaza", "t": 2.0, "summary": "s2"},
        {"type": "bogus_type", "agent_id": "a3"},
    ])
    assert set(grouped) == {"observed"} and len(grouped["observed"]) == 2
    assert "type" not in grouped["observed"][0]
    print("PASS grouping: 2 rows bucketed to UNWIND, unknown type skipped")

    # ---- Test 5: unreachable Neo4j → retries → dead-letter ----------------
    dlq = Path("/tmp/test_dlq.jsonl")
    dlq.unlink(missing_ok=True)
    graph = clients.RealNeo4jGraphClient(
        "bolt://127.0.0.1:1",         # nothing listening
        "neo4j", "pw",
        max_retries=2, backoff_base_s=0.01, connection_timeout_s=0.5, driver_retry_window_s=0.5,
        dead_letter_path=dlq,
    )
    t0 = time.monotonic()
    await graph.execute_batch(
        [{"type": "observed", "agent_id": "a1", "location": "plaza", "t": 1.0, "summary": "s"}]
    )
    record = json.loads(dlq.read_text().strip())
    assert record["mutations"][0]["agent_id"] == "a1"
    assert graph.dead_lettered == 1
    print(f"PASS dead-letter: batch persisted after retries ({time.monotonic()-t0:.1f}s), "
          f"replayable JSONL verified")
    await graph.close()

    await client.close()
    await err_client.close()
    await runner.cleanup()
    await err_runner.cleanup()
    print("\nALL PHASE 1 TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
