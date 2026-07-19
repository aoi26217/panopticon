"""
Project Panopticon — Pre-Flight Silicon Profiler
=================================================
profile_silicon.py

Run on the host AFTER health checks clear and BEFORE the heartbeat ignites.
Maps the REAL Time-To-First-Token curve against concurrency on the actual
sglang deployment, locates the concurrency knee where p95 TTFT crosses the
500 ms budget, and prints the safe `aimd_max_limit` override.

Why this exists: the Phase 5 crucible certified AIMD against a SIMULATED
knee at A≈33. Real silicon has its own knee — model quantization, KV cache
fraction, TP topology, and PCIe layout all move it. AIMD will FIND the real
knee eventually (that is its job), but starting with `aimd_max_limit` far
above it wastes the first minutes of every boot rediscovering physics, and
starting far below it leaves throughput on the table. Measure once, clamp
correctly, let AIMD breathe inside a sane ceiling.

Methodology
-----------
  * Uses the PRODUCTION client (clients.RealSGLangClient) and the PRODUCTION
    shared prefix (tick_engine.SHARED_WORLD_PREFIX), so RadixAttention
    behaves exactly as it will in the plaza — a synthetic prompt would
    measure the wrong prefill.
  * WARMUP first: sequential requests bake the shared root into the radix
    tree; without this, level 1 pays the cold-tree cost and the whole curve
    reads pessimistic.
  * Per level A: two waves of A truly-concurrent streaming requests; TTFT =
    wall time to first delta. p50/p95 over both waves.
  * SAFETY ABORT: the sweep STOPS climbing when p95 exceeds abort_factor ×
    budget or any request errors — we are here to find the budget knee, not
    to hunt the OOM line on hardware we are about to depend on.
  * Recommendation: knee = highest level with p95 <= budget;
    aimd_max_limit = floor(knee × headroom) (default 0.9 — AIMD's additive
    probing supplies the last few slots itself).

Usage:
  python3 profile_silicon.py --sglang-url http://localhost:8000
  python3 profile_silicon.py --sglang-url http://sglang:8000 \\
      --levels 4,8,16,24,32,40,48 --budget-ms 500 --sim-knee 33
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from clients import RealSGLangClient
from tick_engine import EngineConfig, SHARED_WORLD_PREFIX

BAR = "─" * 72


def q(sorted_vals: list[float], frac: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1,
                           int(len(sorted_vals) * frac))]


async def one_request(client: RealSGLangClient, model: str,
                      i: int) -> tuple[float, bool]:
    """Returns (ttft_ms, ok). Leaf varies per request (as in production) so
    only the shared root hits the radix cache — the honest measurement."""
    messages = [
        {"role": "system", "content": SHARED_WORLD_PREFIX},
        {"role": "user",
         "content": f"agent_{i:03d} crossed your path at the fountain. React."},
    ]
    start = time.monotonic()
    try:
        async for _tok in client.stream_chat(model, messages):
            return (time.monotonic() - start) * 1000.0, True
        return (time.monotonic() - start) * 1000.0, False   # empty stream
    except Exception:
        return (time.monotonic() - start) * 1000.0, False


async def measure_level(client: RealSGLangClient, model: str, level: int,
                        waves: int) -> tuple[list[float], int]:
    ttfts: list[float] = []
    errors = 0
    for w in range(waves):
        results = await asyncio.gather(
            *(one_request(client, model, w * level + i)
              for i in range(level)))
        for ms, ok in results:
            if ok:
                ttfts.append(ms)
            else:
                errors += 1
    return sorted(ttfts), errors


async def profile(args: argparse.Namespace) -> int:
    cfg = EngineConfig()
    model = args.model or cfg.model_name
    client = RealSGLangClient(args.sglang_url, max_tokens=args.max_tokens)
    levels = [int(x) for x in args.levels.split(",")]

    print(BAR)
    print(f" PANOPTICON PRE-FLIGHT PROFILER — {args.sglang_url}")
    print(f" model={model}  budget={args.budget_ms:.0f}ms  "
          f"simulated knee A={args.sim_knee}")
    print(BAR)

    print(" warmup: baking the shared prefix into the radix tree ...")
    for i in range(args.warmup):
        await one_request(client, model, i)

    rows: list[dict] = []
    knee = 0
    aborted_at: int | None = None
    for level in levels:
        ttfts, errors = await measure_level(client, model, level, args.waves)
        p50, p95 = q(ttfts, 0.5), q(ttfts, 0.95)
        within = p95 <= args.budget_ms and errors == 0
        rows.append({"concurrency": level, "p50_ms": round(p50, 1),
                     "p95_ms": round(p95, 1), "errors": errors,
                     "within_budget": within})
        marker = "  ✓" if within else ("  ✗ ERRORS" if errors else "  ✗")
        print(f"  A={level:>3}   TTFT p50={p50:7.1f}ms  p95={p95:7.1f}ms  "
              f"err={errors}{marker}")
        if within:
            knee = level
        if errors or p95 > args.abort_factor * args.budget_ms:
            aborted_at = level
            print(f"  sweep aborted at A={level}: past the knee "
                  f"(p95 {p95:.0f}ms / errors {errors}) — the OOM line is "
                  f"not our business")
            break

    await client.close()
    print(BAR)

    if knee == 0:
        print(" VERDICT: budget unreachable at the lowest tested concurrency.")
        print(" This deployment cannot serve the plaza — check TP topology, ")
        print(" quantization, and --mem-fraction-static before proceeding.")
        return 2

    rec = max(args.min_limit, int(knee * args.headroom))
    delta = knee - args.sim_knee
    print(f" REAL KNEE:      A={knee}  (p95 crosses {args.budget_ms:.0f}ms "
          f"just beyond it)")
    print(f" SIMULATED KNEE: A={args.sim_knee}  →  real silicon is "
          f"{'+' if delta >= 0 else ''}{delta} "
          f"({'more' if delta >= 0 else 'LESS'} headroom than the crucible "
          f"assumed)")
    print(f" RECOMMENDED OVERRIDE:")
    print(f"   aimd_max_limit = {rec}"
          f"   (knee {knee} × headroom {args.headroom})")
    print(f"   apply: EngineConfig(aimd_max_limit={rec}) in the orchestrator")
    if delta < 0:
        print("   NOTE: the default ceiling (64) sits ABOVE this silicon's ")
        print("   knee — apply the override BEFORE the heartbeat ignites, or ")
        print("   AIMD will spend its first minutes rediscovering physics ")
        print("   through backoff storms.")
    if args.report:
        with open(args.report, "w") as fh:
            json.dump({"sglang_url": args.sglang_url, "model": model,
                       "budget_ms": args.budget_ms, "rows": rows,
                       "real_knee": knee, "sim_knee": args.sim_knee,
                       "recommended_aimd_max_limit": rec,
                       "aborted_at": aborted_at,
                       "measured_at": time.time()}, fh, indent=1)
        print(f" report → {args.report}")
    print(BAR)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Panopticon pre-flight profiler")
    p.add_argument("--sglang-url", required=True)
    p.add_argument("--model", default=None,
                   help="defaults to EngineConfig.model_name")
    p.add_argument("--budget-ms", type=float, default=500.0)
    p.add_argument("--levels", default="2,4,8,12,16,20,24,28,32,40,48,56,64")
    p.add_argument("--waves", type=int, default=2,
                   help="measurement waves per level")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--sim-knee", type=int, default=33,
                   help="the crucible's simulated knee, for comparison")
    p.add_argument("--headroom", type=float, default=0.9)
    p.add_argument("--min-limit", type=int, default=4)
    p.add_argument("--abort-factor", type=float, default=2.0,
                   help="stop sweeping when p95 exceeds this × budget")
    p.add_argument("--report", default=None, help="write JSON report here")
    sys.exit(asyncio.run(profile(p.parse_args())))


if __name__ == "__main__":
    main()
