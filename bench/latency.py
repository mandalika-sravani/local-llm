"""Latency / throughput / memory benchmark for any OpenAI-compatible backend.

Examples:
  python -m bench.latency --backend ollama --model llama3.2:3b --label q4_K_M
  python -m bench.latency --backend ollama --model llama3.2:3b-instruct-q8_0 --label q8_0
  python -m bench.latency --backend hosted --concurrency 1,4,8
Results are appended to results/latency.csv.
"""
import argparse
import asyncio
import csv
import time
from datetime import datetime
from pathlib import Path

from openai import BadRequestError

from backends import DEFAULT_MODELS, get_client
from bench.memory import MemorySampler

PROMPT = (
    "Explain how a hash map works, including how collisions are handled. "
    "Answer in about 150 words."
)
CSV_PATH = Path("results/latency.csv")


def pct(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


async def one_request(client, model, max_tokens, use_usage):
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
    )
    if use_usage:
        kwargs["stream_options"] = {"include_usage": True}

    t0 = time.perf_counter()
    ttft, chunks, out_tokens = None, 0, None
    stream = await client.chat.completions.create(**kwargs)
    async for ev in stream:
        if getattr(ev, "usage", None):
            out_tokens = ev.usage.completion_tokens
        if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks += 1
    total = time.perf_counter() - t0
    ttft = ttft if ttft is not None else total
    out_tokens = out_tokens or chunks  # fall back to chunk count if usage isn't reported
    gen_time = total - ttft
    decode_tps = (out_tokens - 1) / gen_time if gen_time > 0 and out_tokens > 1 else 0.0
    return {"ttft": ttft, "total": total, "out_tokens": out_tokens, "decode_tps": decode_tps}


async def run_level(client, model, concurrency, runs, max_tokens, use_usage):
    sem = asyncio.Semaphore(concurrency)

    async def guarded():
        async with sem:
            return await one_request(client, model, max_tokens, use_usage)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[guarded() for _ in range(runs)])
    return results, time.perf_counter() - t0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["ollama", "vllm", "hosted"])
    ap.add_argument("--model", help="defaults per backend, see backends.py")
    ap.add_argument("--label", default="", help="free text, e.g. quant level: q4_K_M, fp16, awq")
    ap.add_argument("--concurrency", default="1,2", help="comma-separated levels, e.g. 1,2,4,8")
    ap.add_argument("--runs", type=int, default=6, help="requests per level (CPU: keep small)")
    ap.add_argument("--max-tokens", type=int, default=150)
    args = ap.parse_args()

    model = args.model or DEFAULT_MODELS[args.backend]
    client = get_client(args.backend)

    # Warm-up: loads the model into memory so load time doesn't pollute TTFT.
    # Also detects whether the backend supports stream_options.include_usage.
    print(f"Warming up {args.backend}:{model} (first load can take a minute on CPU)...")
    use_usage = True
    try:
        await one_request(client, model, 16, use_usage)
    except BadRequestError:
        use_usage = False
        await one_request(client, model, 16, use_usage)

    CSV_PATH.parent.mkdir(exist_ok=True)
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = None
        for c in [int(x) for x in args.concurrency.split(",")]:
            runs = max(args.runs, c)
            async with MemorySampler(args.backend) as mem:
                results, wall = await run_level(client, model, c, runs, args.max_tokens, use_usage)
            ttfts = [r["ttft"] for r in results]
            totals = [r["total"] for r in results]
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "backend": args.backend,
                "model": model,
                "label": args.label,
                "concurrency": c,
                "runs": runs,
                "ttft_p50_s": round(pct(ttfts, 50), 3),
                "ttft_p95_s": round(pct(ttfts, 95), 3),
                "e2e_p50_s": round(pct(totals, 50), 3),
                "e2e_p95_s": round(pct(totals, 95), 3),
                "decode_tps_per_req": round(sum(r["decode_tps"] for r in results) / len(results), 2),
                "throughput_tps_total": round(sum(r["out_tokens"] for r in results) / wall, 2),
                **mem.summary(),
            }
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if new_file:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(
                f"c={c:<3} TTFT p50 {row['ttft_p50_s']}s | e2e p50 {row['e2e_p50_s']}s "
                f"p95 {row['e2e_p95_s']}s | {row['decode_tps_per_req']} tok/s per req | "
                f"{row['throughput_tps_total']} tok/s total | RAM {row['peak_sys_ram_gb']} GB"
            )
    print(f"Saved to {CSV_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
