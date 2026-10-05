"""Output-quality comparison: generate answers to a fixed prompt set, then score them
with the hosted model as a judge (1-5 rubric).

  python -m bench.quality generate --backend ollama --model llama3.2:3b
  python -m bench.quality generate --backend hosted
  python -m bench.quality judge "results/answers_*.jsonl"
"""
import argparse
import asyncio
import csv
import glob
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from backends import DEFAULT_MODELS, get_client

PROMPTS = Path("prompts.jsonl")
RESULTS = Path("results")

JUDGE_TEMPLATE = """You are grading an AI assistant's answer.

Question:
{prompt}

What a good answer must do:
{criteria}

Answer to grade:
{answer}

Score 1-5: 5 = fully correct and complete, 4 = minor issues, 3 = partly correct,
2 = mostly wrong or ignores instructions, 1 = wrong or empty.
Respond with ONLY a JSON object: {{"score": <int>, "reason": "<one sentence>"}}"""


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def load_prompts():
    lines = PROMPTS.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def generate(backend, model, max_tokens):
    client = get_client(backend)
    model = model or DEFAULT_MODELS[backend]
    out = RESULTS / f"answers_{backend}_{slug(model)}.jsonl"
    RESULTS.mkdir(exist_ok=True)
    prompts = load_prompts()
    with out.open("w", encoding="utf-8") as f:
        for i, p in enumerate(prompts, 1):
            t0 = time.perf_counter()
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": p["prompt"]}],
                max_tokens=max_tokens,
                temperature=0,
            )
            rec = {
                **p,
                "backend": backend,
                "model": model,
                "answer": resp.choices[0].message.content or "",
                "latency_s": round(time.perf_counter() - t0, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(prompts)}] {p['id']} done in {rec['latency_s']}s")
    print(f"Saved to {out}")


async def judge(files):
    if not files:
        raise SystemExit("No answer files matched. Run 'generate' first.")
    client = get_client("hosted")
    judge_model = DEFAULT_MODELS["hosted"]
    rows = []
    for path in files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            msg = JUDGE_TEMPLATE.format(
                prompt=rec["prompt"], criteria=rec["criteria"], answer=rec["answer"]
            )
            resp = await client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": msg}],
                max_tokens=200,
                temperature=0,
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", text, re.S)
            try:
                verdict = json.loads(m.group(0)) if m else {}
                score = int(verdict.get("score", 0))
            except (ValueError, json.JSONDecodeError):
                verdict, score = {}, 0
            rows.append({
                "model": f"{rec['backend']}:{rec['model']}",
                "id": rec["id"],
                "category": rec["category"],
                "score": score,
                "reason": verdict.get("reason", text[:200]),
            })
            print(f"{rows[-1]['model']} {rec['id']}: {score}")

    out = RESULTS / "quality.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r["score"])
    print("\nMean score (1-5):")
    for name, scores in by_model.items():
        print(f"  {name:<50} {sum(scores) / len(scores):.2f}  (n={len(scores)})")
    print(f"Details in {out}. Spot-check some judgments by hand: LLM judges have biases too.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--backend", required=True, choices=["ollama", "vllm", "hosted"])
    g.add_argument("--model")
    g.add_argument("--max-tokens", type=int, default=350)
    j = sub.add_parser("judge")
    j.add_argument("files", nargs="+", help="answer files or glob patterns")
    args = ap.parse_args()

    if args.cmd == "generate":
        asyncio.run(generate(args.backend, args.model, args.max_tokens))
    else:
        # Expand globs ourselves: Windows shells don't expand *
        files = sorted({f for pattern in args.files for f in glob.glob(pattern)})
        asyncio.run(judge(files))


if __name__ == "__main__":
    main()
