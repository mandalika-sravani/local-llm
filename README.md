# local-llm

Run open-weight LLMs locally, expose them through an API, and benchmark latency, memory and
output quality against a hosted model (Claude Haiku 4.5).

**Part A** runs on a CPU-only Windows laptop with Ollama. **Part B** runs vLLM on a free
Google Colab T4 GPU to test GPU serving and continuous batching.

## Key results

- **Generation speed is limited by memory bandwidth on both CPU and GPU.** Tokens/s ≈ bandwidth ÷ model size:
  about 16 to 17 GB/s on the laptop CPU and about 160 to 180 GB/s on the T4. Smaller weights mean faster generation.
- **Continuous batching works.** vLLM's total throughput rose from 27 to 451 tok/s (FP16) and from 80 to
  864 tok/s (AWQ) as concurrency went from 1 to 16, with time to first token staying under 0.13 s.
  Ollama on the laptop queued requests one at a time: TTFT went from 0.4 s to 21 s at just 2 users.
- **4-bit quantization roughly triples single-user speed** on the GPU (AWQ 81 vs FP16 28 tok/s) and
  nearly halves Q8's cost on CPU, with small quality losses that 20 prompts can't measure precisely.
- Memory estimates from parameter count plus KV cache were within about **5%** of measured values.
- The hosted model (Claude Haiku 4.5) scored highest on quality (4.95/5). On a free T4, the AWQ model
  matched its per-request latency (2.3 s for 200 tokens), but scored 4.05.
- The laptop's small Radeon GPU made Ollama produce **garbage output at about 1 tok/s**. Forcing CPU-only fixed both.

## Hardware

| Part | Machine | Notes |
|---|---|---|
| A | Intel i7-8550U (4 cores / 8 threads), 8 GB RAM (7.87 GB usable), Windows | Radeon 520 (2 GB) disabled for inference, see Findings |
| B | Google Colab, NVIDIA T4 (16 GB VRAM) | FP16 (T4 has no BF16 support) |
| Hosted | Claude Haiku 4.5 via Anthropic's OpenAI-compatible endpoint | Also used as the quality judge |

## Architecture

```
               ┌──────────────── FastAPI gateway (POST /generate) ────────────────┐
client ──────► │  backend = ollama | vllm | hosted                                │
               └──────┬───────────────────────┬──────────────────────┬─────────────┘
                      ▼                       ▼                      ▼
             Ollama (CPU, laptop)     vLLM (T4, Colab)      Claude API (hosted)
             localhost:11434/v1       localhost:8000/v1     api.anthropic.com/v1/

bench/latency.py  ─ streams requests, measures TTFT / tok/s / concurrency / peak memory
bench/quality.py  ─ runs 20 fixed prompts, scores answers 1-5 with an LLM judge
```

All backends use the OpenAI chat-completions format, so one client and one set of
benchmark scripts covers every backend.

## Repository layout

```
backends.py          shared backend config (URLs, keys, default models)
gateway/main.py      FastAPI gateway: /health, /generate
bench/latency.py     latency, throughput and memory benchmark  → results/latency.csv
bench/memory.py      background sampler: system RAM, Ollama /api/ps, NVIDIA GPU memory
bench/quality.py     answer generation + LLM-as-judge scoring    → results/quality.csv
prompts.jsonl        20 evaluation prompts with grading criteria (7 categories)
Modelfile*           CPU-only Ollama model definitions (num_gpu 0)
colab_vllm.md        Part B notebook cells
results/             all raw results and model answers
```

## Setup (Windows, Part A)

```
# 1. Ollama: install from https://ollama.com/download, then
ollama pull llama3.2:3b
ollama pull llama3.2:3b-instruct-q8_0
ollama pull qwen2.5:1.5b

# 2. CPU-only copies (see "GPU problem" in Findings)
ollama create llama3.2-cpu    -f Modelfile
ollama create llama3.2-q8-cpu -f Modelfile.q8
ollama create qwen2.5-cpu     -f Modelfile.qwen

# 3. Python
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # add HOSTED_API_KEY (and HOSTED_WORKSPACE_ID if your key needs it)
```

Each `Modelfile` is two lines, for example:

```
FROM llama3.2:3b
PARAMETER num_gpu 0
```

## How to run

```
# API gateway → http://127.0.0.1:8001/docs
uvicorn gateway.main:app --reload --port 8001

# Latency + memory (run "ollama stop <model>" between runs so only one model is loaded)
python -m bench.latency --backend ollama --model llama3.2-cpu    --label q4_K_M
python -m bench.latency --backend ollama --model llama3.2-q8-cpu --label q8_0
python -m bench.latency --backend ollama --model qwen2.5-cpu     --label q4_K_M
python -m bench.latency --backend hosted --concurrency 1,2

# Quality
python -m bench.quality generate --backend ollama --model llama3.2-cpu --max-tokens 250
python -m bench.quality generate --backend ollama --model llama3.2-q8-cpu --max-tokens 250
python -m bench.quality generate --backend ollama --model qwen2.5-cpu --max-tokens 250
python -m bench.quality generate --backend hosted --max-tokens 250
python -m bench.quality judge "results/answers_*.jsonl"
```

Part B: see [colab_vllm.md](colab_vllm.md).

## Methodology

- **Latency:** streaming requests with a fixed prompt (about 150-word answer, `max_tokens=150`,
  `temperature=0`). One warm-up request first, so model load time isn't counted. 6 requests per
  concurrency level. TTFT is time to first token; decode tok/s excludes TTFT; throughput is total
  output tokens ÷ wall-clock time across all concurrent requests.
- **Memory:** sampled every 0.5 s. Ollama's `/api/ps` reports model size (weights + allocated KV
  cache); NVML reports GPU memory on Colab.
- **Quality:** 20 prompts across reasoning, math, coding, summarization, instruction following,
  knowledge and writing, each with written grading criteria. Every model was capped at
  `max_tokens=250`. Claude Haiku 4.5 scored each answer 1-5, and I spot-checked judgments by hand.

## Memory estimates vs measured

Weights ≈ parameters × bytes per parameter.
KV cache per token = 2 (K and V) × layers × KV heads × head_dim × 2 bytes (FP16).

| Model | Layers / KV heads / head_dim | KV per token | KV at 4096 ctx | Weights | Estimate | **Measured** |
|---|---|---|---|---|---|---|
| Llama 3.2 3B Q4_K_M | 28 / 8 / 128 | ~112 KB | ~0.45 GB | ~2.0 GB | ~2.45 GB | **2.39 GB** |
| Llama 3.2 3B Q8_0 | 28 / 8 / 128 | ~112 KB | ~0.45 GB | ~3.4 GB | ~3.85 GB | **3.69 GB** |
| Qwen 2.5 1.5B Q4_K_M | 28 / 2 / 128 | ~28 KB | ~0.11 GB | ~1.0 GB | ~1.1 GB | **1.09 GB** |
| Llama 3.2 3B FP16 | | | | ~6.4 GB | ~6.9 GB | not run: doesn't fit in 8 GB RAM with Windows |

## Results: Part A (CPU laptop) vs hosted

| Model | Backend | Quant | Memory | TTFT p50 | Decode tok/s | Throughput c=2 (tok/s) | e2e p50 (~150 tok) | Quality (/5) |
|---|---|---|---|---|---|---|---|---|
| Llama 3.2 3B | Ollama CPU | Q4_K_M | 2.39 GB | 0.43 s | 7.05 | 7.17 | 20.9 s | 4.45 |
| Llama 3.2 3B | Ollama CPU | Q8_0 | 3.69 GB | 0.81 s | 4.46 | 4.45 | 33.8 s | 4.75 |
| Qwen 2.5 1.5B | Ollama CPU | Q4_K_M | 1.09 GB | 0.36 s | 14.8 | 13.9 | 10.4 s | 4.20 |
| Claude Haiku 4.5 | Hosted API | n/a | n/a | 0.89 s | 108.5 | 126.7 | 2.2 s | 4.95 |

Raw data: [results/latency.csv](results/latency.csv), [results/quality.csv](results/quality.csv).

### Quality by category (mean score out of 5)

| Category (n prompts) | Haiku 4.5 | Llama 3B Q8 | Llama 3B Q4 | Qwen 3B FP16 | Qwen 1.5B Q4 | Qwen 3B AWQ |
|---|---|---|---|---|---|---|
| Reasoning (4) | 5.00 | 5.00 | 5.00 | 4.25 | 5.00 | 4.25 |
| Math (2) | 5.00 | 5.00 | 3.50 | 5.00 | 5.00 | 5.00 |
| Coding (4) | 5.00 | 5.00 | 4.75 | 5.00 | 4.50 | 4.00 |
| Instruction following (4) | 5.00 | 5.00 | 5.00 | 4.75 | 4.00 | 4.75 |
| Summarization (2) | 5.00 | 5.00 | 5.00 | 5.00 | 4.00 | 4.50 |
| Knowledge (3) | 5.00 | 3.33 | 2.67 | 2.67 | 2.67 | 2.67 |
| Writing (1) | 4.00 | 5.00 | 5.00 | 2.00 | 4.00 | 2.00 |
| **Overall (20)** | **4.95** | **4.75** | **4.45** | **4.30** | **4.20** | **4.05** |

## Results: Part B (vLLM on Colab T4)

Qwen2.5-3B-Instruct, 200-token outputs, 16 requests per concurrency level.

| Model | Quant | Weights | TTFT p50 c=1 | Decode tok/s c=1 | e2e p50 c=1 | Throughput c=1 / 4 / 8 / 16 (tok/s) | TTFT p50 c=16 | Quality (/5) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | FP16 | ~6.2 GB (est.) | 0.046 s | 28.4 | 7.2 s | 27 / 132 / 248 / 451 | 0.121 s | 4.30 |
| Qwen2.5-3B-Instruct-AWQ | AWQ 4-bit | ~2 GB (est.) | 0.036 s | 80.9 | 2.3 s | 80 / 293 / 540 / 864 | 0.116 s | 4.05 |

_Replace the weight estimates with the "weights took" values from `vllm_fp16.log` / `vllm_awq.log`._
`peak_gpu_gb` in the CSV is about 12.5 GB for both models because vLLM reserves
`--gpu-memory-utilization 0.85` of VRAM up front and fills the space not used by weights with
KV cache blocks (PagedAttention). Smaller AWQ weights therefore leave more room for KV cache,
meaning more concurrent sequences.

The vLLM latency runs were repeated in a second Colab session, and results agreed within about 2 to 4%
(e.g. FP16 throughput at c=16: 445 vs 451 tok/s; AWQ: 876 vs 864 tok/s).

### All models side by side

| Model | Where | Quant | Decode tok/s (1 user) | Best total throughput | Quality (/5) |
|---|---|---|---|---|---|
| Claude Haiku 4.5 | Hosted API | n/a | 108.5 | 127 tok/s at c=2 (not tested higher) | 4.95 |
| Llama 3.2 3B | Laptop CPU | Q8_0 | 4.5 | 4.5 tok/s | 4.75 |
| Llama 3.2 3B | Laptop CPU | Q4_K_M | 7.1 | 7.2 tok/s | 4.45 |
| Qwen2.5 3B | T4 GPU, vLLM | FP16 | 28.4 | 451 tok/s at c=16 | 4.30 |
| Qwen2.5 1.5B | Laptop CPU | Q4_K_M | 14.8 | 13.9 tok/s | 4.20 |
| Qwen2.5 3B | T4 GPU, vLLM | AWQ 4-bit | 80.9 | 864 tok/s at c=16 | 4.05 |

## Findings

**1. Generation speed is limited by memory bandwidth, on CPU and GPU.** To generate each token, the model reads all
its weights from RAM once, so tokens/s ≈ RAM bandwidth ÷ model size. All three local models fit
this closely:

| Model | Size × decode speed |
|---|---|
| Llama 3B Q4 | 2.39 GB × 7.05 tok/s = 16.8 GB/s |
| Llama 3B Q8 | 3.69 GB × 4.46 tok/s = 16.5 GB/s |
| Qwen 1.5B Q4 | 1.09 GB × 14.8 tok/s = 16.1 GB/s |

Q8 is 1.54x larger than Q4 and was 1.58x slower. On this hardware, a smaller model (fewer
parameters or fewer bits) is directly a faster model.

The same holds on the GPU. Qwen 3B FP16 (about 6.2 GB) × 28.4 tok/s ≈ 176 GB/s, and AWQ (about 2 GB)
× 80.9 tok/s ≈ 162 GB/s: roughly half the T4's ~320 GB/s theoretical bandwidth. AWQ's weights are
about 3x smaller, and it generated about 2.9x faster. The GPU's ~10x higher bandwidth is why a 3B FP16 model
on the T4 ran 4x faster than a 3B Q4 model on the laptop.

**2. Memory can be predicted from the architecture.** Every estimate was within about 5% of
measured. Grouped-query attention matters: Qwen 1.5B has only 2 KV heads, so its KV cache per
token is 4x smaller than Llama 3B's (28 KB vs 112 KB). The FP16 3B model (about 6.9 GB) was ruled
out by calculation before running anything, since peak system RAM was already 7.2 to 7.5 GB with Q4/Q8.

**3. The small GPU made inference worse, not better.** By default Ollama offloaded part of
Llama 3B to the Radeon 520 (2 GB). The result was unreadable output at about 1 tok/s. Creating
CPU-only models with `PARAMETER num_gpu 0` gave correct output at 7 tok/s. Checking which
processor actually runs the model (`ollama ps`) is part of hardware profiling.

**4. Continuous batching: vLLM scaled with concurrency, Ollama queued requests one at a time.**
On the laptop at concurrency 2, Ollama's TTFT rose from 0.43 s to 21 s (one full request's duration)
while throughput stayed flat at about 7 tok/s: the second request waited in a queue. On the T4, vLLM's total
throughput grew almost linearly from 1 to 16 users (FP16: 27 → 451 tok/s, 16.7x; AWQ: 80 → 864
tok/s, 10.8x), while TTFT stayed below 0.13 s. Because decoding is memory-bound, one pass over the
weights can produce the next token for all 16 sequences at once, so extra users are nearly free.
Per-request speed held steady for FP16 (28 to 33 tok/s) but fell 30% for AWQ (81 → 56 tok/s) at 16 users.
With smaller weights, other per-step costs (attention over 16 KV caches, dequantizing 4-bit weights)
become a larger share of each step. My interpretation is that AWQ starts moving from
memory-bound toward compute-bound sooner, but I did not measure this directly.

**5. Local wins on time to first token, hosted wins on everything else.** Local TTFT (0.36 to 0.43 s
on the laptop, 0.04 s on the T4) beat Claude (0.89 s) because there is no network round trip, but Claude generated about 15x
faster, so a 150-token answer took 2.2 s hosted against 21 s locally. Output length dominated
local latency: Qwen's per-prompt times ranged from 0.3 s (one-word answer) to 28.6 s (long
explanation) at the same generation speed.

**6. Quality: small models do well on reasoning but lack factual knowledge.** The local and Colab models
scored 4.25 to 5.0 on reasoning, but only 2.67 to 3.33 on knowledge, against Haiku's 5.0. Every one of the
five open-weight models described the KV cache as a general-purpose cache instead of stored attention
keys and values, regardless of size (1.5B or 3B) or precision (4-bit to FP16). More bits didn't add
missing knowledge.

**7. Quantization and quality: small differences, noisy measurement.** Q8 scored 4.75 against Q4's 4.45,
and FP16 scored 4.30 against AWQ's 4.05. In both cases the higher-precision version was slightly better, and
in both cases the gap came from only 3 or 4 of 20 prompts. AWQ's biggest loss was the Python list question,
where it answered `[1, 2, 3]` instead of `[1, 2, 3, 4]` after correctly explaining that both names refer to
the same list. The direction is consistent, but 20 prompts cannot tell a 0.25-point difference from noise.
For the cost: Q8 used 54% more memory and was 37% slower than Q4; FP16 was about 3x larger and 2.9x slower than AWQ.

**8. Model family matters as much as size.** Qwen 3B FP16 (4.30) scored below Llama 3B Q4 (4.45)
and only 0.1 above Qwen 1.5B (4.20). Both Qwen 3B versions answered "No" to the syllogism (r2) while
their own explanation showed the answer was "Yes", and both wrote a full email template instead of the
requested two sentences (w1), failures the smaller Qwen 1.5B did not make. Parameter count alone did
not predict quality on this prompt set.

**9. Results were reproducible.** Repeating the vLLM benchmarks in a new Colab session gave throughput within
about 2 to 4%, and two Qwen 1.5B CPU runs gave 14.7 and 14.9 tok/s.

**10. One unexplained result.** FP16 at concurrency 1 was slower per request (28.4 tok/s, p95 latency
9.5 s vs p50 7.2 s) than at concurrency 4 (33.3 tok/s). This happened in both sessions, so it is not
warm-up. AWQ did not show it. A possible cause is less efficient GPU kernels for a batch of one on the
T4, but I did not investigate further.

## Limitations

- **Small evaluation set.** 20 prompts, with 1 to 4 per category. Differences of 1 to 3 prompts are not
  statistically meaningful; a larger benchmark (e.g. `lm-eval` on a few hundred questions) would be needed.
- **LLM judge.** Scores varied between two judge runs on identical answers (Haiku 4.90 → 4.95,
  one prompt rescored 4 → 5), and some judgments were strict (e.g. rejecting grapes as a red
  fruit). The judge is also the same model being compared, which may favor its own answers.
  I spot-checked judgments by hand.
- **Truncation.** The 250-token cap cut off some longer answers, which cost points for every model.
- **Few runs.** 6 requests per concurrency level; p95 values come from a small sample.
- **Hosted latency** depends on network conditions and API load at the time of testing. The hosted model
  was only tested up to 2 concurrent requests to stay within rate limits.
- **Different engines and hardware.** Part A (Ollama, GGUF, CPU) and Part B (vLLM, Safetensors, GPU) use
  different software as well as hardware, and Q4_K_M (GGUF) and AWQ are different 4-bit methods,
  so cross-part comparisons show overall differences, not the effect of one variable.
- **Measurement contamination (fixed).** An early Qwen run measured 3.69 GB because the Q8 model was
  still loaded (Ollama keeps models in memory for 5 minutes). It was rerun after `ollama stop`; the
  contaminated run also showed higher TTFT (0.61 s vs 0.36 s) and RAM (7.2 vs 5.1 GB).

## Licenses

| Model | License | Notes |
|---|---|---|
| Llama 3.2 3B | Llama 3.2 Community License | Custom license with an acceptable use policy and attribution requirements; not OSI open source |
| Qwen 2.5 1.5B | Apache 2.0 | Permissive, commercial use allowed |
| Qwen 2.5 3B (Part B) | Qwen Research License | Different from other Qwen 2.5 sizes; check the model card before commercial use |
| Claude Haiku 4.5 | Anthropic commercial terms | Proprietary, API access only |

Licenses can differ between sizes in the same model family, so check each model card.

## When to choose local vs hosted

**Local** makes sense when data must not leave the machine, when the app needs to work offline,
when outputs are short (classification, extraction, yes/no), or when request volume is high enough
that API costs matter and suitable hardware is available. **Hosted** is the better choice for
long outputs, many concurrent users, and tasks needing broad factual knowledge, and on
hardware like this laptop it is far faster and better at almost no setup cost.
