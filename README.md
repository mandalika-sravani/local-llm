# local-llm-bench

Run an open-weight LLM locally, expose it through an API, and benchmark latency, memory
and output quality against a hosted model.

## Hardware and plan

| Part | Where | What |
|---|---|---|
| A | Laptop: i7-8550U (4 cores / 8 threads), 8 GB RAM, no usable GPU | Ollama on CPU, Q4 vs Q8 quantization, RAM profiling, FastAPI gateway |
| B | Google Colab, free T4 GPU (16 GB) | vLLM, continuous batching, GPU VRAM, FP16 vs AWQ ([colab_vllm.md](colab_vllm.md)) |
| Hosted | Anthropic or OpenAI API | Baseline for latency and quality |

The Radeon 520 (2 GB) isn't supported by Ollama's GPU backends and is too small to help,
so local inference is CPU-only. With 8 GB RAM, stay at about 3B parameters or below.

## Memory estimate (do this before running anything)

Weights ~= parameters x bytes per parameter
KV cache per token = 2 (K and V) x layers x kv_heads x head_dim x bytes

Worked example, Llama 3.2 3B (28 layers, 8 KV heads, head_dim 128, FP16 cache):

| | Estimate |
|---|---|
| Weights Q4_K_M (~4.5-5 bits/param) | ~2.0 GB |
| Weights Q8_0 | ~3.4 GB |
| Weights FP16 | ~6.4 GB (will not fit alongside Windows in 8 GB) |
| KV cache per token | 2 x 28 x 8 x 128 x 2 B = ~112 KB |
| KV cache at 4096 context | ~0.45 GB per sequence |

Compare these predictions with the measured `ollama_model_gb` column.

Decode on CPU is memory-bandwidth bound: tokens/s ~= RAM bandwidth / model size.
That is why Q4 should generate noticeably faster than Q8 on this laptop.

## Setup (Windows)

1. Install Ollama from https://ollama.com/download, then:
   ```
   ollama pull llama3.2:3b
   ollama pull llama3.2:3b-instruct-q8_0
   ollama pull qwen2.5:1.5b
   ```
2. Python environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env      # then add your hosted API key
   ```
3. Before benchmarking: plug in the charger, set Windows power mode to Best performance,
   and close Chrome and other heavy apps (RAM is the bottleneck).

## Run

```
# Gateway (your own API in front of every backend)
uvicorn gateway.main:app --reload --port 8001
# then try POST /generate at http://127.0.0.1:8001/docs

python -m bench.latency --backend ollama --model llama3.2-cpu --label q4_K_M
python -m bench.latency --backend ollama --model llama3.2-q8-cpu --label q8_0
python -m bench.latency --backend ollama --model qwen2.5-cpu --label q4_K_M
python -m bench.latency --backend hosted --concurrency 1,2

# Latency + memory
python -m bench.latency --backend ollama --model llama3.2:3b --label q4_K_M
python -m bench.latency --backend ollama --model llama3.2:3b-instruct-q8_0 --label q8_0
python -m bench.latency --backend ollama --model qwen2.5:1.5b --label q4_K_M
python -m bench.latency --backend hosted --concurrency 1,4,8 --runs 8

# Quality

python -m bench.quality generate --backend ollama --model llama3.2-cpu --max-tokens 250
python -m bench.quality generate --backend ollama --model qwen2.5-cpu --max-tokens 250
python -m bench.quality generate --backend hosted --max-tokens 250

python -m bench.quality generate --backend ollama --model llama3.2:3b
python -m bench.quality generate --backend ollama --model qwen2.5:1.5b
python -m bench.quality generate --backend hosted
python -m bench.quality judge "results/answers_*.jsonl"
```

Each local latency run takes a few minutes on CPU. To test Ollama under concurrency,
set `OLLAMA_NUM_PARALLEL=2` (System > Environment Variables), restart Ollama, and run
with `--concurrency 1,2`. Watch how per-request speed drops while total throughput
barely moves on CPU, then compare with vLLM on the GPU.

## Results

| Model | Backend | Quant | Memory (GB) | TTFT p50 (s) | Decode tok/s | Throughput at max concurrency | Quality (1-5) | License |
|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | Ollama CPU | Q4_K_M | | | | | | Llama 3.2 Community |
| llama3.2:3b | Ollama CPU | Q8_0 | | | | | | Llama 3.2 Community |
| qwen2.5:1.5b | Ollama CPU | Q4_K_M | | | | | | Apache 2.0 |
| Qwen2.5-3B-Instruct | vLLM T4 | FP16 | | | | | | Qwen Research (check the model card) |
| Qwen2.5-3B-Instruct-AWQ | vLLM T4 | AWQ 4-bit | | | | | | Qwen Research (check the model card) |
| hosted model | API | n/a | n/a | | | | | Proprietary terms |

Check each license on its Hugging Face model card yourself and record it. License terms
differ between model sizes even within one family.

## Findings

Write 5-8 sentences: memory estimate vs measured, the Q4 vs Q8 speed and quality tradeoff,
CPU vs GPU, Ollama vs vLLM under concurrency, and when you'd choose local vs hosted.
