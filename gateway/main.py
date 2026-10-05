"""FastAPI gateway: one /generate endpoint in front of local and hosted models.

  uvicorn gateway.main:app --reload --port 8001
  Open http://127.0.0.1:8001/docs
"""
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backends import DEFAULT_MODELS, get_client

app = FastAPI(title="Local LLM Gateway", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    backend: Literal["ollama", "vllm", "hosted"] = "ollama"
    model: str | None = None
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0, le=2)


class GenerateResponse(BaseModel):
    backend: str
    model: str
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "default_models": DEFAULT_MODELS}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    model = req.model or DEFAULT_MODELS[req.backend]
    try:
        client = get_client(req.backend)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": req.prompt}],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except Exception as e:  # backend down, model not pulled, bad key, ...
        raise HTTPException(status_code=502, detail=f"{req.backend} error: {e}")

    usage = resp.usage
    return GenerateResponse(
        backend=req.backend,
        model=model,
        text=resp.choices[0].message.content or "",
        latency_s=round(time.perf_counter() - t0, 3),
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
    )
