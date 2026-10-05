"""Shared backend config. Every backend (Ollama, vLLM, hosted) speaks the OpenAI chat API,
so one client and one benchmark script cover all of them."""
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(override=True)

BACKENDS = {
    "ollama": {"base_url": os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), "api_key": "ollama"},
    "vllm": {"base_url": os.getenv("VLLM_URL", "http://localhost:8000/v1"), "api_key": "EMPTY"},
    "hosted": {
        "base_url": os.getenv("HOSTED_BASE_URL", "https://api.anthropic.com/v1/"),
        "api_key": os.getenv("HOSTED_API_KEY", ""),
    },
}

DEFAULT_MODELS = {
    "ollama": "llama3.2:3b",
    "vllm": "Qwen/Qwen2.5-3B-Instruct",
    "hosted": os.getenv("HOSTED_MODEL", "claude-haiku-4-5"),
}

_clients: dict[str, AsyncOpenAI] = {}


def get_client(backend: str) -> AsyncOpenAI:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {list(BACKENDS)}")
    if backend not in _clients:
        cfg = BACKENDS[backend]
        if backend == "hosted" and not cfg["api_key"]:
            raise RuntimeError("HOSTED_API_KEY is not set (see .env.example)")
        headers = {}
        if backend == "hosted" and os.getenv("HOSTED_WORKSPACE_ID"):
            headers["anthropic-workspace-id"] = os.getenv("HOSTED_WORKSPACE_ID")
        _clients[backend] = AsyncOpenAI(timeout=600, default_headers=headers or None, **cfg)
    return _clients[backend]


def ollama_native_url() -> str:
    """Ollama's native API (for /api/ps memory stats) lives at the root, not under /v1."""
    return BACKENDS["ollama"]["base_url"].rstrip("/").removesuffix("/v1")
