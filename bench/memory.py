"""Samples memory in the background while requests run and keeps the peak values.

- System RAM: psutil (what matters on a CPU-only laptop)
- Ollama model footprint: Ollama's /api/ps reports size (total) and size_vram (part on GPU)
- GPU memory: NVML, if an NVIDIA GPU exists (Colab). Note vLLM pre-allocates ~90% of VRAM
  by default, so for vLLM also read the "model weights took" / KV cache lines in its log.
"""
import asyncio

import httpx
import psutil

from backends import ollama_native_url

GB = 1024**3


class MemorySampler:
    def __init__(self, backend: str, interval: float = 0.5):
        self.backend = backend
        self.interval = interval
        self.peak_ram = 0
        self.peak_ollama_total = 0
        self.peak_ollama_vram = 0
        self.peak_gpu = 0
        self._task = None
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass  # no NVIDIA GPU: fine on a CPU-only machine

    async def _poll(self):
        async with httpx.AsyncClient(timeout=2) as http:
            while True:
                self.peak_ram = max(self.peak_ram, psutil.virtual_memory().used)
                if self.backend == "ollama":
                    try:
                        r = await http.get(f"{ollama_native_url()}/api/ps")
                        for m in r.json().get("models", []):
                            self.peak_ollama_total = max(self.peak_ollama_total, m.get("size", 0))
                            self.peak_ollama_vram = max(self.peak_ollama_vram, m.get("size_vram", 0))
                    except Exception:
                        pass
                if self._nvml:
                    used = self._nvml.nvmlDeviceGetMemoryInfo(self._gpu).used
                    self.peak_gpu = max(self.peak_gpu, used)
                await asyncio.sleep(self.interval)

    async def __aenter__(self):
        self._task = asyncio.create_task(self._poll())
        return self

    async def __aexit__(self, *exc):
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    def summary(self) -> dict:
        return {
            "peak_sys_ram_gb": round(self.peak_ram / GB, 2),
            "ollama_model_gb": round(self.peak_ollama_total / GB, 2),
            "ollama_on_gpu_gb": round(self.peak_ollama_vram / GB, 2),
            "peak_gpu_gb": round(self.peak_gpu / GB, 2),
        }
