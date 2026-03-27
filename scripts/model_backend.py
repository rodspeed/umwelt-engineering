"""
Multi-model backend for LLM experiments.
Supports Anthropic, OpenAI, Google Gemini, and Ollama (local).

Usage:
    from model_backend import create_backend

    backend = create_backend("claude-haiku-4-5-20251001")
    result = backend.complete_sync("You are helpful.", "What is 2+2?")
    print(result.text)

    # Or async:
    result = await backend.complete("You are helpful.", "What is 2+2?")
"""

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    model: str
    provider: str
    stop_reason: str


class ModelBackend(ABC):
    """Abstract base for LLM API backends."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    @property
    @abstractmethod
    def provider(self) -> str: ...

    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> CompletionResult: ...

    def complete_sync(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Synchronous wrapper around complete()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.complete(system, user, max_tokens, temperature))
                return future.result()
        else:
            return asyncio.run(self.complete(system, user, max_tokens, temperature))


class AnthropicBackend(ModelBackend):
    provider = "anthropic"

    def __init__(self, model_id: str):
        super().__init__(model_id)
        import anthropic
        self._async_client = anthropic.AsyncAnthropic()
        self._sync_client = anthropic.Anthropic()

    async def complete(self, system, user, max_tokens=2048, temperature=0.0):
        t0 = time.time()
        response = await self._async_client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        elapsed = time.time() - t0
        return CompletionResult(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_seconds=round(elapsed, 3),
            model=response.model,
            provider="anthropic",
            stop_reason=response.stop_reason,
        )

    def complete_sync(self, system, user, max_tokens=2048, temperature=0.0):
        t0 = time.time()
        response = self._sync_client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        elapsed = time.time() - t0
        return CompletionResult(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_seconds=round(elapsed, 3),
            model=response.model,
            provider="anthropic",
            stop_reason=response.stop_reason,
        )


class OpenAIBackend(ModelBackend):
    provider = "openai"

    def __init__(self, model_id: str):
        super().__init__(model_id)
        from openai import AsyncOpenAI, OpenAI
        self._async_client = AsyncOpenAI()
        self._sync_client = OpenAI()

    async def complete(self, system, user, max_tokens=2048, temperature=0.0):
        t0 = time.time()
        response = await self._async_client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        elapsed = time.time() - t0
        choice = response.choices[0]
        return CompletionResult(
            text=choice.message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_seconds=round(elapsed, 3),
            model=response.model,
            provider="openai",
            stop_reason=choice.finish_reason,
        )

    def complete_sync(self, system, user, max_tokens=2048, temperature=0.0):
        t0 = time.time()
        response = self._sync_client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        elapsed = time.time() - t0
        choice = response.choices[0]
        return CompletionResult(
            text=choice.message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_seconds=round(elapsed, 3),
            model=response.model,
            provider="openai",
            stop_reason=choice.finish_reason,
        )


class GeminiBackend(ModelBackend):
    provider = "google"

    def __init__(self, model_id: str):
        super().__init__(model_id)
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        self._client = genai.Client(api_key=api_key)

    async def complete(self, system, user, max_tokens=2048, temperature=0.0):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.complete_sync, system, user, max_tokens, temperature
        )

    def complete_sync(self, system, user, max_tokens=2048, temperature=0.0):
        from google.genai import types
        t0 = time.time()
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        elapsed = time.time() - t0
        usage = response.usage_metadata
        return CompletionResult(
            text=response.text,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            latency_seconds=round(elapsed, 3),
            model=self.model_id,
            provider="google",
            stop_reason=(
                response.candidates[0].finish_reason.name
                if response.candidates else "unknown"
            ),
        )


class OllamaBackend(ModelBackend):
    provider = "ollama"

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self._model_name = model_id.removeprefix("ollama/")
        self._base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    async def complete(self, system, user, max_tokens=2048, temperature=0.0):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.complete_sync, system, user, max_tokens, temperature
        )

    def complete_sync(self, system, user, max_tokens=2048, temperature=0.0):
        import requests
        t0 = time.time()
        resp = requests.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - t0
        return CompletionResult(
            text=data["message"]["content"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_seconds=round(elapsed, 3),
            model=self._model_name,
            provider="ollama",
            stop_reason=data.get("done_reason", "stop"),
        )


# --- Factory ---

_PROVIDER_PREFIXES = {
    "claude-": AnthropicBackend,
    "gpt-": OpenAIBackend,
    "o1": OpenAIBackend,
    "o3": OpenAIBackend,
    "o4": OpenAIBackend,
    "gemini-": GeminiBackend,
    "ollama/": OllamaBackend,
}


def create_backend(model_id: str) -> ModelBackend:
    """Create the appropriate backend from a model ID string.

    Routing:
        claude-*     → Anthropic
        gpt-*, o1*   → OpenAI
        gemini-*     → Google Gemini
        ollama/*     → Local Ollama
    """
    for prefix, cls in _PROVIDER_PREFIXES.items():
        if model_id.startswith(prefix):
            return cls(model_id)
    raise ValueError(
        f"Unknown model '{model_id}'. "
        f"Prefix must start with one of: {list(_PROVIDER_PREFIXES.keys())}"
    )


def list_available_backends() -> dict[str, bool]:
    """Check which backends have their dependencies installed."""
    available = {}
    try:
        import anthropic  # noqa: F401
        available["anthropic"] = True
    except ImportError:
        available["anthropic"] = False
    try:
        import openai  # noqa: F401
        available["openai"] = True
    except ImportError:
        available["openai"] = False
    try:
        from google import genai  # noqa: F401
        available["google"] = True
    except ImportError:
        available["google"] = False
    try:
        import requests  # noqa: F401
        available["ollama"] = True
    except ImportError:
        available["ollama"] = False
    return available
