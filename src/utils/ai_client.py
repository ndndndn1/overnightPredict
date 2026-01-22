"""AI client abstraction for multiple providers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import Settings

logger = structlog.get_logger(__name__)


class AIClient(ABC):
    """Abstract base class for AI clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the AI model."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Any:
        """Generate a streaming response from the AI model."""
        pass


class AnthropicClient(AIClient):
    """Anthropic Claude API client."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the Anthropic client."""
        self.settings = settings
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Get the Anthropic client (lazy init)."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=self.settings.anthropic_api_key
                )
            except ImportError:
                raise ImportError("anthropic package not installed")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response using Claude."""
        max_tokens = max_tokens or self.settings.ai.anthropic_max_tokens
        temperature = temperature or self.settings.ai.anthropic_temperature

        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.client.messages.create(
                model=self.settings.ai.anthropic_model,
                max_tokens=max_tokens,
                system=system_prompt or "",
                messages=messages,
                temperature=temperature,
            )

            return response.content[0].text
        except Exception as e:
            logger.error("Anthropic API error", error=str(e))
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a streaming response using Claude."""
        max_tokens = max_tokens or self.settings.ai.anthropic_max_tokens
        temperature = temperature or self.settings.ai.anthropic_temperature

        messages = [{"role": "user", "content": prompt}]

        async with self.client.messages.stream(
            model=self.settings.ai.anthropic_model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=messages,
            temperature=temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class OpenAIClient(AIClient):
    """OpenAI API client."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the OpenAI client."""
        self.settings = settings
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Get the OpenAI client (lazy init)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            except ImportError:
                raise ImportError("openai package not installed")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response using OpenAI."""
        max_tokens = max_tokens or self.settings.ai.openai_max_tokens
        temperature = temperature or self.settings.ai.openai_temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.ai.openai_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("OpenAI API error", error=str(e))
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a streaming response using OpenAI."""
        max_tokens = max_tokens or self.settings.ai.openai_max_tokens
        temperature = temperature or self.settings.ai.openai_temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.settings.ai.openai_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class MockAIClient(AIClient):
    """Mock AI client for testing."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the mock client."""
        self.settings = settings
        self._responses: list[str] = []
        self._response_index = 0

    def set_responses(self, responses: list[str]) -> None:
        """Set predefined responses for testing."""
        self._responses = responses
        self._response_index = 0

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a mock response."""
        if self._responses:
            response = self._responses[self._response_index % len(self._responses)]
            self._response_index += 1
            return response

        # Generate a basic response based on the prompt
        return f"""### Answer
Based on the prompt, here is a comprehensive response.

### Code
```python
def example_implementation():
    \"\"\"Example implementation based on the prompt.\"\"\"
    pass
```

### Follow-up Questions
1. What additional features should be implemented?
2. How should error handling be improved?
3. What tests are needed?
"""

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Any:
        """Generate a mock streaming response."""
        response = await self.generate(
            prompt, system_prompt, max_tokens, temperature, **kwargs
        )
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.01)


_cached_ai_client: AIClient | None = None


def get_ai_client(settings: Settings | None = None) -> AIClient:
    """Get the appropriate AI client based on settings."""
    global _cached_ai_client

    if _cached_ai_client is not None:
        return _cached_ai_client

    if settings is None:
        from src.core.config import get_settings
        settings = get_settings()

    provider = settings.ai.primary_provider.lower()

    if provider == "anthropic":
        _cached_ai_client = AnthropicClient(settings)
    elif provider == "openai":
        _cached_ai_client = OpenAIClient(settings)
    elif provider == "mock":
        _cached_ai_client = MockAIClient(settings)
    else:
        logger.warning(
            "Unknown AI provider, using mock",
            provider=provider,
        )
        _cached_ai_client = MockAIClient(settings)

    return _cached_ai_client
