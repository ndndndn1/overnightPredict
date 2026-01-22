"""OpenAI LLM provider implementation."""

import time
from typing import Any, AsyncIterator, Optional

from src.core.config.settings import OpenAIConfig
from src.core.exceptions import LLMProviderError, RateLimitError
from src.domain.interfaces.llm_provider import LLMMessage, LLMResponse
from src.infrastructure.llm_providers.base_provider import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider implementation.

    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    """

    def __init__(self, config: OpenAIConfig):
        """Initialize the OpenAI provider.

        Args:
            config: OpenAI configuration.
        """
        super().__init__(config)
        self._openai_config = config
        self._client: Optional[Any] = None

    async def _get_client(self) -> Any:
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                api_key = self._openai_config.api_key
                if api_key is None:
                    raise LLMProviderError(
                        "OpenAI API key not configured",
                        provider=self.provider_name,
                    )

                self._client = AsyncOpenAI(
                    api_key=api_key.get_secret_value(),
                    organization=self._openai_config.organization_id,
                )
            except ImportError:
                raise LLMProviderError(
                    "OpenAI package not installed. Run: pip install openai",
                    provider=self.provider_name,
                )

        return self._client

    async def _do_complete(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop: Optional[list[str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Perform completion using OpenAI API."""
        client = await self._get_client()

        try:
            response = await client.chat.completions.create(
                model=self._config.model,
                messages=[m.to_dict() for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                **kwargs,
            )

            choice = response.choices[0]
            usage = response.usage

            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                provider=self.provider_name,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                finish_reason=choice.finish_reason or "stop",
                raw_response=response.model_dump(),
            )

        except Exception as e:
            error_str = str(e)

            # Handle rate limiting
            if "rate_limit" in error_str.lower():
                # Try to parse retry-after
                retry_after = 60.0  # Default
                if hasattr(e, "response") and e.response:
                    retry_header = e.response.headers.get("retry-after")
                    if retry_header:
                        retry_after = float(retry_header)

                self._handle_rate_limit(retry_after)
                raise RateLimitError(
                    f"OpenAI rate limit exceeded",
                    provider=self.provider_name,
                    retry_after=retry_after,
                    cause=e,
                )

            raise LLMProviderError(
                f"OpenAI completion failed: {error_str}",
                provider=self.provider_name,
                cause=e,
            )

    async def _do_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop: Optional[list[str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Perform streaming completion using OpenAI API."""
        client = await self._get_client()

        try:
            stream = await client.chat.completions.create(
                model=self._config.model,
                messages=[m.to_dict() for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            error_str = str(e)

            if "rate_limit" in error_str.lower():
                self._handle_rate_limit(60.0)
                raise RateLimitError(
                    f"OpenAI rate limit exceeded",
                    provider=self.provider_name,
                    retry_after=60.0,
                    cause=e,
                )

            raise LLMProviderError(
                f"OpenAI stream failed: {error_str}",
                provider=self.provider_name,
                cause=e,
            )

    async def get_embeddings(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        """Get embeddings for texts.

        Args:
            texts: Texts to embed.
            model: Embedding model to use.

        Returns:
            List of embedding vectors.
        """
        client = await self._get_client()

        try:
            response = await client.embeddings.create(
                model=model,
                input=texts,
            )

            return [item.embedding for item in response.data]

        except Exception as e:
            raise LLMProviderError(
                f"OpenAI embeddings failed: {str(e)}",
                provider=self.provider_name,
                cause=e,
            )
