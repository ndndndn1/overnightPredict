"""DeepSeek LLM provider implementation."""

from typing import Any, AsyncIterator, Optional

from src.core.config.settings import DeepSeekConfig
from src.core.exceptions import LLMProviderError, RateLimitError
from src.domain.interfaces.llm_provider import LLMMessage, LLMResponse
from src.infrastructure.llm_providers.base_provider import BaseLLMProvider


class DeepSeekProvider(BaseLLMProvider):
    """
    DeepSeek API provider implementation.

    DeepSeek provides code-focused LLM models with OpenAI-compatible API.
    """

    def __init__(self, config: DeepSeekConfig):
        """Initialize the DeepSeek provider.

        Args:
            config: DeepSeek configuration.
        """
        super().__init__(config)
        self._deepseek_config = config
        self._client: Optional[Any] = None

    async def _get_client(self) -> Any:
        """Get or create the DeepSeek client (OpenAI-compatible)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                api_key = self._deepseek_config.api_key
                if api_key is None:
                    raise LLMProviderError(
                        "DeepSeek API key not configured",
                        provider=self.provider_name,
                    )

                self._client = AsyncOpenAI(
                    api_key=api_key.get_secret_value(),
                    base_url=self._deepseek_config.api_base,
                )
            except ImportError:
                raise LLMProviderError(
                    "OpenAI package not installed (required for DeepSeek). Run: pip install openai",
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
        """Perform completion using DeepSeek API."""
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

            if "rate_limit" in error_str.lower() or "429" in error_str:
                retry_after = 60.0
                self._handle_rate_limit(retry_after)
                raise RateLimitError(
                    "DeepSeek rate limit exceeded",
                    provider=self.provider_name,
                    retry_after=retry_after,
                    cause=e,
                )

            raise LLMProviderError(
                f"DeepSeek completion failed: {error_str}",
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
        """Perform streaming completion using DeepSeek API."""
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

            if "rate_limit" in error_str.lower() or "429" in error_str:
                self._handle_rate_limit(60.0)
                raise RateLimitError(
                    "DeepSeek rate limit exceeded",
                    provider=self.provider_name,
                    retry_after=60.0,
                    cause=e,
                )

            raise LLMProviderError(
                f"DeepSeek stream failed: {error_str}",
                provider=self.provider_name,
                cause=e,
            )

    async def code_complete(
        self,
        prefix: str,
        suffix: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Code completion with fill-in-the-middle.

        DeepSeek Coder supports FIM (Fill-In-Middle) completion.

        Args:
            prefix: Code before the cursor.
            suffix: Code after the cursor.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated code to fill the middle.
        """
        # DeepSeek uses special tokens for FIM
        prompt = f"<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"

        response = await self.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.content
