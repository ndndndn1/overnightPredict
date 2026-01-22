"""Base LLM provider implementation."""

import asyncio
import time
from abc import abstractmethod
from typing import Any, AsyncIterator, Optional

from src.core.config.settings import LLMProviderConfig
from src.core.exceptions import LLMProviderError, RateLimitError
from src.core.utils.async_helpers import retry_async, AsyncThrottler
from src.core.utils.logging import get_logger
from src.domain.interfaces.llm_provider import ILLMProvider, LLMMessage, LLMResponse


class BaseLLMProvider(ILLMProvider):
    """
    Base implementation for LLM providers.

    Provides common functionality like rate limiting, retries, and logging.
    """

    def __init__(self, config: LLMProviderConfig):
        """Initialize the provider.

        Args:
            config: Provider configuration.
        """
        self._config = config
        self._logger = get_logger(
            f"llm.{config.provider.value}",
            provider=config.provider.value,
            model=config.model,
        )

        # Rate limiting
        self._throttler = AsyncThrottler(
            rate_limit=config.requests_per_minute,
            period=60.0,
        )

        # Usage tracking
        self._total_tokens = 0
        self._total_requests = 0
        self._rate_limited_until: Optional[float] = None

        # Health
        self._available = True
        self._last_error: Optional[str] = None

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._config.provider.value

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._config.model

    @property
    def is_available(self) -> bool:
        """Check if provider is available."""
        return self._available and self._config.enabled

    @abstractmethod
    async def _do_complete(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop: Optional[list[str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Actual completion implementation."""
        ...

    @abstractmethod
    async def _do_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop: Optional[list[str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Actual streaming implementation."""
        ...

    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion with rate limiting and retries."""
        if not self.is_available:
            raise LLMProviderError(
                f"Provider {self.provider_name} is not available",
                provider=self.provider_name,
            )

        # Check and wait for rate limit
        is_limited, retry_after = await self.check_rate_limit()
        if is_limited and retry_after:
            self._logger.warning(
                "Rate limited, waiting",
                retry_after=retry_after,
            )
            await asyncio.sleep(retry_after)

        # Acquire throttle token
        await self._throttler.acquire()

        start_time = time.time()
        try:
            response = await self._retry_complete(
                messages=messages,
                max_tokens=max_tokens or self._config.max_tokens,
                temperature=temperature or self._config.temperature,
                stop=stop,
                **kwargs,
            )

            # Track usage
            self._total_tokens += response.total_tokens
            self._total_requests += 1

            # Calculate latency
            response.latency_ms = (time.time() - start_time) * 1000

            self._logger.debug(
                "Completion successful",
                tokens=response.total_tokens,
                latency_ms=response.latency_ms,
            )

            return response

        except Exception as e:
            self._logger.error(
                "Completion failed",
                error=str(e),
            )
            raise

    @retry_async(
        max_retries=3,
        delay=1.0,
        multiplier=2.0,
        exceptions=(LLMProviderError,),
    )
    async def _retry_complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Completion with retry logic."""
        return await self._do_complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion."""
        if not self.is_available:
            raise LLMProviderError(
                f"Provider {self.provider_name} is not available",
                provider=self.provider_name,
            )

        # Check rate limit
        is_limited, retry_after = await self.check_rate_limit()
        if is_limited and retry_after:
            await asyncio.sleep(retry_after)

        await self._throttler.acquire()

        try:
            async for chunk in self._do_stream(
                messages=messages,
                max_tokens=max_tokens or self._config.max_tokens,
                temperature=temperature or self._config.temperature,
                stop=stop,
                **kwargs,
            ):
                yield chunk

            self._total_requests += 1

        except Exception as e:
            self._logger.error("Stream failed", error=str(e))
            raise

    async def check_rate_limit(self) -> tuple[bool, Optional[float]]:
        """Check if currently rate limited."""
        if self._rate_limited_until is None:
            return (False, None)

        now = time.time()
        if now >= self._rate_limited_until:
            self._rate_limited_until = None
            return (False, None)

        retry_after = self._rate_limited_until - now
        return (True, retry_after)

    async def wait_for_rate_limit(self) -> None:
        """Wait until rate limit is cleared."""
        while True:
            is_limited, retry_after = await self.check_rate_limit()
            if not is_limited:
                return

            if retry_after:
                self._logger.info(
                    "Waiting for rate limit",
                    retry_after=retry_after,
                )
                await asyncio.sleep(retry_after)

    def _handle_rate_limit(self, retry_after: float) -> None:
        """Handle rate limit response."""
        self._rate_limited_until = time.time() + retry_after
        self._logger.warning(
            "Rate limit hit",
            retry_after=retry_after,
        )

    async def get_usage(self) -> dict[str, int]:
        """Get current usage statistics."""
        return {
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
            "available_tokens": int(self._throttler.available_tokens),
        }

    async def health_check(self) -> bool:
        """Check provider health."""
        try:
            response = await self.complete(
                [LLMMessage(role="user", content="ping")],
                max_tokens=5,
            )
            self._available = response.has_content
            return self._available
        except Exception as e:
            self._available = False
            self._last_error = str(e)
            return False

    def set_available(self, available: bool) -> None:
        """Set availability status."""
        self._available = available
