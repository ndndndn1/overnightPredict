"""Rate limiting implementations."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.core.utils.logging import get_logger


@dataclass
class RateLimitState:
    """State of a rate limiter."""

    requests_made: int = 0
    tokens_used: int = 0
    last_request_at: Optional[datetime] = None
    blocked_until: Optional[datetime] = None
    violations: int = 0


class TokenBucket:
    """
    Token bucket rate limiter.

    Provides smooth rate limiting with burst capability.
    """

    def __init__(
        self,
        rate: float,
        capacity: int,
        initial_tokens: Optional[float] = None,
    ):
        """Initialize token bucket.

        Args:
            rate: Tokens added per second.
            capacity: Maximum bucket capacity.
            initial_tokens: Initial tokens (defaults to capacity).
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = initial_tokens if initial_tokens is not None else float(capacity)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            Wait time in seconds (0 if no wait needed).
        """
        async with self._lock:
            wait_time = await self._wait_for_tokens(tokens)
            self._tokens -= tokens
            return wait_time

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired.
        """
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def _wait_for_tokens(self, tokens: int) -> float:
        """Wait until enough tokens are available."""
        total_wait = 0.0

        while True:
            self._refill()

            if self._tokens >= tokens:
                return total_wait

            # Calculate wait time
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self._rate

            # Release lock while waiting
            self._lock.release()
            try:
                await asyncio.sleep(wait_time)
                total_wait += wait_time
            finally:
                await self._lock.acquire()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_update = now

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        now = time.monotonic()
        elapsed = now - self._last_update
        return min(self._capacity, self._tokens + elapsed * self._rate)

    @property
    def is_empty(self) -> bool:
        """Check if bucket is empty."""
        return self.available_tokens < 1


class SlidingWindowLimiter:
    """
    Sliding window rate limiter.

    Provides more accurate rate limiting than fixed windows.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
    ):
        """Initialize sliding window limiter.

        Args:
            max_requests: Maximum requests per window.
            window_seconds: Window duration in seconds.
        """
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Acquire permission, waiting if necessary.

        Returns:
            Wait time in seconds.
        """
        async with self._lock:
            self._cleanup()

            if len(self._requests) < self._max_requests:
                self._requests.append(time.monotonic())
                return 0.0

            # Calculate wait time
            oldest = self._requests[0]
            wait_time = oldest + self._window_seconds - time.monotonic()

            if wait_time > 0:
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()
                self._cleanup()

            self._requests.append(time.monotonic())
            return max(0.0, wait_time)

    async def try_acquire(self) -> bool:
        """Try to acquire permission without waiting.

        Returns:
            True if permission was granted.
        """
        async with self._lock:
            self._cleanup()
            if len(self._requests) < self._max_requests:
                self._requests.append(time.monotonic())
                return True
            return False

    def _cleanup(self) -> None:
        """Remove expired requests."""
        cutoff = time.monotonic() - self._window_seconds
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

    @property
    def current_count(self) -> int:
        """Get current request count in window."""
        self._cleanup()
        return len(self._requests)

    @property
    def available_requests(self) -> int:
        """Get available requests in window."""
        return max(0, self._max_requests - self.current_count)


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiter."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    burst_multiplier: float = 1.5
    retry_after_violation: float = 60.0


class RateLimiter:
    """
    Composite rate limiter for LLM API calls.

    Combines request and token limits with violation handling.
    """

    def __init__(self, config: RateLimiterConfig, name: str = "default"):
        """Initialize rate limiter.

        Args:
            config: Rate limiter configuration.
            name: Limiter name for logging.
        """
        self._config = config
        self._name = name
        self._logger = get_logger(f"ratelimit.{name}")

        # Request limiter
        self._request_bucket = TokenBucket(
            rate=config.requests_per_minute / 60.0,
            capacity=int(config.requests_per_minute * config.burst_multiplier),
        )

        # Token limiter
        self._token_bucket = TokenBucket(
            rate=config.tokens_per_minute / 60.0,
            capacity=int(config.tokens_per_minute * config.burst_multiplier),
        )

        # State tracking
        self._state = RateLimitState()
        self._blocked_until: Optional[float] = None

    async def acquire(self, estimated_tokens: int = 1000) -> float:
        """Acquire permission for a request.

        Args:
            estimated_tokens: Estimated tokens for the request.

        Returns:
            Total wait time in seconds.
        """
        # Check if blocked
        if self._blocked_until:
            now = time.monotonic()
            if now < self._blocked_until:
                wait_time = self._blocked_until - now
                self._logger.info("Blocked, waiting", wait_time=wait_time)
                await asyncio.sleep(wait_time)
            self._blocked_until = None

        # Acquire both limits
        request_wait = await self._request_bucket.acquire(1)
        token_wait = await self._token_bucket.acquire(estimated_tokens)

        total_wait = request_wait + token_wait

        # Update state
        self._state.requests_made += 1
        self._state.tokens_used += estimated_tokens
        self._state.last_request_at = datetime.utcnow()

        if total_wait > 0:
            self._logger.debug(
                "Rate limited",
                wait_time=total_wait,
                requests=self._state.requests_made,
            )

        return total_wait

    async def try_acquire(self, estimated_tokens: int = 1000) -> bool:
        """Try to acquire permission without waiting.

        Args:
            estimated_tokens: Estimated tokens for the request.

        Returns:
            True if permission was granted.
        """
        # Check if blocked
        if self._blocked_until and time.monotonic() < self._blocked_until:
            return False

        # Try both limits
        if not await self._request_bucket.try_acquire(1):
            return False

        if not await self._token_bucket.try_acquire(estimated_tokens):
            # Return the request token
            self._request_bucket._tokens += 1
            return False

        self._state.requests_made += 1
        self._state.tokens_used += estimated_tokens
        self._state.last_request_at = datetime.utcnow()

        return True

    def report_violation(self, retry_after: Optional[float] = None) -> None:
        """Report a rate limit violation from the API.

        Args:
            retry_after: Seconds to wait (from API response).
        """
        self._state.violations += 1

        wait_time = retry_after or self._config.retry_after_violation
        self._blocked_until = time.monotonic() + wait_time
        self._state.blocked_until = datetime.utcnow()

        self._logger.warning(
            "Rate limit violation",
            violations=self._state.violations,
            blocked_for=wait_time,
        )

    def report_tokens_used(self, actual_tokens: int, estimated_tokens: int) -> None:
        """Report actual token usage for adjustment.

        Args:
            actual_tokens: Actual tokens used.
            estimated_tokens: Originally estimated tokens.
        """
        # Adjust token bucket if we under/overestimated
        diff = estimated_tokens - actual_tokens
        if diff != 0:
            self._token_bucket._tokens += diff

    @property
    def state(self) -> RateLimitState:
        """Get current state."""
        return self._state

    @property
    def is_blocked(self) -> bool:
        """Check if currently blocked."""
        if self._blocked_until is None:
            return False
        return time.monotonic() < self._blocked_until

    @property
    def block_remaining(self) -> float:
        """Get remaining block time in seconds."""
        if self._blocked_until is None:
            return 0.0
        remaining = self._blocked_until - time.monotonic()
        return max(0.0, remaining)

    def reset(self) -> None:
        """Reset the rate limiter state."""
        self._state = RateLimitState()
        self._blocked_until = None
        self._request_bucket = TokenBucket(
            rate=self._config.requests_per_minute / 60.0,
            capacity=int(self._config.requests_per_minute * self._config.burst_multiplier),
        )
        self._token_bucket = TokenBucket(
            rate=self._config.tokens_per_minute / 60.0,
            capacity=int(self._config.tokens_per_minute * self._config.burst_multiplier),
        )
