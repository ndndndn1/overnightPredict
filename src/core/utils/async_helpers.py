"""Async utility functions for concurrent operations."""

import asyncio
from functools import wraps
from typing import Any, Awaitable, Callable, Optional, TypeVar

from src.core.exceptions import OvernightPredictError

T = TypeVar("T")


async def run_with_timeout(
    coro: Awaitable[T],
    timeout: float,
    error_message: str = "Operation timed out",
) -> T:
    """Run a coroutine with a timeout.

    Args:
        coro: The coroutine to run.
        timeout: Timeout in seconds.
        error_message: Error message if timeout occurs.

    Returns:
        The result of the coroutine.

    Raises:
        OvernightPredictError: If the operation times out.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise OvernightPredictError(error_message, details={"timeout": timeout})


async def gather_with_concurrency(
    tasks: list[Awaitable[T]],
    max_concurrent: int,
    return_exceptions: bool = False,
) -> list[T | BaseException]:
    """Execute tasks with limited concurrency.

    Args:
        tasks: List of coroutines to execute.
        max_concurrent: Maximum number of concurrent tasks.
        return_exceptions: If True, exceptions are returned as results.

    Returns:
        List of results in the same order as tasks.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_task(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    return await asyncio.gather(
        *[limited_task(task) for task in tasks],
        return_exceptions=return_exceptions,
    )


def retry_async(
    max_retries: int = 3,
    delay: float = 1.0,
    multiplier: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for retrying async functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Initial delay between retries in seconds.
        multiplier: Multiplier for exponential backoff.
        exceptions: Tuple of exception types to catch and retry.

    Returns:
        Decorated function.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= multiplier

            raise OvernightPredictError(
                f"Max retries ({max_retries}) exceeded for {func.__name__}",
                details={"attempts": max_retries + 1},
                cause=last_exception,
            )

        return wrapper

    return decorator


class AsyncThrottler:
    """Async throttler for rate limiting operations.

    Implements token bucket algorithm for smooth rate limiting.
    """

    def __init__(
        self,
        rate_limit: float,
        period: float = 1.0,
        burst: Optional[int] = None,
    ):
        """Initialize throttler.

        Args:
            rate_limit: Maximum operations per period.
            period: Time period in seconds.
            burst: Maximum burst size (defaults to rate_limit).
        """
        self.rate_limit = rate_limit
        self.period = period
        self.burst = burst or int(rate_limit)

        self._tokens = float(self.burst)
        self._last_update = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire.
        """
        async with self._lock:
            await self._wait_for_tokens(tokens)
            self._tokens -= tokens

    async def _wait_for_tokens(self, tokens: int) -> None:
        """Wait until enough tokens are available."""
        while True:
            self._refill_tokens()

            if self._tokens >= tokens:
                return

            # Calculate wait time
            tokens_needed = tokens - self._tokens
            wait_time = (tokens_needed / self.rate_limit) * self.period
            await asyncio.sleep(wait_time)

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        try:
            now = asyncio.get_event_loop().time()
        except RuntimeError:
            return

        elapsed = now - self._last_update
        self._tokens = min(
            self.burst,
            self._tokens + (elapsed / self.period) * self.rate_limit,
        )
        self._last_update = now

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        self._refill_tokens()
        return self._tokens


class AsyncEventEmitter:
    """Simple async event emitter for pub/sub patterns."""

    def __init__(self):
        self._listeners: dict[str, list[Callable[..., Awaitable[None]]]] = {}

    def on(self, event: str, callback: Callable[..., Awaitable[None]]) -> None:
        """Register an event listener.

        Args:
            event: Event name.
            callback: Async callback function.
        """
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)

    def off(self, event: str, callback: Callable[..., Awaitable[None]]) -> None:
        """Remove an event listener.

        Args:
            event: Event name.
            callback: Callback to remove.
        """
        if event in self._listeners:
            self._listeners[event] = [
                cb for cb in self._listeners[event] if cb != callback
            ]

    async def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Emit an event to all listeners.

        Args:
            event: Event name.
            *args: Positional arguments for callbacks.
            **kwargs: Keyword arguments for callbacks.
        """
        if event in self._listeners:
            await asyncio.gather(
                *[callback(*args, **kwargs) for callback in self._listeners[event]],
                return_exceptions=True,
            )
