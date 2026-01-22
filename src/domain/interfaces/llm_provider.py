"""LLM Provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Optional


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    provider: str

    # Usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0

    # Metadata
    finish_reason: str = "stop"
    raw_response: Optional[dict[str, Any]] = None

    @property
    def has_content(self) -> bool:
        """Check if response has content."""
        return bool(self.content.strip())


@dataclass
class LLMMessage:
    """Message for LLM conversation."""

    role: str  # system, user, assistant
    content: str
    name: Optional[str] = None  # For tool calls

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


class ILLMProvider(ABC):
    """
    Interface for LLM providers.

    Implementations should handle provider-specific API interactions.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion.

        Args:
            messages: List of conversation messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop: Stop sequences.
            **kwargs: Provider-specific arguments.

        Returns:
            LLM response.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming completion.

        Args:
            messages: List of conversation messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop: Stop sequences.
            **kwargs: Provider-specific arguments.

        Yields:
            Response content chunks.
        """
        ...

    @abstractmethod
    async def check_rate_limit(self) -> tuple[bool, Optional[float]]:
        """
        Check if rate limited.

        Returns:
            Tuple of (is_rate_limited, retry_after_seconds).
        """
        ...

    @abstractmethod
    async def wait_for_rate_limit(self) -> None:
        """Wait until rate limit is cleared."""
        ...

    @abstractmethod
    async def get_usage(self) -> dict[str, int]:
        """
        Get current usage statistics.

        Returns:
            Dictionary with tokens_used, requests_made, etc.
        """
        ...

    async def health_check(self) -> bool:
        """
        Check provider health.

        Returns:
            True if provider is healthy.
        """
        try:
            response = await self.complete(
                [LLMMessage(role="user", content="ping")],
                max_tokens=5,
            )
            return response.has_content
        except Exception:
            return False


class ICodeExecutionProvider(ILLMProvider):
    """
    Extended interface for code execution providers (like Claude Code).

    These providers can execute code and interact with the file system.
    """

    @abstractmethod
    async def execute_code(
        self,
        code: str,
        language: str = "python",
        working_directory: Optional[str] = None,
        timeout: float = 60.0,
    ) -> tuple[str, bool]:
        """
        Execute code.

        Args:
            code: Code to execute.
            language: Programming language.
            working_directory: Directory to execute in.
            timeout: Execution timeout.

        Returns:
            Tuple of (output, success).
        """
        ...

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read a file."""
        ...

    @abstractmethod
    async def write_file(self, path: str, content: str) -> bool:
        """Write a file."""
        ...

    @abstractmethod
    async def list_files(
        self,
        path: str,
        pattern: Optional[str] = None,
    ) -> list[str]:
        """List files in directory."""
        ...
