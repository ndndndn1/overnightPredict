"""Base exception classes for the Overnight Predict system."""

from typing import Any, Optional
from datetime import datetime


class OvernightPredictError(Exception):
    """Base exception for all Overnight Predict errors."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()

    def __str__(self) -> str:
        base = f"{self.__class__.__name__}: {self.message}"
        if self.details:
            base += f" | Details: {self.details}"
        if self.cause:
            base += f" | Caused by: {self.cause}"
        return base

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
            "timestamp": self.timestamp.isoformat(),
        }


class ConfigurationError(OvernightPredictError):
    """Raised when there's a configuration issue."""

    pass


class LLMProviderError(OvernightPredictError):
    """Base exception for LLM provider errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, details, cause)
        self.provider = provider
        self.details["provider"] = provider


class RateLimitError(LLMProviderError):
    """Raised when rate limit is reached."""

    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: Optional[float] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, provider, details, cause)
        self.retry_after = retry_after
        self.details["retry_after"] = retry_after


class SessionError(OvernightPredictError):
    """Raised when there's a session-related error."""

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, details, cause)
        self.session_id = session_id
        if session_id:
            self.details["session_id"] = session_id


class PredictionError(OvernightPredictError):
    """Raised when prediction fails."""

    pass


class ContextSharingError(OvernightPredictError):
    """Raised when context sharing fails."""

    def __init__(
        self,
        message: str,
        sharing_type: str,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, details, cause)
        self.sharing_type = sharing_type
        self.details["sharing_type"] = sharing_type


class EvaluationError(OvernightPredictError):
    """Raised when evaluation fails."""

    pass


class StrategyError(OvernightPredictError):
    """Raised when strategy-related operations fail."""

    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, details, cause)
        self.strategy_name = strategy_name
        if strategy_name:
            self.details["strategy_name"] = strategy_name
