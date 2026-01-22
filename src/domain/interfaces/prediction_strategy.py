"""Prediction strategy interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from src.domain.entities.prediction import Prediction, PredictionBatch
from src.domain.value_objects.context import ContextSnapshot


@dataclass
class StrategyMetadata:
    """Metadata about a prediction strategy."""

    name: str
    description: str
    version: str = "1.0.0"

    # Performance characteristics
    typical_accuracy: float = 0.0
    typical_latency_ms: float = 0.0
    token_efficiency: float = 1.0  # Lower is more efficient

    # Requirements
    requires_history: bool = True
    min_context_length: int = 100
    max_lookahead: int = 5

    # Parameters
    default_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "typical_accuracy": self.typical_accuracy,
            "typical_latency_ms": self.typical_latency_ms,
            "token_efficiency": self.token_efficiency,
        }


@dataclass
class PredictionContext:
    """Context for making predictions."""

    session_id: str
    context_snapshot: ContextSnapshot

    # History
    previous_questions: list[str] = field(default_factory=list)
    previous_predictions: list[Prediction] = field(default_factory=list)
    recent_accuracy: float = 0.0

    # Configuration
    lookahead_count: int = 5
    confidence_threshold: float = 0.6

    # Current state
    current_task: Optional[str] = None
    error_context: Optional[str] = None


class IPredictionStrategy(ABC):
    """
    Interface for prediction strategies (Strategy Pattern).

    Implementations provide different approaches to predicting
    future questions based on context.
    """

    @property
    @abstractmethod
    def metadata(self) -> StrategyMetadata:
        """Get strategy metadata."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        ...

    @abstractmethod
    async def predict(
        self,
        context: PredictionContext,
        **params: Any,
    ) -> PredictionBatch:
        """
        Generate predictions for future questions.

        Args:
            context: Context for making predictions.
            **params: Strategy-specific parameters.

        Returns:
            Batch of predictions.
        """
        ...

    @abstractmethod
    async def predict_single(
        self,
        context: PredictionContext,
        position: int = 0,
        **params: Any,
    ) -> Prediction:
        """
        Generate a single prediction.

        Args:
            context: Context for making predictions.
            position: Lookahead position.
            **params: Strategy-specific parameters.

        Returns:
            A single prediction.
        """
        ...

    @abstractmethod
    def can_handle(self, context: PredictionContext) -> bool:
        """
        Check if strategy can handle the given context.

        Args:
            context: Context to check.

        Returns:
            True if strategy can handle the context.
        """
        ...

    async def initialize(self) -> None:
        """Initialize the strategy (optional)."""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources (optional)."""
        pass


class IAdaptiveStrategy(IPredictionStrategy):
    """
    Extended interface for adaptive strategies.

    These strategies can adjust their parameters based on feedback.
    """

    @abstractmethod
    async def adapt(
        self,
        feedback: list[tuple[Prediction, float]],  # (prediction, accuracy_score)
    ) -> None:
        """
        Adapt strategy based on feedback.

        Args:
            feedback: List of predictions and their accuracy scores.
        """
        ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Get current strategy parameters."""
        ...

    @abstractmethod
    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set strategy parameters."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset strategy to default state."""
        ...


@dataclass
class StrategyPerformance:
    """Track performance of a strategy."""

    strategy_name: str
    total_predictions: int = 0
    accurate_predictions: int = 0
    total_latency_ms: float = 0.0
    total_tokens_used: int = 0
    adaptations: int = 0
    last_used: Optional[datetime] = None

    @property
    def accuracy(self) -> float:
        """Get accuracy rate."""
        if self.total_predictions == 0:
            return 0.0
        return self.accurate_predictions / self.total_predictions

    @property
    def average_latency(self) -> float:
        """Get average latency in ms."""
        if self.total_predictions == 0:
            return 0.0
        return self.total_latency_ms / self.total_predictions

    def record_prediction(
        self,
        accurate: bool,
        latency_ms: float,
        tokens_used: int,
    ) -> None:
        """Record a prediction result."""
        self.total_predictions += 1
        if accurate:
            self.accurate_predictions += 1
        self.total_latency_ms += latency_ms
        self.total_tokens_used += tokens_used
        self.last_used = datetime.utcnow()

    def record_adaptation(self) -> None:
        """Record a strategy adaptation."""
        self.adaptations += 1
