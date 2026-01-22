"""Evaluator interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from src.domain.entities.prediction import Prediction
from src.domain.entities.question import Question
from src.domain.value_objects.accuracy import AccuracyScore


@dataclass
class EvaluationResult:
    """Result of evaluating a prediction against an actual question."""

    prediction: Prediction
    actual_question: Question
    accuracy_score: AccuracyScore

    # Detailed metrics
    semantic_similarity: float = 0.0
    keyword_overlap: float = 0.0
    intent_match: bool = False

    # Evaluation metadata
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    evaluation_method: str = "semantic"
    evaluation_time_ms: float = 0.0

    # Decision
    is_match: bool = False
    should_use_prediction: bool = False  # Use pre-computed answer?

    # Reasoning
    reasoning: str = ""

    @property
    def is_accurate(self) -> bool:
        """Check if prediction was accurate."""
        return self.accuracy_score.is_acceptable

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction_id": self.prediction.id,
            "question_id": self.actual_question.id,
            "accuracy": self.accuracy_score.value,
            "accuracy_level": self.accuracy_score.level.value,
            "semantic_similarity": self.semantic_similarity,
            "intent_match": self.intent_match,
            "is_match": self.is_match,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


@dataclass
class BatchEvaluationResult:
    """Result of evaluating multiple predictions."""

    results: list[EvaluationResult] = field(default_factory=list)
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    total_time_ms: float = 0.0

    @property
    def average_accuracy(self) -> float:
        """Get average accuracy."""
        if not self.results:
            return 0.0
        return sum(r.accuracy_score.value for r in self.results) / len(self.results)

    @property
    def match_rate(self) -> float:
        """Get match rate."""
        if not self.results:
            return 0.0
        matches = sum(1 for r in self.results if r.is_match)
        return matches / len(self.results)

    @property
    def accurate_count(self) -> int:
        """Get count of accurate predictions."""
        return sum(1 for r in self.results if r.is_accurate)

    def add_result(self, result: EvaluationResult) -> None:
        """Add an evaluation result."""
        self.results.append(result)


class IEvaluator(ABC):
    """
    Interface for prediction evaluators (Critic).

    Evaluates the accuracy of predictions against actual questions.
    """

    @property
    @abstractmethod
    def evaluation_method(self) -> str:
        """Get the evaluation method name."""
        ...

    @abstractmethod
    async def evaluate(
        self,
        prediction: Prediction,
        actual_question: Question,
        threshold: float = 0.6,
    ) -> EvaluationResult:
        """
        Evaluate a single prediction.

        Args:
            prediction: The prediction to evaluate.
            actual_question: The actual question that occurred.
            threshold: Similarity threshold for matching.

        Returns:
            Evaluation result.
        """
        ...

    @abstractmethod
    async def evaluate_batch(
        self,
        predictions: list[Prediction],
        actual_questions: list[Question],
        threshold: float = 0.6,
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple predictions.

        Args:
            predictions: List of predictions.
            actual_questions: List of actual questions.
            threshold: Similarity threshold for matching.

        Returns:
            Batch evaluation result.
        """
        ...

    @abstractmethod
    async def compute_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score (0.0 to 1.0).
        """
        ...

    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        ...

    async def initialize(self) -> None:
        """Initialize evaluator resources."""
        pass

    async def cleanup(self) -> None:
        """Cleanup evaluator resources."""
        pass


class IIntentClassifier(ABC):
    """
    Interface for intent classification.

    Classifies the intent behind questions for better matching.
    """

    @abstractmethod
    async def classify(self, text: str) -> tuple[str, float]:
        """
        Classify the intent of text.

        Args:
            text: Text to classify.

        Returns:
            Tuple of (intent_label, confidence).
        """
        ...

    @abstractmethod
    async def compare_intents(
        self,
        text1: str,
        text2: str,
    ) -> bool:
        """
        Compare if two texts have the same intent.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            True if intents match.
        """
        ...
