"""Prediction entity - represents a predicted question or task."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.core.utils.id_generator import generate_prediction_id


class PredictionStatus(str, Enum):
    """Status of a prediction."""

    PENDING = "pending"  # Prediction made, waiting for actual question
    MATCHED = "matched"  # Prediction matched actual question
    UNMATCHED = "unmatched"  # Prediction did not match
    EXPIRED = "expired"  # Prediction window expired
    INVALIDATED = "invalidated"  # Strategy changed, prediction invalidated


@dataclass
class Prediction:
    """
    Represents a predicted question or task.

    The Forecaster generates predictions about what questions/tasks
    will come next based on the current context.
    """

    id: str = field(default_factory=generate_prediction_id)
    session_id: str = ""

    # Predicted content
    predicted_question: str = ""
    predicted_answer: Optional[str] = None  # Pre-computed answer if confidence is high
    predicted_context: Optional[str] = None  # Context that led to this prediction

    # Prediction metadata
    strategy_used: str = "default"
    confidence: float = 0.0  # 0.0 to 1.0
    reasoning: Optional[str] = None  # Why this prediction was made

    # Status and evaluation
    status: PredictionStatus = PredictionStatus.PENDING
    actual_question: Optional[str] = None
    similarity_score: Optional[float] = None  # Semantic similarity when evaluated

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    evaluated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Position in prediction sequence
    sequence_number: int = 0  # Order in batch of predictions
    lookahead_position: int = 0  # How far ahead this prediction is

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def match(self, actual_question: str, similarity_score: float) -> None:
        """Mark prediction as matched."""
        self.status = PredictionStatus.MATCHED
        self.actual_question = actual_question
        self.similarity_score = similarity_score
        self.evaluated_at = datetime.utcnow()

    def unmatch(self, actual_question: str, similarity_score: float) -> None:
        """Mark prediction as unmatched."""
        self.status = PredictionStatus.UNMATCHED
        self.actual_question = actual_question
        self.similarity_score = similarity_score
        self.evaluated_at = datetime.utcnow()

    def expire(self) -> None:
        """Mark prediction as expired."""
        self.status = PredictionStatus.EXPIRED
        self.evaluated_at = datetime.utcnow()

    def invalidate(self) -> None:
        """Invalidate prediction (e.g., when strategy changes)."""
        self.status = PredictionStatus.INVALIDATED
        self.evaluated_at = datetime.utcnow()

    @property
    def is_evaluated(self) -> bool:
        """Check if prediction has been evaluated."""
        return self.status != PredictionStatus.PENDING

    @property
    def is_successful(self) -> bool:
        """Check if prediction was successful."""
        return self.status == PredictionStatus.MATCHED

    @property
    def evaluation_delay(self) -> Optional[float]:
        """Get time between creation and evaluation in seconds."""
        if not self.evaluated_at:
            return None
        return (self.evaluated_at - self.created_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "predicted_question": self.predicted_question,
            "predicted_answer": self.predicted_answer,
            "strategy_used": self.strategy_used,
            "confidence": self.confidence,
            "status": self.status.value,
            "actual_question": self.actual_question,
            "similarity_score": self.similarity_score,
            "created_at": self.created_at.isoformat(),
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
            "sequence_number": self.sequence_number,
            "lookahead_position": self.lookahead_position,
        }


@dataclass
class PredictionBatch:
    """A batch of predictions made together."""

    predictions: list[Prediction] = field(default_factory=list)
    session_id: str = ""
    strategy_used: str = "default"
    created_at: datetime = field(default_factory=datetime.utcnow)
    context_snapshot: Optional[str] = None

    def add(self, prediction: Prediction) -> None:
        """Add a prediction to the batch."""
        prediction.sequence_number = len(self.predictions)
        prediction.lookahead_position = len(self.predictions)
        prediction.session_id = self.session_id
        prediction.strategy_used = self.strategy_used
        self.predictions.append(prediction)

    def invalidate_all(self) -> None:
        """Invalidate all predictions in the batch."""
        for prediction in self.predictions:
            if prediction.status == PredictionStatus.PENDING:
                prediction.invalidate()

    def get_pending(self) -> list[Prediction]:
        """Get pending predictions."""
        return [p for p in self.predictions if p.status == PredictionStatus.PENDING]

    def get_next_pending(self) -> Optional[Prediction]:
        """Get the next pending prediction."""
        pending = self.get_pending()
        return pending[0] if pending else None

    @property
    def accuracy(self) -> float:
        """Calculate batch accuracy."""
        evaluated = [p for p in self.predictions if p.is_evaluated]
        if not evaluated:
            return 0.0
        matched = sum(1 for p in evaluated if p.is_successful)
        return matched / len(evaluated)

    @property
    def average_similarity(self) -> Optional[float]:
        """Calculate average similarity score."""
        scores = [
            p.similarity_score
            for p in self.predictions
            if p.similarity_score is not None
        ]
        return sum(scores) / len(scores) if scores else None
