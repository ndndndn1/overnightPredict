"""Accuracy score value object."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AccuracyLevel(str, Enum):
    """Qualitative accuracy levels."""

    EXCELLENT = "excellent"  # >= 0.9
    GOOD = "good"  # >= 0.7
    MODERATE = "moderate"  # >= 0.5
    POOR = "poor"  # >= 0.3
    VERY_POOR = "very_poor"  # < 0.3


@dataclass(frozen=True)
class AccuracyScore:
    """
    Immutable accuracy score value object.

    Represents semantic similarity between predicted and actual questions.
    """

    value: float
    predicted: str
    actual: str
    method: str = "cosine_similarity"

    def __post_init__(self) -> None:
        """Validate the score value."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Accuracy score must be between 0 and 1, got {self.value}")

    @property
    def level(self) -> AccuracyLevel:
        """Get qualitative accuracy level."""
        if self.value >= 0.9:
            return AccuracyLevel.EXCELLENT
        elif self.value >= 0.7:
            return AccuracyLevel.GOOD
        elif self.value >= 0.5:
            return AccuracyLevel.MODERATE
        elif self.value >= 0.3:
            return AccuracyLevel.POOR
        else:
            return AccuracyLevel.VERY_POOR

    @property
    def is_acceptable(self) -> bool:
        """Check if accuracy is acceptable (>= 0.6)."""
        return self.value >= 0.6

    @property
    def requires_strategy_change(self) -> bool:
        """Check if accuracy is low enough to warrant strategy change."""
        return self.value < 0.5

    @property
    def percentage(self) -> float:
        """Get accuracy as percentage."""
        return self.value * 100

    def __str__(self) -> str:
        """String representation."""
        return f"{self.percentage:.1f}% ({self.level.value})"

    def __repr__(self) -> str:
        """Debug representation."""
        return f"AccuracyScore(value={self.value:.3f}, level={self.level.value})"


@dataclass(frozen=True)
class AccuracyWindow:
    """Rolling window of accuracy scores for trend analysis."""

    scores: tuple[float, ...]
    window_size: int

    def __post_init__(self) -> None:
        """Validate window."""
        if len(self.scores) > self.window_size:
            object.__setattr__(self, "scores", self.scores[-self.window_size:])

    @property
    def average(self) -> float:
        """Get average accuracy."""
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def trend(self) -> Optional[str]:
        """Determine accuracy trend."""
        if len(self.scores) < 3:
            return None

        recent = self.scores[-3:]
        if recent[-1] > recent[0] + 0.1:
            return "improving"
        elif recent[-1] < recent[0] - 0.1:
            return "declining"
        return "stable"

    @property
    def is_consistently_poor(self) -> bool:
        """Check if accuracy is consistently poor."""
        if len(self.scores) < 3:
            return False
        return all(s < 0.5 for s in self.scores[-3:])

    def add_score(self, score: float) -> "AccuracyWindow":
        """Add a new score and return new window."""
        new_scores = self.scores + (score,)
        return AccuracyWindow(scores=new_scores, window_size=self.window_size)
