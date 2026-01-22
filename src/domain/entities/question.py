"""Question entity - represents a question or task input."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.core.utils.id_generator import generate_question_id


class QuestionType(str, Enum):
    """Type of question."""

    INITIAL = "initial"  # Initial user requirement
    FOLLOWUP = "followup"  # Follow-up question
    CLARIFICATION = "clarification"  # Clarification request
    ERROR = "error"  # Error/bug report
    FEATURE = "feature"  # Feature request
    REFACTOR = "refactor"  # Refactoring request
    REVIEW = "review"  # Code review request
    TEST = "test"  # Testing request
    DOCUMENTATION = "documentation"  # Documentation request
    OTHER = "other"


class QuestionSource(str, Enum):
    """Source of the question."""

    USER = "user"  # Direct user input
    PREDICTED = "predicted"  # From prediction
    SYSTEM = "system"  # System-generated
    DERIVED = "derived"  # Derived from analysis


@dataclass
class Question:
    """
    Represents a question or task input.

    Questions can come from users, predictions, or system analysis.
    """

    id: str = field(default_factory=generate_question_id)
    session_id: str = ""

    # Content
    content: str = ""
    context: Optional[str] = None  # Additional context
    expected_output: Optional[str] = None  # Expected outcome if known

    # Classification
    question_type: QuestionType = QuestionType.OTHER
    source: QuestionSource = QuestionSource.USER

    # Linking
    parent_question_id: Optional[str] = None  # For follow-ups
    prediction_id: Optional[str] = None  # If matched from prediction

    # Processing
    processed: bool = False
    answer: Optional[str] = None
    answer_at: Optional[datetime] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Priority and ordering
    priority: int = 0  # Higher = more important
    sequence_number: int = 0  # Order in session

    # Embeddings (for semantic similarity)
    embedding: Optional[list[float]] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_processed(self, answer: str) -> None:
        """Mark question as processed with answer."""
        self.processed = True
        self.answer = answer
        self.answer_at = datetime.utcnow()

    def link_prediction(self, prediction_id: str) -> None:
        """Link question to a prediction."""
        self.prediction_id = prediction_id

    @property
    def is_from_prediction(self) -> bool:
        """Check if question originated from prediction."""
        return self.source == QuestionSource.PREDICTED

    @property
    def has_answer(self) -> bool:
        """Check if question has been answered."""
        return self.answer is not None

    @property
    def processing_time(self) -> Optional[float]:
        """Get processing time in seconds."""
        if not self.answer_at:
            return None
        return (self.answer_at - self.created_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert question to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "content": self.content,
            "question_type": self.question_type.value,
            "source": self.source.value,
            "processed": self.processed,
            "has_answer": self.has_answer,
            "created_at": self.created_at.isoformat(),
            "answer_at": self.answer_at.isoformat() if self.answer_at else None,
            "processing_time": self.processing_time,
            "priority": self.priority,
            "sequence_number": self.sequence_number,
        }


@dataclass
class QuestionQueue:
    """Queue of questions for a session."""

    session_id: str = ""
    questions: list[Question] = field(default_factory=list)

    def enqueue(self, question: Question) -> None:
        """Add question to queue."""
        question.session_id = self.session_id
        question.sequence_number = len(self.questions)
        self.questions.append(question)
        # Sort by priority (higher first)
        self.questions.sort(key=lambda q: (-q.priority, q.sequence_number))

    def dequeue(self) -> Optional[Question]:
        """Get next unprocessed question."""
        for question in self.questions:
            if not question.processed:
                return question
        return None

    def get_pending(self) -> list[Question]:
        """Get all pending questions."""
        return [q for q in self.questions if not q.processed]

    def get_processed(self) -> list[Question]:
        """Get all processed questions."""
        return [q for q in self.questions if q.processed]

    @property
    def pending_count(self) -> int:
        """Get count of pending questions."""
        return len(self.get_pending())

    @property
    def total_count(self) -> int:
        """Get total question count."""
        return len(self.questions)
