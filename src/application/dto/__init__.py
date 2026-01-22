"""Data Transfer Objects for the application layer."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class CreateSessionDTO:
    """DTO for creating a session."""

    provider: str
    name: str = ""
    initial_prompt: str = ""
    working_directory: str = ""
    group_id: Optional[str] = None


@dataclass
class SessionStatusDTO:
    """DTO for session status."""

    session_id: str
    name: str
    provider: str
    status: str
    model: str
    questions_processed: int
    predictions_made: int
    prediction_accuracy: float
    current_strategy: str
    created_at: datetime
    last_activity_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class ProcessQuestionDTO:
    """DTO for processing a question."""

    session_id: str
    content: str
    question_type: str = "other"
    context: Optional[str] = None


@dataclass
class QuestionResultDTO:
    """DTO for question result."""

    question_id: str
    session_id: str
    answer: str
    processing_time: float
    used_prediction: bool
    prediction_accuracy: Optional[float] = None


@dataclass
class GroupStatusDTO:
    """DTO for group status."""

    group_id: str
    session_count: int
    sessions: list[SessionStatusDTO]
    total_questions: int
    total_predictions: int
    average_accuracy: float


@dataclass
class PredictionDTO:
    """DTO for prediction information."""

    prediction_id: str
    predicted_question: str
    confidence: float
    strategy: str
    status: str
    actual_question: Optional[str] = None
    accuracy: Optional[float] = None
