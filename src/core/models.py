"""Core data models for OvernightPredict."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class QuestionType(str, Enum):
    """Types of questions in the system."""

    CLARIFICATION = "clarification"
    IMPLEMENTATION = "implementation"
    ARCHITECTURE = "architecture"
    DEBUGGING = "debugging"
    TESTING = "testing"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"


class PredictionStrategy(str, Enum):
    """Available prediction strategies."""

    CONTEXT_BASED = "context_based"
    PATTERN_MATCHING = "pattern_matching"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HYBRID = "hybrid"


class SessionStatus(str, Enum):
    """Status of a coding session."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    FAILED = "failed"


class Question(BaseModel):
    """Represents a question in the system."""

    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    question_type: QuestionType
    context: dict[str, Any] = Field(default_factory=dict)
    parent_question_id: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Answer(BaseModel):
    """Represents an answer to a question."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_id: str
    content: str
    code_snippets: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    derived_questions: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Prediction(BaseModel):
    """Represents a predicted question."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    predicted_question: str
    question_type: QuestionType
    confidence: float = Field(ge=0.0, le=1.0)
    strategy_used: PredictionStrategy
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    predicted_answer: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)


class PredictionResult(BaseModel):
    """Result of comparing prediction with actual question."""

    prediction_id: str
    actual_question_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    is_accurate: bool
    strategy_used: PredictionStrategy
    feedback: str = ""
    timestamp: datetime = Field(default_factory=utcnow)


class AccuracyMetrics(BaseModel):
    """Metrics for tracking prediction accuracy."""

    session_id: str
    strategy: PredictionStrategy
    total_predictions: int = 0
    accurate_predictions: int = 0
    accuracy_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    average_similarity: float = Field(ge=0.0, le=1.0, default=0.0)
    strategy_adjustments: int = 0
    last_updated: datetime = Field(default_factory=utcnow)

    def update_accuracy(self, is_accurate: bool, similarity: float) -> None:
        """Update accuracy metrics with new prediction result."""
        self.total_predictions += 1
        if is_accurate:
            self.accurate_predictions += 1
        self.accuracy_rate = self.accurate_predictions / self.total_predictions
        # Moving average for similarity
        n = self.total_predictions
        self.average_similarity = ((n - 1) * self.average_similarity + similarity) / n
        self.last_updated = utcnow()


class StrategyConfig(BaseModel):
    """Configuration for a prediction strategy."""

    strategy: PredictionStrategy
    weight: float = Field(ge=0.0, le=1.0, default=0.25)
    parameters: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    performance_history: list[float] = Field(default_factory=list)

    def update_performance(self, score: float) -> None:
        """Update performance history with new score."""
        self.performance_history.append(score)
        # Keep only last 100 scores
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]


class CodeArtifact(BaseModel):
    """Represents a generated code artifact."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    file_path: str
    content: str
    language: str
    question_id: str | None = None
    version: int = 1
    is_valid: bool = True
    lint_errors: list[str] = Field(default_factory=list)
    test_results: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utcnow)


class SessionState(BaseModel):
    """State of a coding session."""

    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    status: SessionStatus = SessionStatus.INITIALIZING
    current_context: dict[str, Any] = Field(default_factory=dict)

    # Question and answer history
    questions: list[Question] = Field(default_factory=list)
    answers: list[Answer] = Field(default_factory=list)

    # Predictions
    pending_predictions: list[Prediction] = Field(default_factory=list)
    prediction_results: list[PredictionResult] = Field(default_factory=list)

    # Strategy and metrics
    active_strategy: PredictionStrategy = PredictionStrategy.CONTEXT_BASED
    accuracy_metrics: AccuracyMetrics | None = None

    # Code artifacts
    artifacts: list[CodeArtifact] = Field(default_factory=list)

    # Timing
    started_at: datetime = Field(default_factory=utcnow)
    last_activity: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None

    # Checkpointing
    checkpoint_version: int = 0

    def add_question(self, question: Question) -> None:
        """Add a question to the session."""
        self.questions.append(question)
        self.last_activity = utcnow()

    def add_answer(self, answer: Answer) -> None:
        """Add an answer to the session."""
        self.answers.append(answer)
        self.last_activity = utcnow()

    def add_prediction(self, prediction: Prediction) -> None:
        """Add a prediction to pending predictions."""
        self.pending_predictions.append(prediction)

    def add_artifact(self, artifact: CodeArtifact) -> None:
        """Add a code artifact to the session."""
        self.artifacts.append(artifact)
        self.last_activity = utcnow()


class ProjectContext(BaseModel):
    """Context for an enterprise project being built."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    target_languages: list[str] = Field(default_factory=list)
    architecture_type: str = "microservices"
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    current_phase: str = "planning"
    completed_components: list[str] = Field(default_factory=list)
    pending_components: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrchestratorState(BaseModel):
    """State of the session orchestrator."""

    active_sessions: dict[str, SessionState] = Field(default_factory=dict)
    completed_sessions: list[str] = Field(default_factory=list)
    project_context: ProjectContext | None = None
    global_metrics: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utcnow)
