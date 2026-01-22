"""Session entity - represents a coding session with an LLM provider."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.core.utils.id_generator import generate_session_id


class SessionStatus(str, Enum):
    """Status of a session."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_RATE_LIMIT = "waiting_rate_limit"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SessionMetrics:
    """Metrics for a session."""

    questions_processed: int = 0
    predictions_made: int = 0
    predictions_accurate: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_used: int = 0
    total_api_calls: int = 0
    strategy_changes: int = 0

    @property
    def prediction_accuracy(self) -> float:
        """Calculate prediction accuracy."""
        if self.predictions_made == 0:
            return 0.0
        return self.predictions_accurate / self.predictions_made

    @property
    def task_success_rate(self) -> float:
        """Calculate task success rate."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 0.0
        return self.tasks_completed / total


@dataclass
class Session:
    """
    Represents a coding session with an LLM provider.

    A session maintains state across multiple interactions and tracks
    predictions, questions, and execution history.
    """

    id: str = field(default_factory=generate_session_id)
    name: str = ""
    provider: str = ""  # LLM provider name (openai, deepseek, claude)
    status: SessionStatus = SessionStatus.INITIALIZING

    # Configuration
    model: str = ""
    initial_prompt: str = ""
    working_directory: str = ""

    # Group for context sharing
    group_id: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None

    # Rate limiting
    rate_limited_until: Optional[datetime] = None

    # Current state
    current_task_id: Optional[str] = None
    current_prediction_id: Optional[str] = None
    current_strategy: str = "default"

    # History
    question_ids: list[str] = field(default_factory=list)
    prediction_ids: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)

    # Metrics
    metrics: SessionMetrics = field(default_factory=SessionMetrics)

    # Error tracking
    last_error: Optional[str] = None
    error_count: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Start the session."""
        self.status = SessionStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.last_activity_at = datetime.utcnow()

    def pause(self) -> None:
        """Pause the session."""
        self.status = SessionStatus.PAUSED
        self.last_activity_at = datetime.utcnow()

    def resume(self) -> None:
        """Resume the session."""
        if self.status == SessionStatus.PAUSED:
            self.status = SessionStatus.RUNNING
            self.last_activity_at = datetime.utcnow()

    def complete(self) -> None:
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.last_activity_at = datetime.utcnow()

    def fail(self, error: str) -> None:
        """Mark session as failed."""
        self.status = SessionStatus.FAILED
        self.last_error = error
        self.error_count += 1
        self.completed_at = datetime.utcnow()
        self.last_activity_at = datetime.utcnow()

    def cancel(self) -> None:
        """Cancel the session."""
        self.status = SessionStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.last_activity_at = datetime.utcnow()

    def set_rate_limited(self, until: datetime) -> None:
        """Mark session as rate limited."""
        self.status = SessionStatus.WAITING_RATE_LIMIT
        self.rate_limited_until = until
        self.last_activity_at = datetime.utcnow()

    def clear_rate_limit(self) -> None:
        """Clear rate limit status."""
        if self.status == SessionStatus.WAITING_RATE_LIMIT:
            self.status = SessionStatus.RUNNING
            self.rate_limited_until = None
            self.last_activity_at = datetime.utcnow()

    def add_question(self, question_id: str) -> None:
        """Add a question to the session history."""
        self.question_ids.append(question_id)
        self.metrics.questions_processed += 1
        self.last_activity_at = datetime.utcnow()

    def add_prediction(self, prediction_id: str) -> None:
        """Add a prediction to the session history."""
        self.prediction_ids.append(prediction_id)
        self.metrics.predictions_made += 1
        self.last_activity_at = datetime.utcnow()

    def add_task(self, task_id: str) -> None:
        """Add a task to the session history."""
        self.task_ids.append(task_id)
        self.last_activity_at = datetime.utcnow()

    def record_accurate_prediction(self) -> None:
        """Record an accurate prediction."""
        self.metrics.predictions_accurate += 1

    def record_task_completion(self, success: bool) -> None:
        """Record task completion."""
        if success:
            self.metrics.tasks_completed += 1
        else:
            self.metrics.tasks_failed += 1

    def change_strategy(self, strategy_name: str) -> None:
        """Change the prediction strategy."""
        self.current_strategy = strategy_name
        self.metrics.strategy_changes += 1
        self.last_activity_at = datetime.utcnow()

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity_at = datetime.utcnow()

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status in (
            SessionStatus.RUNNING,
            SessionStatus.WAITING_RATE_LIMIT,
        )

    @property
    def is_terminal(self) -> bool:
        """Check if session is in a terminal state."""
        return self.status in (
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.CANCELLED,
        )

    @property
    def duration(self) -> Optional[float]:
        """Get session duration in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "status": self.status.value,
            "model": self.model,
            "group_id": self.group_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_strategy": self.current_strategy,
            "metrics": {
                "questions_processed": self.metrics.questions_processed,
                "predictions_made": self.metrics.predictions_made,
                "prediction_accuracy": self.metrics.prediction_accuracy,
                "tasks_completed": self.metrics.tasks_completed,
                "task_success_rate": self.metrics.task_success_rate,
            },
            "last_error": self.last_error,
            "error_count": self.error_count,
        }
