"""Task entity - represents a coding task to be executed."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.core.utils.id_generator import generate_task_id


class TaskStatus(str, Enum):
    """Status of a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TaskPriority(int, Enum):
    """Priority levels for tasks."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskType(str, Enum):
    """Type of task."""

    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    OTHER = "other"


@dataclass
class TaskResult:
    """Result of task execution."""

    success: bool = False
    output: str = ""
    error: Optional[str] = None
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    tokens_used: int = 0
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "files_modified": self.files_modified,
            "files_created": self.files_created,
            "tokens_used": self.tokens_used,
            "execution_time": self.execution_time,
        }


@dataclass
class Task:
    """
    Represents a coding task to be executed.

    Tasks are generated from questions and executed by workers.
    """

    id: str = field(default_factory=generate_task_id)
    session_id: str = ""
    question_id: str = ""  # Source question

    # Task details
    description: str = ""
    instructions: str = ""
    task_type: TaskType = TaskType.OTHER
    priority: TaskPriority = TaskPriority.MEDIUM

    # Status
    status: TaskStatus = TaskStatus.PENDING

    # Execution
    result: Optional[TaskResult] = None
    retry_count: int = 0
    max_retries: int = 3

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # Task IDs
    blocked_by: Optional[str] = None  # Task ID blocking this one

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Context
    context: Optional[str] = None  # Additional context for execution
    working_directory: Optional[str] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()

    def complete(self, result: TaskResult) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        self.result = result
        self.completed_at = datetime.utcnow()

    def fail(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.result = TaskResult(success=False, error=error)
        self.completed_at = datetime.utcnow()

    def cancel(self) -> None:
        """Cancel the task."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.utcnow()

    def block(self, blocked_by: str) -> None:
        """Block task by another task."""
        self.status = TaskStatus.BLOCKED
        self.blocked_by = blocked_by

    def unblock(self) -> None:
        """Unblock task."""
        if self.status == TaskStatus.BLOCKED:
            self.status = TaskStatus.PENDING
            self.blocked_by = None

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries

    def retry(self) -> None:
        """Prepare task for retry."""
        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.result = None
        self.started_at = None
        self.completed_at = None

    @property
    def is_terminal(self) -> bool:
        """Check if task is in terminal state."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )

    @property
    def execution_time(self) -> Optional[float]:
        """Get execution time in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "question_id": self.question_id,
            "description": self.description,
            "task_type": self.task_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time": self.execution_time,
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass
class TaskPlan:
    """A plan consisting of multiple tasks."""

    tasks: list[Task] = field(default_factory=list)
    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    total_estimated_tokens: int = 0

    def add_task(
        self,
        description: str,
        instructions: str,
        task_type: TaskType = TaskType.OTHER,
        priority: TaskPriority = TaskPriority.MEDIUM,
        depends_on: Optional[list[str]] = None,
    ) -> Task:
        """Add a task to the plan."""
        task = Task(
            session_id=self.session_id,
            description=description,
            instructions=instructions,
            task_type=task_type,
            priority=priority,
            depends_on=depends_on or [],
        )
        self.tasks.append(task)
        return task

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks that are ready to execute (no pending dependencies)."""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        ready = []
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                if all(dep_id in completed_ids for dep_id in task.depends_on):
                    ready.append(task)
        return ready

    def get_next_task(self) -> Optional[Task]:
        """Get next task to execute based on priority."""
        ready = self.get_ready_tasks()
        if not ready:
            return None
        # Sort by priority (higher first)
        ready.sort(key=lambda t: -t.priority.value)
        return ready[0]

    @property
    def is_complete(self) -> bool:
        """Check if all tasks are complete."""
        return all(t.is_terminal for t in self.tasks)

    @property
    def progress(self) -> float:
        """Get completion progress (0.0 to 1.0)."""
        if not self.tasks:
            return 1.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)
