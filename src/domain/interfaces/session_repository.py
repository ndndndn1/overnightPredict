"""Session repository interface."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from src.domain.entities.session import Session, SessionStatus
from src.domain.entities.prediction import Prediction, PredictionBatch
from src.domain.entities.question import Question
from src.domain.entities.task import Task


class ISessionRepository(ABC):
    """
    Interface for session persistence.

    Handles storage and retrieval of sessions and related entities.
    """

    # Session operations

    @abstractmethod
    async def save_session(self, session: Session) -> str:
        """
        Save or update a session.

        Args:
            session: Session to save.

        Returns:
            Session ID.
        """
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session or None if not found.
        """
        ...

    @abstractmethod
    async def get_sessions_by_status(
        self,
        status: SessionStatus,
        limit: int = 100,
    ) -> list[Session]:
        """
        Get sessions by status.

        Args:
            status: Status to filter by.
            limit: Maximum number of sessions.

        Returns:
            List of sessions.
        """
        ...

    @abstractmethod
    async def get_sessions_by_group(
        self,
        group_id: str,
    ) -> list[Session]:
        """
        Get sessions in a group.

        Args:
            group_id: Group identifier.

        Returns:
            List of sessions.
        """
        ...

    @abstractmethod
    async def get_active_sessions(self) -> list[Session]:
        """
        Get all active sessions.

        Returns:
            List of active sessions.
        """
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and related data.

        Args:
            session_id: Session to delete.

        Returns:
            True if deleted.
        """
        ...

    # Prediction operations

    @abstractmethod
    async def save_prediction(self, prediction: Prediction) -> str:
        """
        Save a prediction.

        Args:
            prediction: Prediction to save.

        Returns:
            Prediction ID.
        """
        ...

    @abstractmethod
    async def get_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """
        Get a prediction by ID.

        Args:
            prediction_id: Prediction identifier.

        Returns:
            Prediction or None.
        """
        ...

    @abstractmethod
    async def get_session_predictions(
        self,
        session_id: str,
        pending_only: bool = False,
        limit: int = 100,
    ) -> list[Prediction]:
        """
        Get predictions for a session.

        Args:
            session_id: Session identifier.
            pending_only: Only return pending predictions.
            limit: Maximum number of predictions.

        Returns:
            List of predictions.
        """
        ...

    @abstractmethod
    async def save_prediction_batch(self, batch: PredictionBatch) -> list[str]:
        """
        Save a batch of predictions.

        Args:
            batch: Prediction batch.

        Returns:
            List of prediction IDs.
        """
        ...

    # Question operations

    @abstractmethod
    async def save_question(self, question: Question) -> str:
        """
        Save a question.

        Args:
            question: Question to save.

        Returns:
            Question ID.
        """
        ...

    @abstractmethod
    async def get_question(self, question_id: str) -> Optional[Question]:
        """
        Get a question by ID.

        Args:
            question_id: Question identifier.

        Returns:
            Question or None.
        """
        ...

    @abstractmethod
    async def get_session_questions(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[Question]:
        """
        Get questions for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of questions.

        Returns:
            List of questions.
        """
        ...

    # Task operations

    @abstractmethod
    async def save_task(self, task: Task) -> str:
        """
        Save a task.

        Args:
            task: Task to save.

        Returns:
            Task ID.
        """
        ...

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.

        Args:
            task_id: Task identifier.

        Returns:
            Task or None.
        """
        ...

    @abstractmethod
    async def get_session_tasks(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[Task]:
        """
        Get tasks for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of tasks.

        Returns:
            List of tasks.
        """
        ...

    # Metrics and analytics

    @abstractmethod
    async def get_session_metrics(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Get comprehensive metrics for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Dictionary of metrics.
        """
        ...

    @abstractmethod
    async def get_strategy_performance(
        self,
        strategy_name: str,
        since: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Get performance metrics for a strategy.

        Args:
            strategy_name: Strategy name.
            since: Only include data since this time.

        Returns:
            Dictionary of performance metrics.
        """
        ...
