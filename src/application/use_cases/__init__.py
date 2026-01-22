"""Use cases for the application layer."""

# Use cases are primarily implemented in the services
# This module provides simplified interfaces for common operations

from typing import Optional
from src.application.services.orchestrator import Orchestrator
from src.application.dto import (
    CreateSessionDTO,
    SessionStatusDTO,
    ProcessQuestionDTO,
    QuestionResultDTO,
)


class CreateSessionUseCase:
    """Use case for creating a session."""

    def __init__(self, orchestrator: Orchestrator):
        self._orchestrator = orchestrator

    async def execute(self, dto: CreateSessionDTO) -> str:
        """Execute the use case."""
        from src.core.config.settings import LLMProvider

        provider_map = {
            "openai": LLMProvider.OPENAI,
            "deepseek": LLMProvider.DEEPSEEK,
            "claude": LLMProvider.CLAUDE,
        }

        provider = provider_map.get(dto.provider.lower())
        if not provider:
            raise ValueError(f"Unknown provider: {dto.provider}")

        return await self._orchestrator.create_session(
            provider=provider,
            name=dto.name,
            initial_prompt=dto.initial_prompt,
            working_directory=dto.working_directory,
            group_id=dto.group_id,
        )


class GetSessionStatusUseCase:
    """Use case for getting session status."""

    def __init__(self, orchestrator: Orchestrator):
        self._orchestrator = orchestrator

    def execute(self, session_id: str) -> Optional[SessionStatusDTO]:
        """Execute the use case."""
        status = self._orchestrator.get_session_status(session_id)
        if not status:
            return None

        metrics = status.get("metrics", {})

        return SessionStatusDTO(
            session_id=status["session_id"],
            name=status.get("name", ""),
            provider=status["provider"],
            status=status["status"],
            model=status.get("model", ""),
            questions_processed=metrics.get("questions_processed", 0),
            predictions_made=metrics.get("predictions_made", 0),
            prediction_accuracy=metrics.get("prediction_accuracy", 0.0),
            current_strategy=status.get("current_strategy", "default"),
            created_at=status.get("created_at"),
            last_activity_at=status.get("last_activity_at"),
        )


class ProcessQuestionUseCase:
    """Use case for processing a question."""

    def __init__(self, orchestrator: Orchestrator):
        self._orchestrator = orchestrator

    async def execute(self, dto: ProcessQuestionDTO) -> QuestionResultDTO:
        """Execute the use case."""
        import time
        from src.domain.entities.question import QuestionType

        type_map = {
            "other": QuestionType.OTHER,
            "initial": QuestionType.INITIAL,
            "followup": QuestionType.FOLLOWUP,
            "error": QuestionType.ERROR,
            "feature": QuestionType.FEATURE,
        }

        start = time.time()

        result = await self._orchestrator.process_question(
            dto.session_id,
            dto.content,
            type_map.get(dto.question_type, QuestionType.OTHER),
        )

        processing_time = time.time() - start

        return QuestionResultDTO(
            question_id=result.question_id,
            session_id=dto.session_id,
            answer=result.answer,
            processing_time=processing_time,
            used_prediction=result.used_prediction,
            prediction_id=result.prediction_id,
            prediction_accuracy=result.prediction_accuracy,
        )
