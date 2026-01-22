"""
SessionManager service - Manages individual session lifecycle.

Coordinates the OODA loop (Forecast -> Execute -> Evaluate -> Tune)
for a single session.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class QuestionProcessingResult:
    """Result of processing a question through the OODA loop."""

    question_id: str
    answer: str
    used_prediction: bool = False
    prediction_id: Optional[str] = None
    prediction_accuracy: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question_id": self.question_id,
            "answer": self.answer,
            "used_prediction": self.used_prediction,
            "prediction_id": self.prediction_id,
            "prediction_accuracy": self.prediction_accuracy,
        }

from src.core.utils.logging import get_logger
from src.domain.entities.session import Session, SessionStatus
from src.domain.entities.prediction import Prediction, PredictionBatch
from src.domain.entities.question import Question, QuestionSource
from src.domain.entities.task import Task, TaskType, TaskPriority
from src.domain.interfaces.llm_provider import ILLMProvider
from src.domain.interfaces.session_repository import ISessionRepository
from src.domain.interfaces.event_bus import (
    IEventBus,
    SessionStartedEvent,
    SessionCompletedEvent,
    SessionFailedEvent,
    PredictionMadeEvent,
    PredictionEvaluatedEvent,
    RateLimitHitEvent,
)
from src.domain.interfaces.prediction_strategy import PredictionContext
from src.domain.value_objects.context import Context, ContextType, ContextSnapshot
from src.application.services.forecaster import Forecaster
from src.application.services.executor import Executor
from src.application.services.evaluator import Evaluator
from src.application.services.meta_tuner import MetaTuner


class SessionManager:
    """
    Manages the lifecycle and OODA loop for a single session.

    Coordinates:
    - Forecasting future questions
    - Executing tasks
    - Evaluating predictions
    - Tuning strategies
    """

    def __init__(
        self,
        session: Session,
        llm_provider: ILLMProvider,
        repository: ISessionRepository,
        forecaster: Forecaster,
        evaluator: Evaluator,
        meta_tuner: MetaTuner,
        event_bus: Optional[IEventBus] = None,
    ):
        """Initialize the session manager.

        Args:
            session: Session to manage.
            llm_provider: LLM provider for execution.
            repository: Session repository.
            forecaster: Forecaster service.
            evaluator: Evaluator service.
            meta_tuner: MetaTuner service.
            event_bus: Optional event bus.
        """
        self._session = session
        self._llm = llm_provider
        self._repository = repository
        self._forecaster = forecaster
        self._evaluator = evaluator
        self._meta_tuner = meta_tuner
        self._event_bus = event_bus

        self._executor = Executor(llm_provider, event_bus)
        self._logger = get_logger(
            "session_manager",
            session_id=session.id,
            provider=llm_provider.provider_name,
        )

        # Current state
        self._current_predictions: PredictionBatch = PredictionBatch()
        self._context_snapshot = ContextSnapshot(session_id=session.id)
        self._running = False
        self._run_task: Optional[asyncio.Task] = None

    @property
    def session(self) -> Session:
        """Get the managed session."""
        return self._session

    @property
    def is_running(self) -> bool:
        """Check if session is running."""
        return self._running

    async def start(self) -> None:
        """Start the session."""
        if self._running:
            return

        self._running = True
        self._session.start()
        await self._repository.save_session(self._session)

        if self._event_bus:
            await self._event_bus.publish(SessionStartedEvent(
                aggregate_id=self._session.id,
                payload={"provider": self._session.provider},
            ))

        self._logger.info("Session started")

        # Start the main loop
        self._run_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the session gracefully."""
        self._running = False

        if self._run_task:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass

        self._session.complete()
        await self._repository.save_session(self._session)

        if self._event_bus:
            await self._event_bus.publish(SessionCompletedEvent(
                aggregate_id=self._session.id,
                payload={
                    "questions_processed": self._session.metrics.questions_processed,
                    "prediction_accuracy": self._session.metrics.prediction_accuracy,
                },
            ))

        self._logger.info("Session stopped")

    async def pause(self) -> None:
        """Pause the session."""
        self._session.pause()
        await self._repository.save_session(self._session)
        self._logger.info("Session paused")

    async def resume(self) -> None:
        """Resume a paused session."""
        if self._session.status == SessionStatus.PAUSED:
            self._session.resume()
            await self._repository.save_session(self._session)
            self._logger.info("Session resumed")

    async def process_question(self, question: Question) -> "QuestionProcessingResult":
        """Process a single question through the OODA loop.

        Args:
            question: Question to process.

        Returns:
            QuestionProcessingResult with answer and metadata.
        """
        self._logger.info(
            "Processing question",
            question_id=question.id,
            content=question.content[:100],
        )

        # 1. OBSERVE: Check predictions
        matched_prediction = await self._check_predictions(question)

        # 2. ORIENT: Update context
        await self._update_context(question)

        # 3. DECIDE: Use prediction or execute fresh
        used_prediction = False
        prediction_id = None
        prediction_accuracy = None

        if matched_prediction and matched_prediction.predicted_answer:
            answer = matched_prediction.predicted_answer
            used_prediction = True
            prediction_id = matched_prediction.id
            prediction_accuracy = matched_prediction.similarity_score
            self._logger.info("Using predicted answer", prediction_id=matched_prediction.id)
        else:
            # Execute fresh
            answer = await self._executor.execute_question(question, self._context_snapshot.to_string())

        # 4. ACT: Update state and forecast next
        self._session.add_question(question.id)
        await self._repository.save_question(question)
        await self._repository.save_session(self._session)

        # Generate new predictions
        await self._forecast_next()

        return QuestionProcessingResult(
            question_id=question.id,
            answer=answer,
            used_prediction=used_prediction,
            prediction_id=prediction_id,
            prediction_accuracy=prediction_accuracy,
        )

    async def add_context(self, content: str, context_type: ContextType) -> None:
        """Add context to the session.

        Args:
            content: Context content.
            context_type: Type of context.
        """
        context = Context(
            content=content,
            context_type=context_type,
            source=self._session.id,
        )
        self._context_snapshot.add(context)

    async def _run_loop(self) -> None:
        """Main OODA loop."""
        try:
            # Initial forecast
            await self._forecast_next()

            while self._running:
                # Check for rate limiting
                is_limited, retry_after = await self._llm.check_rate_limit()
                if is_limited:
                    await self._handle_rate_limit(retry_after or 60.0)
                    continue

                # Wait for next iteration
                await asyncio.sleep(1.0)

                # Update session activity
                self._session.update_activity()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._logger.error("Session loop failed", error=str(e))
            self._session.fail(str(e))

            if self._event_bus:
                await self._event_bus.publish(SessionFailedEvent(
                    aggregate_id=self._session.id,
                    payload={"error": str(e)},
                ))

    async def _check_predictions(self, question: Question) -> Optional[Prediction]:
        """Check if any prediction matches the question.

        Args:
            question: Actual question received.

        Returns:
            Matched prediction or None.
        """
        pending = self._current_predictions.get_pending()
        if not pending:
            return None

        for prediction in pending:
            result = await self._evaluator.evaluate(
                prediction,
                question,
                threshold=self._meta_tuner._config.accuracy_threshold,
            )

            if result.is_match:
                question.link_prediction(prediction.id)
                self._session.record_accurate_prediction()

                if self._event_bus:
                    await self._event_bus.publish(PredictionEvaluatedEvent(
                        aggregate_id=prediction.id,
                        payload={
                            "session_id": self._session.id,
                            "accuracy": result.accuracy_score.value,
                            "matched": True,
                        },
                    ))

                # Evaluate and tune
                await self._meta_tuner.evaluate_and_tune(
                    self._session,
                    [(prediction, result.accuracy_score.value)],
                )

                await self._repository.save_prediction(prediction)
                return prediction

        # No match - record for tuning
        if pending:
            await self._meta_tuner.evaluate_and_tune(
                self._session,
                [(p, 0.0) for p in pending[:3]],  # Evaluate top 3 predictions
            )

        return None

    async def _update_context(self, question: Question) -> None:
        """Update context with the new question."""
        context = Context(
            content=f"User Question: {question.content}",
            context_type=ContextType.CONVERSATION,
            source=question.id,
        )
        self._context_snapshot.add(context)

    async def _forecast_next(self) -> None:
        """Generate predictions for future questions."""
        # Build prediction context
        questions = await self._repository.get_session_questions(self._session.id, limit=10)
        predictions = await self._repository.get_session_predictions(self._session.id, limit=10)

        context = PredictionContext(
            session_id=self._session.id,
            context_snapshot=self._context_snapshot,
            previous_questions=[q.content for q in questions],
            previous_predictions=predictions,
            recent_accuracy=self._meta_tuner.get_session_accuracy(self._session.id) or 0.5,
            current_task=self._session.current_task_id,
        )

        # Generate predictions
        self._current_predictions = await self._forecaster.forecast(
            context,
            strategy_name=self._session.current_strategy,
        )

        # Save predictions
        for prediction in self._current_predictions.predictions:
            self._session.add_prediction(prediction.id)
            await self._repository.save_prediction(prediction)

        if self._event_bus and self._current_predictions.predictions:
            await self._event_bus.publish(PredictionMadeEvent(
                aggregate_id=self._session.id,
                payload={
                    "count": len(self._current_predictions.predictions),
                    "strategy": self._session.current_strategy,
                },
            ))

        self._logger.debug(
            "Forecast complete",
            predictions=len(self._current_predictions.predictions),
        )

    async def _handle_rate_limit(self, retry_after: float) -> None:
        """Handle rate limiting."""
        self._session.set_rate_limited(
            datetime.utcnow()
        )
        await self._repository.save_session(self._session)

        if self._event_bus:
            await self._event_bus.publish(RateLimitHitEvent(
                aggregate_id=self._session.id,
                payload={
                    "retry_after": retry_after,
                    "provider": self._session.provider,
                },
            ))

        self._logger.info("Rate limited, waiting", retry_after=retry_after)

        # Wait for rate limit
        await self._llm.wait_for_rate_limit()

        self._session.clear_rate_limit()
        await self._repository.save_session(self._session)

    def get_status(self) -> dict[str, Any]:
        """Get current session status."""
        return {
            "session_id": self._session.id,
            "status": self._session.status.value,
            "provider": self._session.provider,
            "model": self._session.model,
            "metrics": {
                "questions_processed": self._session.metrics.questions_processed,
                "predictions_made": self._session.metrics.predictions_made,
                "prediction_accuracy": self._session.metrics.prediction_accuracy,
                "tasks_completed": self._session.metrics.tasks_completed,
            },
            "current_strategy": self._session.current_strategy,
            "pending_predictions": len(self._current_predictions.get_pending()),
        }
