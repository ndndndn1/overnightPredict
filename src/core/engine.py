"""
OvernightEngine - The core autonomous coding engine.

This engine orchestrates the entire overnight coding process, managing:
- Question prediction and validation
- Answer generation
- Accuracy evaluation and strategy adjustment
- Code generation
- Session management
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

import structlog

from src.core.config import Settings, get_settings
from src.core.models import (
    AccuracyMetrics,
    Answer,
    CodeArtifact,
    Prediction,
    PredictionResult,
    PredictionStrategy,
    ProjectContext,
    Question,
    QuestionType,
    SessionState,
    SessionStatus,
)

if TYPE_CHECKING:
    from src.evaluators.accuracy import AccuracyEvaluator
    from src.generators.answer import AnswerGenerator
    from src.generators.code import CodeGenerator
    from src.predictors.question import QuestionPredictor
    from src.sessions.checkpoint import CheckpointManager
    from src.strategies.manager import StrategyManager

logger = structlog.get_logger(__name__)


class OvernightEngine:
    """
    Core engine for autonomous overnight coding.

    Implements the main loop:
    1. Predict upcoming questions based on current context
    2. Generate answers for predicted questions
    3. When actual question arrives, compare with predictions
    4. Evaluate accuracy and adjust strategy if needed
    5. Generate code based on answers
    6. Continue autonomously
    """

    def __init__(
        self,
        settings: Settings | None = None,
        question_predictor: "QuestionPredictor | None" = None,
        answer_generator: "AnswerGenerator | None" = None,
        accuracy_evaluator: "AccuracyEvaluator | None" = None,
        strategy_manager: "StrategyManager | None" = None,
        code_generator: "CodeGenerator | None" = None,
        checkpoint_manager: "CheckpointManager | None" = None,
    ) -> None:
        """Initialize the overnight engine."""
        self.settings = settings or get_settings()
        self._question_predictor = question_predictor
        self._answer_generator = answer_generator
        self._accuracy_evaluator = accuracy_evaluator
        self._strategy_manager = strategy_manager
        self._code_generator = code_generator
        self._checkpoint_manager = checkpoint_manager

        # State
        self._sessions: dict[str, SessionState] = {}
        self._is_running = False
        self._shutdown_event = asyncio.Event()

        # Callbacks
        self._on_question_predicted: list[Callable[[Prediction], Any]] = []
        self._on_answer_generated: list[Callable[[Answer], Any]] = []
        self._on_code_generated: list[Callable[[CodeArtifact], Any]] = []
        self._on_strategy_adjusted: list[Callable[[PredictionStrategy, PredictionStrategy], Any]] = []

        logger.info("OvernightEngine initialized", settings=self.settings.environment)

    @property
    def question_predictor(self) -> "QuestionPredictor":
        """Get the question predictor (lazy initialization)."""
        if self._question_predictor is None:
            from src.predictors.question import QuestionPredictor
            self._question_predictor = QuestionPredictor(self.settings)
        return self._question_predictor

    @property
    def answer_generator(self) -> "AnswerGenerator":
        """Get the answer generator (lazy initialization)."""
        if self._answer_generator is None:
            from src.generators.answer import AnswerGenerator
            self._answer_generator = AnswerGenerator(self.settings)
        return self._answer_generator

    @property
    def accuracy_evaluator(self) -> "AccuracyEvaluator":
        """Get the accuracy evaluator (lazy initialization)."""
        if self._accuracy_evaluator is None:
            from src.evaluators.accuracy import AccuracyEvaluator
            self._accuracy_evaluator = AccuracyEvaluator(self.settings)
        return self._accuracy_evaluator

    @property
    def strategy_manager(self) -> "StrategyManager":
        """Get the strategy manager (lazy initialization)."""
        if self._strategy_manager is None:
            from src.strategies.manager import StrategyManager
            self._strategy_manager = StrategyManager(self.settings)
        return self._strategy_manager

    @property
    def code_generator(self) -> "CodeGenerator":
        """Get the code generator (lazy initialization)."""
        if self._code_generator is None:
            from src.generators.code import CodeGenerator
            self._code_generator = CodeGenerator(self.settings)
        return self._code_generator

    @property
    def checkpoint_manager(self) -> "CheckpointManager":
        """Get the checkpoint manager (lazy initialization)."""
        if self._checkpoint_manager is None:
            from src.sessions.checkpoint import CheckpointManager
            self._checkpoint_manager = CheckpointManager()
        return self._checkpoint_manager

    async def create_session(
        self,
        topic: str,
        project_context: ProjectContext | None = None,
        initial_context: dict[str, Any] | None = None,
    ) -> SessionState:
        """
        Create a new coding session.

        Args:
            topic: The main topic/goal for this session
            project_context: Optional project context for enterprise projects
            initial_context: Optional initial context dictionary

        Returns:
            The created session state
        """
        session = SessionState(
            topic=topic,
            status=SessionStatus.INITIALIZING,
            current_context=initial_context or {},
        )

        # Initialize accuracy metrics
        session.accuracy_metrics = AccuracyMetrics(
            session_id=session.id,
            strategy=PredictionStrategy(self.settings.prediction.initial_strategy),
        )

        # Store project context if provided
        if project_context:
            session.current_context["project"] = project_context.model_dump()

        self._sessions[session.id] = session

        logger.info(
            "Session created",
            session_id=session.id,
            topic=topic,
            has_project_context=project_context is not None,
        )

        return session

    async def start_session(self, session_id: str) -> None:
        """
        Start an autonomous coding session.

        This begins the main prediction-generation loop.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        session.status = SessionStatus.RUNNING
        self._is_running = True

        logger.info("Starting session", session_id=session_id)

        try:
            await self._run_session_loop(session)
        except Exception as e:
            logger.error("Session error", session_id=session_id, error=str(e))
            session.status = SessionStatus.FAILED
            raise
        finally:
            if session.status == SessionStatus.RUNNING:
                session.status = SessionStatus.COMPLETED
                session.completed_at = datetime.now(timezone.utc)

    async def _run_session_loop(self, session: SessionState) -> None:
        """
        Main session loop implementing the prediction-validation cycle.

        Loop steps:
        1. Predict next questions based on current context
        2. Pre-generate answers for predictions
        3. Wait for actual question or timeout
        4. Compare actual vs predicted, evaluate accuracy
        5. Adjust strategy if accuracy is low
        6. Generate code from answers
        7. Update context and continue
        """
        while self._is_running and session.status == SessionStatus.RUNNING:
            try:
                # Step 1: Predict upcoming questions
                predictions = await self._predict_questions(session)
                session.pending_predictions.extend(predictions)

                for prediction in predictions:
                    await self._notify_prediction(prediction)

                # Step 2: Pre-generate answers for predictions
                for prediction in predictions:
                    if prediction.predicted_answer is None:
                        answer_content = await self._pre_generate_answer(session, prediction)
                        prediction.predicted_answer = answer_content

                # Step 3: Wait for actual question (simulated or from external source)
                actual_question = await self._wait_for_question(session)

                if actual_question is None:
                    # No question received, continue with predictions
                    await self._process_predictions_as_actual(session)
                else:
                    # Step 4: Compare and evaluate
                    await self._evaluate_predictions(session, actual_question)

                    # Step 5: Check if strategy adjustment needed
                    await self._check_strategy_adjustment(session)

                    # Step 6: Generate answer and code
                    answer = await self._generate_answer(session, actual_question)
                    session.add_answer(answer)
                    await self._notify_answer(answer)

                    # Generate code if applicable
                    if self._should_generate_code(actual_question, answer):
                        artifact = await self._generate_code(session, actual_question, answer)
                        session.add_artifact(artifact)
                        await self._notify_code(artifact)

                # Step 7: Update context
                await self._update_context(session)

                # Checkpoint
                await self._checkpoint_session(session)

            except asyncio.CancelledError:
                logger.info("Session cancelled", session_id=session.id)
                break
            except Exception as e:
                logger.error("Error in session loop", session_id=session.id, error=str(e))
                await asyncio.sleep(1)  # Brief pause before retry

    async def _predict_questions(self, session: SessionState) -> list[Prediction]:
        """Predict upcoming questions based on current context."""
        predictions = await self.question_predictor.predict(
            context=session.current_context,
            history=session.questions,
            strategy=session.active_strategy,
            count=self.settings.prediction.lookahead_count,
        )

        logger.info(
            "Questions predicted",
            session_id=session.id,
            count=len(predictions),
            strategy=session.active_strategy,
        )

        return predictions

    async def _pre_generate_answer(
        self, session: SessionState, prediction: Prediction
    ) -> str:
        """Pre-generate an answer for a predicted question."""
        return await self.answer_generator.generate_preview(
            predicted_question=prediction.predicted_question,
            question_type=prediction.question_type,
            context=session.current_context,
        )

    async def _wait_for_question(
        self, session: SessionState, timeout: float = 30.0
    ) -> Question | None:
        """
        Wait for an actual question to arrive.

        In autonomous mode, this may generate questions based on project needs.
        """
        # In autonomous mode, derive next question from project context
        if session.current_context.get("autonomous_mode", True):
            return await self._derive_next_question(session)

        # Otherwise wait for external input (with timeout)
        session.status = SessionStatus.WAITING_INPUT
        try:
            # This would be replaced with actual input mechanism
            await asyncio.sleep(timeout)
            return None
        finally:
            session.status = SessionStatus.RUNNING

    async def _derive_next_question(self, session: SessionState) -> Question:
        """Derive the next logical question based on project state."""
        project_data = session.current_context.get("project", {})
        pending_components = project_data.get("pending_components", [])

        if pending_components:
            component = pending_components[0]
            question = Question(
                content=f"How should I implement the {component} component?",
                question_type=QuestionType.IMPLEMENTATION,
                context={"component": component, "project": project_data},
            )
        else:
            # Ask about next phase or optimization
            question = Question(
                content="What should be the next step in the project?",
                question_type=QuestionType.ARCHITECTURE,
                context={"project": project_data},
            )

        session.add_question(question)
        return question

    async def _evaluate_predictions(
        self, session: SessionState, actual_question: Question
    ) -> list[PredictionResult]:
        """Evaluate how well predictions matched the actual question."""
        results = []

        for prediction in session.pending_predictions:
            result = await self.accuracy_evaluator.evaluate(
                prediction=prediction,
                actual_question=actual_question,
            )
            results.append(result)
            session.prediction_results.append(result)

            # Update accuracy metrics
            if session.accuracy_metrics:
                session.accuracy_metrics.update_accuracy(
                    is_accurate=result.is_accurate,
                    similarity=result.similarity_score,
                )

        # Clear pending predictions
        session.pending_predictions = []

        logger.info(
            "Predictions evaluated",
            session_id=session.id,
            total=len(results),
            accurate=sum(1 for r in results if r.is_accurate),
        )

        return results

    async def _check_strategy_adjustment(self, session: SessionState) -> None:
        """Check if prediction strategy needs adjustment based on accuracy."""
        if session.accuracy_metrics is None:
            return

        metrics = session.accuracy_metrics
        threshold = self.settings.prediction.accuracy_threshold
        min_samples = self.settings.prediction.min_samples_for_adjustment

        if metrics.total_predictions < min_samples:
            return  # Not enough data yet

        if metrics.accuracy_rate < threshold:
            # Accuracy too low, adjust strategy
            old_strategy = session.active_strategy
            new_strategy = await self.strategy_manager.adjust_strategy(
                current_strategy=old_strategy,
                metrics=metrics,
                context=session.current_context,
            )

            if new_strategy != old_strategy:
                session.active_strategy = new_strategy
                metrics.strategy_adjustments += 1

                # Reset metrics for new strategy
                session.accuracy_metrics = AccuracyMetrics(
                    session_id=session.id,
                    strategy=new_strategy,
                    strategy_adjustments=metrics.strategy_adjustments,
                )

                logger.info(
                    "Strategy adjusted",
                    session_id=session.id,
                    old_strategy=old_strategy,
                    new_strategy=new_strategy,
                    previous_accuracy=metrics.accuracy_rate,
                )

                await self._notify_strategy_change(old_strategy, new_strategy)

    async def _generate_answer(
        self, session: SessionState, question: Question
    ) -> Answer:
        """Generate an answer for a question."""
        # Check if we have a pre-generated answer from predictions
        for prediction in session.pending_predictions:
            if prediction.predicted_answer:
                similarity = await self.accuracy_evaluator.compute_similarity(
                    prediction.predicted_question, question.content
                )
                if similarity > self.settings.prediction.accuracy_threshold:
                    # Use pre-generated answer
                    return Answer(
                        question_id=question.id,
                        content=prediction.predicted_answer,
                        confidence=similarity,
                    )

        # Generate new answer
        return await self.answer_generator.generate(
            question=question,
            context=session.current_context,
            history=session.answers,
        )

    def _should_generate_code(self, question: Question, answer: Answer) -> bool:
        """Determine if code should be generated for this Q&A pair."""
        code_related_types = {
            QuestionType.IMPLEMENTATION,
            QuestionType.DEBUGGING,
            QuestionType.OPTIMIZATION,
        }
        return (
            question.question_type in code_related_types
            or bool(answer.code_snippets)
        )

    async def _generate_code(
        self,
        session: SessionState,
        question: Question,
        answer: Answer,
    ) -> CodeArtifact:
        """Generate code artifact from the Q&A pair."""
        return await self.code_generator.generate(
            question=question,
            answer=answer,
            context=session.current_context,
            existing_artifacts=session.artifacts,
        )

    async def _process_predictions_as_actual(self, session: SessionState) -> None:
        """Process predictions as actual questions when no input received."""
        if not session.pending_predictions:
            return

        # Take the highest confidence prediction
        prediction = max(session.pending_predictions, key=lambda p: p.confidence)

        # Convert to actual question
        question = Question(
            content=prediction.predicted_question,
            question_type=prediction.question_type,
            context=prediction.context_snapshot,
        )

        session.add_question(question)

        # Generate answer (use pre-generated if available)
        if prediction.predicted_answer:
            answer = Answer(
                question_id=question.id,
                content=prediction.predicted_answer,
                confidence=prediction.confidence,
            )
        else:
            answer = await self.answer_generator.generate(
                question=question,
                context=session.current_context,
                history=session.answers,
            )

        session.add_answer(answer)

        # Generate code if applicable
        if self._should_generate_code(question, answer):
            artifact = await self._generate_code(session, question, answer)
            session.add_artifact(artifact)
            await self._notify_code(artifact)

        # Clear predictions
        session.pending_predictions = []

        logger.info(
            "Processed prediction as actual",
            session_id=session.id,
            question=question.content[:50],
        )

    async def _update_context(self, session: SessionState) -> None:
        """Update session context based on recent activity."""
        # Add recent questions and answers to context
        recent_qa = []
        for q, a in zip(
            session.questions[-5:],
            session.answers[-5:],
            strict=False,
        ):
            recent_qa.append({"question": q.content, "answer": a.content[:500]})

        session.current_context["recent_qa"] = recent_qa

        # Update component status if applicable
        project_data = session.current_context.get("project", {})
        if project_data:
            completed = project_data.get("completed_components", [])
            pending = project_data.get("pending_components", [])

            # Move first pending to completed if we just implemented it
            if pending and session.artifacts:
                latest_artifact = session.artifacts[-1]
                if latest_artifact.is_valid:
                    component = pending.pop(0)
                    completed.append(component)
                    project_data["completed_components"] = completed
                    project_data["pending_components"] = pending
                    session.current_context["project"] = project_data

    async def _checkpoint_session(self, session: SessionState) -> None:
        """Save session checkpoint for recovery."""
        session.checkpoint_version += 1

        # Persist to storage using checkpoint manager
        checkpoint_path = await self.checkpoint_manager.save_checkpoint(
            session=session,
            sync=False,  # Async save for performance
        )

        logger.debug(
            "Session checkpointed",
            session_id=session.id,
            version=session.checkpoint_version,
            path=checkpoint_path,
        )

    async def recover_session(self, session_id: str) -> SessionState | None:
        """Recover a session from the latest checkpoint."""
        session = await self.checkpoint_manager.load_checkpoint(session_id)

        if session:
            self._sessions[session.id] = session
            logger.info(
                "Session recovered from checkpoint",
                session_id=session.id,
                version=session.checkpoint_version,
            )

        return session

    async def stop_session(self, session_id: str) -> None:
        """Stop a running session."""
        session = self._sessions.get(session_id)
        if session:
            session.status = SessionStatus.PAUSED
            logger.info("Session stopped", session_id=session_id)

    async def shutdown(self) -> None:
        """Shutdown the engine gracefully."""
        self._is_running = False
        self._shutdown_event.set()

        for session in self._sessions.values():
            if session.status == SessionStatus.RUNNING:
                session.status = SessionStatus.PAUSED

        logger.info("OvernightEngine shutdown complete")

    # Event registration methods
    def on_question_predicted(
        self, callback: Callable[[Prediction], Any]
    ) -> None:
        """Register callback for question predictions."""
        self._on_question_predicted.append(callback)

    def on_answer_generated(self, callback: Callable[[Answer], Any]) -> None:
        """Register callback for answer generation."""
        self._on_answer_generated.append(callback)

    def on_code_generated(self, callback: Callable[[CodeArtifact], Any]) -> None:
        """Register callback for code generation."""
        self._on_code_generated.append(callback)

    def on_strategy_adjusted(
        self, callback: Callable[[PredictionStrategy, PredictionStrategy], Any]
    ) -> None:
        """Register callback for strategy adjustments."""
        self._on_strategy_adjusted.append(callback)

    # Notification helpers
    async def _notify_prediction(self, prediction: Prediction) -> None:
        """Notify callbacks about a new prediction."""
        for callback in self._on_question_predicted:
            try:
                result = callback(prediction)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Prediction callback error", error=str(e))

    async def _notify_answer(self, answer: Answer) -> None:
        """Notify callbacks about a generated answer."""
        for callback in self._on_answer_generated:
            try:
                result = callback(answer)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Answer callback error", error=str(e))

    async def _notify_code(self, artifact: CodeArtifact) -> None:
        """Notify callbacks about generated code."""
        for callback in self._on_code_generated:
            try:
                result = callback(artifact)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Code callback error", error=str(e))

    async def _notify_strategy_change(
        self, old: PredictionStrategy, new: PredictionStrategy
    ) -> None:
        """Notify callbacks about strategy changes."""
        for callback in self._on_strategy_adjusted:
            try:
                result = callback(old, new)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Strategy callback error", error=str(e))

    # Session accessors
    def get_session(self, session_id: str) -> SessionState | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def get_all_sessions(self) -> list[SessionState]:
        """Get all sessions."""
        return list(self._sessions.values())

    def get_session_metrics(self, session_id: str) -> dict[str, Any]:
        """Get metrics for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return {}

        return {
            "session_id": session.id,
            "status": session.status,
            "topic": session.topic,
            "questions_count": len(session.questions),
            "answers_count": len(session.answers),
            "artifacts_count": len(session.artifacts),
            "accuracy_metrics": (
                session.accuracy_metrics.model_dump()
                if session.accuracy_metrics
                else None
            ),
            "active_strategy": session.active_strategy,
            "started_at": session.started_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
        }
