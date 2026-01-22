"""
Forecaster service (Prefrontal Cortex) - Predicts future questions.

Analyzes context to predict what questions or tasks will come next,
enabling proactive preparation and caching.
"""

import time
from datetime import datetime, timedelta
from typing import Any, Optional

from src.core.config.settings import PredictionConfig
from src.core.utils.id_generator import generate_prediction_id
from src.core.utils.logging import get_logger
from src.domain.entities.prediction import Prediction, PredictionBatch
from src.domain.interfaces.llm_provider import ILLMProvider, LLMMessage
from src.domain.interfaces.prediction_strategy import (
    IPredictionStrategy,
    IAdaptiveStrategy,
    PredictionContext,
    StrategyMetadata,
)
from src.domain.value_objects.context import ContextSnapshot


class LLMPredictionStrategy(IAdaptiveStrategy):
    """
    Prediction strategy using LLM to generate predictions.

    Uses the LLM to analyze context and predict future questions.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        config: PredictionConfig,
    ):
        """Initialize the strategy.

        Args:
            llm_provider: LLM provider for generation.
            config: Prediction configuration.
        """
        self._llm = llm_provider
        self._config = config
        self._logger = get_logger("strategy.llm")

        # Adaptive parameters
        self._temperature = 0.7
        self._lookahead_count = config.lookahead_count
        self._prompt_template = self._get_default_prompt_template()

    @property
    def metadata(self) -> StrategyMetadata:
        """Get strategy metadata."""
        return StrategyMetadata(
            name="llm_prediction",
            description="Uses LLM to predict future questions based on context",
            version="1.0.0",
            typical_accuracy=0.65,
            typical_latency_ms=2000,
            requires_history=True,
            min_context_length=50,
            max_lookahead=10,
            default_params={
                "temperature": 0.7,
                "lookahead_count": 5,
            },
        )

    @property
    def name(self) -> str:
        """Get strategy name."""
        return "llm_prediction"

    def can_handle(self, context: PredictionContext) -> bool:
        """Check if strategy can handle the context."""
        if not context.context_snapshot.contexts:
            return False
        return context.context_snapshot.total_length >= self.metadata.min_context_length

    async def predict(
        self,
        context: PredictionContext,
        **params: Any,
    ) -> PredictionBatch:
        """Generate predictions for future questions."""
        start_time = time.time()

        batch = PredictionBatch(
            session_id=context.session_id,
            strategy_used=self.name,
            context_snapshot=context.context_snapshot.to_string()[:1000],
        )

        lookahead = params.get("lookahead_count", self._lookahead_count)
        temperature = params.get("temperature", self._temperature)

        try:
            # Build prompt
            prompt = self._build_prediction_prompt(context, lookahead)

            # Generate predictions using LLM
            response = await self._llm.complete(
                messages=[
                    LLMMessage(
                        role="system",
                        content=self._prompt_template,
                    ),
                    LLMMessage(
                        role="user",
                        content=prompt,
                    ),
                ],
                temperature=temperature,
                max_tokens=2000,
            )

            # Parse predictions from response
            predictions = self._parse_predictions(
                response.content,
                context.session_id,
            )

            # Add to batch with expiration
            for pred in predictions[:lookahead]:
                pred.expires_at = datetime.utcnow() + timedelta(minutes=30)
                batch.add(pred)

            latency = (time.time() - start_time) * 1000
            self._logger.debug(
                "Predictions generated",
                count=len(batch.predictions),
                latency_ms=latency,
            )

        except Exception as e:
            self._logger.error("Prediction failed", error=str(e))
            # Return empty batch on error
            pass

        return batch

    async def predict_single(
        self,
        context: PredictionContext,
        position: int = 0,
        **params: Any,
    ) -> Prediction:
        """Generate a single prediction."""
        batch = await self.predict(context, lookahead_count=position + 1, **params)

        if batch.predictions and len(batch.predictions) > position:
            return batch.predictions[position]

        # Return empty prediction if generation failed
        return Prediction(
            session_id=context.session_id,
            predicted_question="",
            strategy_used=self.name,
            confidence=0.0,
        )

    async def adapt(
        self,
        feedback: list[tuple[Prediction, float]],
    ) -> None:
        """Adapt strategy based on feedback."""
        if not feedback:
            return

        # Calculate average accuracy
        avg_accuracy = sum(score for _, score in feedback) / len(feedback)

        # Adjust temperature based on accuracy
        if avg_accuracy < 0.4:
            # Low accuracy: increase temperature for more diversity
            self._temperature = min(1.0, self._temperature + 0.1)
            self._logger.info("Increased temperature", new_temp=self._temperature)
        elif avg_accuracy > 0.8:
            # High accuracy: decrease temperature for consistency
            self._temperature = max(0.3, self._temperature - 0.1)
            self._logger.info("Decreased temperature", new_temp=self._temperature)

        # Analyze patterns in failures
        failures = [(p, s) for p, s in feedback if s < 0.5]
        if len(failures) > len(feedback) * 0.6:
            # Most predictions failing - adjust lookahead
            self._lookahead_count = max(1, self._lookahead_count - 1)
            self._logger.info(
                "Reduced lookahead",
                new_lookahead=self._lookahead_count,
            )

    def get_parameters(self) -> dict[str, Any]:
        """Get current strategy parameters."""
        return {
            "temperature": self._temperature,
            "lookahead_count": self._lookahead_count,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set strategy parameters."""
        if "temperature" in params:
            self._temperature = float(params["temperature"])
        if "lookahead_count" in params:
            self._lookahead_count = int(params["lookahead_count"])

    def reset(self) -> None:
        """Reset strategy to default state."""
        self._temperature = 0.7
        self._lookahead_count = self._config.lookahead_count

    def _get_default_prompt_template(self) -> str:
        """Get the default system prompt template."""
        return """You are a predictive AI assistant that anticipates what questions or tasks will come next in a software development conversation.

Based on the context provided, predict the most likely follow-up questions or requests. Consider:
1. The logical next steps in the current task
2. Common follow-up patterns in software development
3. Potential clarifications or refinements needed
4. Likely errors or issues that might arise

For each prediction, provide:
- The predicted question or request
- A confidence score (0.0-1.0)
- Brief reasoning for the prediction

Format your response as a numbered list with each prediction on a new line:
1. [QUESTION] <predicted question> [CONFIDENCE] <0.0-1.0> [REASON] <brief reasoning>
2. [QUESTION] <predicted question> [CONFIDENCE] <0.0-1.0> [REASON] <brief reasoning>
..."""

    def _build_prediction_prompt(
        self,
        context: PredictionContext,
        lookahead: int,
    ) -> str:
        """Build the prediction prompt from context."""
        parts = []

        # Add context snapshot
        parts.append("## Current Context")
        parts.append(context.context_snapshot.to_string()[:2000])

        # Add previous questions
        if context.previous_questions:
            parts.append("\n## Previous Questions in this Session")
            for q in context.previous_questions[-5:]:
                parts.append(f"- {q}")

        # Add error context if present
        if context.error_context:
            parts.append(f"\n## Current Error/Issue\n{context.error_context}")

        # Add current task
        if context.current_task:
            parts.append(f"\n## Current Task\n{context.current_task}")

        # Request predictions
        parts.append(f"\n## Request")
        parts.append(f"Predict the next {lookahead} most likely questions or requests.")

        return "\n".join(parts)

    def _parse_predictions(
        self,
        response: str,
        session_id: str,
    ) -> list[Prediction]:
        """Parse predictions from LLM response."""
        predictions = []

        for line in response.split("\n"):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue

            try:
                # Parse format: N. [QUESTION] ... [CONFIDENCE] ... [REASON] ...
                question = ""
                confidence = 0.5
                reasoning = ""

                if "[QUESTION]" in line:
                    q_start = line.find("[QUESTION]") + len("[QUESTION]")
                    q_end = line.find("[CONFIDENCE]") if "[CONFIDENCE]" in line else len(line)
                    question = line[q_start:q_end].strip()

                if "[CONFIDENCE]" in line:
                    c_start = line.find("[CONFIDENCE]") + len("[CONFIDENCE]")
                    c_end = line.find("[REASON]") if "[REASON]" in line else len(line)
                    try:
                        confidence = float(line[c_start:c_end].strip())
                    except ValueError:
                        confidence = 0.5

                if "[REASON]" in line:
                    r_start = line.find("[REASON]") + len("[REASON]")
                    reasoning = line[r_start:].strip()

                if question:
                    predictions.append(Prediction(
                        id=generate_prediction_id(),
                        session_id=session_id,
                        predicted_question=question,
                        strategy_used=self.name,
                        confidence=min(1.0, max(0.0, confidence)),
                        reasoning=reasoning,
                    ))

            except Exception:
                continue

        return predictions


class StrategyPerformanceTracker:
    """Tracks performance metrics for strategy selection."""

    def __init__(self):
        self.total_predictions: int = 0
        self.successful_predictions: int = 0
        self.total_latency_ms: float = 0.0
        self.context_type_scores: dict[str, list[float]] = {}

    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        if self.total_predictions == 0:
            return 0.5  # Default for untested strategies
        return self.successful_predictions / self.total_predictions

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_predictions == 0:
            return 0.0
        return self.total_latency_ms / self.total_predictions

    def record(self, accuracy: float, latency_ms: float, context_type: str = "general") -> None:
        """Record a prediction result."""
        self.total_predictions += 1
        if accuracy >= 0.5:
            self.successful_predictions += 1
        self.total_latency_ms += latency_ms

        if context_type not in self.context_type_scores:
            self.context_type_scores[context_type] = []
        self.context_type_scores[context_type].append(accuracy)

    def get_context_score(self, context_type: str) -> float:
        """Get average score for a specific context type."""
        scores = self.context_type_scores.get(context_type, [])
        if not scores:
            return 0.5  # Default
        return sum(scores) / len(scores)


class Forecaster:
    """
    Forecaster service managing prediction strategies.

    Coordinates multiple prediction strategies and selects
    the best one based on context and performance.
    """

    def __init__(
        self,
        config: PredictionConfig,
        default_strategy: Optional[IPredictionStrategy] = None,
    ):
        """Initialize the forecaster.

        Args:
            config: Prediction configuration.
            default_strategy: Optional default strategy.
        """
        self._config = config
        self._logger = get_logger("forecaster")
        self._strategies: dict[str, IPredictionStrategy] = {}
        self._default_strategy = default_strategy
        self._performance_tracker: dict[str, StrategyPerformanceTracker] = {}

    def register_strategy(
        self,
        strategy: IPredictionStrategy,
        set_default: bool = False,
    ) -> None:
        """Register a prediction strategy.

        Args:
            strategy: Strategy to register.
            set_default: Set as default strategy.
        """
        self._strategies[strategy.name] = strategy
        self._performance_tracker[strategy.name] = StrategyPerformanceTracker()
        if set_default or self._default_strategy is None:
            self._default_strategy = strategy

        self._logger.info(
            "Strategy registered",
            name=strategy.name,
            is_default=set_default,
        )

    def get_strategy(self, name: str) -> Optional[IPredictionStrategy]:
        """Get a strategy by name."""
        return self._strategies.get(name)

    async def forecast(
        self,
        context: PredictionContext,
        strategy_name: Optional[str] = None,
    ) -> PredictionBatch:
        """Generate predictions using specified or best strategy.

        Args:
            context: Context for prediction.
            strategy_name: Optional specific strategy to use.

        Returns:
            Batch of predictions.
        """
        # Select strategy
        if strategy_name:
            strategy = self._strategies.get(strategy_name)
            if not strategy:
                self._logger.warning(f"Strategy {strategy_name} not found, using default")
                strategy = self._default_strategy
        else:
            strategy = self._select_best_strategy(context)

        if not strategy:
            self._logger.error("No strategy available")
            return PredictionBatch(session_id=context.session_id)

        # Generate predictions
        self._logger.debug(
            "Forecasting",
            strategy=strategy.name,
            session_id=context.session_id,
        )

        batch = await strategy.predict(context)

        self._logger.info(
            "Forecast complete",
            predictions=len(batch.predictions),
            strategy=strategy.name,
        )

        return batch

    def _select_best_strategy(
        self,
        context: PredictionContext,
    ) -> Optional[IPredictionStrategy]:
        """Select the best strategy for the context.

        Considers:
        - Historical performance per strategy
        - Context characteristics (type, length)
        - Resource constraints (latency requirements)
        """
        # Filter strategies that can handle the context
        candidates = [
            s for s in self._strategies.values()
            if s.can_handle(context)
        ]

        if not candidates:
            return self._default_strategy

        if len(candidates) == 1:
            return candidates[0]

        # Determine context characteristics
        context_type = self._classify_context(context)
        context_length = context.context_snapshot.total_length
        requires_low_latency = context_length > 5000  # Large context = prefer faster strategy

        # Score each candidate strategy
        scored_candidates: list[tuple[IPredictionStrategy, float]] = []

        for strategy in candidates:
            score = self._calculate_strategy_score(
                strategy,
                context_type,
                requires_low_latency,
            )
            scored_candidates.append((strategy, score))

        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: -x[1])

        best_strategy = scored_candidates[0][0]

        self._logger.debug(
            "Strategy selected",
            selected=best_strategy.name,
            context_type=context_type,
            scores={s.name: f"{score:.2f}" for s, score in scored_candidates},
        )

        return best_strategy

    def _classify_context(self, context: PredictionContext) -> str:
        """Classify the context type for strategy selection."""
        # Analyze context to determine its type
        context_str = context.context_snapshot.to_string().lower()

        if context.error_context:
            return "error_debugging"
        elif any(kw in context_str for kw in ["test", "unittest", "pytest", "spec"]):
            return "testing"
        elif any(kw in context_str for kw in ["api", "endpoint", "route", "request", "response"]):
            return "api_development"
        elif any(kw in context_str for kw in ["refactor", "optimize", "clean", "improve"]):
            return "refactoring"
        elif any(kw in context_str for kw in ["bug", "fix", "error", "issue", "problem"]):
            return "bug_fixing"
        elif any(kw in context_str for kw in ["feature", "implement", "add", "create", "build"]):
            return "feature_development"
        else:
            return "general"

    def _calculate_strategy_score(
        self,
        strategy: IPredictionStrategy,
        context_type: str,
        requires_low_latency: bool,
    ) -> float:
        """Calculate a score for a strategy based on various factors."""
        tracker = self._performance_tracker.get(strategy.name)
        if not tracker:
            return 0.5  # Default score for untracked strategies

        # Base score from overall accuracy (weight: 40%)
        accuracy_score = tracker.accuracy * 0.4

        # Context-specific score (weight: 35%)
        context_score = tracker.get_context_score(context_type) * 0.35

        # Latency score (weight: 15%)
        # Lower latency is better, normalize to 0-1 range (assuming max 5000ms)
        if requires_low_latency and tracker.avg_latency_ms > 0:
            latency_score = max(0, 1 - (tracker.avg_latency_ms / 5000)) * 0.15
        else:
            latency_score = 0.15  # Full score if latency doesn't matter or unknown

        # Experience score - favor strategies with more data (weight: 10%)
        # More predictions = more reliable performance data
        experience_score = min(1.0, tracker.total_predictions / 50) * 0.1

        total_score = accuracy_score + context_score + latency_score + experience_score

        return total_score

    def record_prediction_result(
        self,
        strategy_name: str,
        accuracy: float,
        latency_ms: float,
        context_type: str = "general",
    ) -> None:
        """Record a prediction result for performance tracking.

        Args:
            strategy_name: Name of the strategy used.
            accuracy: Accuracy score (0.0-1.0).
            latency_ms: Prediction latency in milliseconds.
            context_type: Type of context for context-specific tracking.
        """
        tracker = self._performance_tracker.get(strategy_name)
        if tracker:
            tracker.record(accuracy, latency_ms, context_type)
            self._logger.debug(
                "Prediction result recorded",
                strategy=strategy_name,
                accuracy=accuracy,
                latency_ms=latency_ms,
                context_type=context_type,
            )

    def get_strategy_performance(self, strategy_name: str) -> Optional[dict[str, Any]]:
        """Get performance metrics for a strategy.

        Args:
            strategy_name: Name of the strategy.

        Returns:
            Dictionary with performance metrics or None.
        """
        tracker = self._performance_tracker.get(strategy_name)
        if not tracker:
            return None

        return {
            "accuracy": tracker.accuracy,
            "total_predictions": tracker.total_predictions,
            "avg_latency_ms": tracker.avg_latency_ms,
            "context_scores": {
                ctx: sum(scores) / len(scores) if scores else 0
                for ctx, scores in tracker.context_type_scores.items()
            },
        }

    async def adapt_strategy(
        self,
        strategy_name: str,
        feedback: list[tuple[Prediction, float]],
    ) -> None:
        """Adapt a strategy based on feedback.

        Args:
            strategy_name: Strategy to adapt.
            feedback: List of (prediction, accuracy_score) tuples.
        """
        strategy = self._strategies.get(strategy_name)

        if strategy and isinstance(strategy, IAdaptiveStrategy):
            await strategy.adapt(feedback)
            self._logger.info(
                "Strategy adapted",
                strategy=strategy_name,
                feedback_count=len(feedback),
            )
