"""
MetaTuner service - Adjusts strategies based on performance.

Implements meta-cognition by monitoring prediction accuracy and
automatically adjusting strategies when performance degrades.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from src.core.config.settings import PredictionConfig
from src.core.utils.logging import get_logger
from src.domain.entities.prediction import Prediction
from src.domain.entities.session import Session
from src.domain.interfaces.event_bus import IEventBus, StrategyChangedEvent
from src.domain.interfaces.prediction_strategy import (
    IPredictionStrategy,
    IAdaptiveStrategy,
    StrategyPerformance,
)
from src.domain.value_objects.accuracy import AccuracyWindow


@dataclass
class TuningDecision:
    """Decision made by the meta-tuner."""

    action: str  # "keep", "adapt", "switch"
    current_strategy: str
    new_strategy: Optional[str] = None
    reasoning: str = ""
    parameters_changed: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MetaTuner:
    """
    MetaTuner service for automatic strategy optimization.

    Monitors prediction accuracy and makes decisions about:
    - Adapting current strategy parameters
    - Switching to a different strategy
    - Resetting strategies after persistent failures
    """

    def __init__(
        self,
        config: PredictionConfig,
        event_bus: Optional[IEventBus] = None,
    ):
        """Initialize the meta-tuner.

        Args:
            config: Prediction configuration.
            event_bus: Optional event bus for notifications.
        """
        self._config = config
        self._event_bus = event_bus
        self._logger = get_logger("meta_tuner")

        # Strategy registry
        self._strategies: dict[str, IPredictionStrategy] = {}
        self._performance: dict[str, StrategyPerformance] = {}

        # Accuracy tracking per session
        self._accuracy_windows: dict[str, AccuracyWindow] = {}

        # Decision history
        self._decisions: list[TuningDecision] = []

    def register_strategy(self, strategy: IPredictionStrategy) -> None:
        """Register a strategy for management.

        Args:
            strategy: Strategy to register.
        """
        self._strategies[strategy.name] = strategy
        self._performance[strategy.name] = StrategyPerformance(
            strategy_name=strategy.name
        )
        self._logger.info("Strategy registered", name=strategy.name)

    async def evaluate_and_tune(
        self,
        session: Session,
        recent_predictions: list[tuple[Prediction, float]],
    ) -> TuningDecision:
        """Evaluate performance and decide on tuning actions.

        Args:
            session: Current session.
            recent_predictions: Recent (prediction, accuracy_score) pairs.

        Returns:
            Tuning decision.
        """
        current_strategy = session.current_strategy

        # Update accuracy window
        accuracy_window = self._update_accuracy_window(
            session.id,
            [score for _, score in recent_predictions],
        )

        # Update strategy performance
        self._update_performance(current_strategy, recent_predictions)

        # Make decision
        decision = await self._make_decision(
            session,
            current_strategy,
            accuracy_window,
        )

        # Apply decision
        await self._apply_decision(session, decision)

        # Record decision
        self._decisions.append(decision)

        return decision

    def _update_accuracy_window(
        self,
        session_id: str,
        scores: list[float],
    ) -> AccuracyWindow:
        """Update accuracy window for a session."""
        if session_id not in self._accuracy_windows:
            self._accuracy_windows[session_id] = AccuracyWindow(
                scores=tuple(),
                window_size=self._config.evaluation_window,
            )

        window = self._accuracy_windows[session_id]
        for score in scores:
            window = window.add_score(score)

        self._accuracy_windows[session_id] = window
        return window

    def _update_performance(
        self,
        strategy_name: str,
        predictions: list[tuple[Prediction, float]],
    ) -> None:
        """Update strategy performance metrics."""
        if strategy_name not in self._performance:
            self._performance[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name
            )

        perf = self._performance[strategy_name]
        for prediction, score in predictions:
            perf.record_prediction(
                accurate=score >= self._config.min_accuracy_for_keep,
                latency_ms=0,  # Would be tracked from prediction timing
                tokens_used=0,  # Would be tracked from LLM response
            )

    async def _make_decision(
        self,
        session: Session,
        current_strategy: str,
        accuracy_window: AccuracyWindow,
    ) -> TuningDecision:
        """Make a tuning decision based on current state."""
        avg_accuracy = accuracy_window.average
        trend = accuracy_window.trend
        is_poor = accuracy_window.is_consistently_poor

        self._logger.debug(
            "Evaluating performance",
            session_id=session.id,
            strategy=current_strategy,
            accuracy=avg_accuracy,
            trend=trend,
        )

        # Decision logic

        # Case 1: Consistently poor - switch strategy
        if is_poor:
            new_strategy = self._find_alternative_strategy(current_strategy)
            if new_strategy:
                return TuningDecision(
                    action="switch",
                    current_strategy=current_strategy,
                    new_strategy=new_strategy,
                    reasoning=f"Consistently poor accuracy ({avg_accuracy:.1%}). Switching to {new_strategy}.",
                )

        # Case 2: Below threshold but not critical - adapt
        if avg_accuracy < self._config.min_accuracy_for_keep:
            strategy = self._strategies.get(current_strategy)
            if strategy and isinstance(strategy, IAdaptiveStrategy):
                return TuningDecision(
                    action="adapt",
                    current_strategy=current_strategy,
                    reasoning=f"Accuracy below threshold ({avg_accuracy:.1%} < {self._config.min_accuracy_for_keep:.1%}). Adapting parameters.",
                )

            # Can't adapt, try switching
            new_strategy = self._find_alternative_strategy(current_strategy)
            if new_strategy:
                return TuningDecision(
                    action="switch",
                    current_strategy=current_strategy,
                    new_strategy=new_strategy,
                    reasoning=f"Accuracy below threshold and strategy not adaptable. Switching to {new_strategy}.",
                )

        # Case 3: Declining trend - preemptive adaptation
        if trend == "declining" and avg_accuracy < 0.75:
            strategy = self._strategies.get(current_strategy)
            if strategy and isinstance(strategy, IAdaptiveStrategy):
                return TuningDecision(
                    action="adapt",
                    current_strategy=current_strategy,
                    reasoning=f"Declining accuracy trend detected ({trend}). Preemptive adaptation.",
                )

        # Case 4: Good performance - keep current
        return TuningDecision(
            action="keep",
            current_strategy=current_strategy,
            reasoning=f"Performance acceptable ({avg_accuracy:.1%}, {trend or 'stable'}).",
        )

    def _find_alternative_strategy(
        self,
        current_strategy: str,
    ) -> Optional[str]:
        """Find an alternative strategy to switch to."""
        # Get strategies with better performance
        alternatives = []

        for name, perf in self._performance.items():
            if name == current_strategy:
                continue

            # Consider strategy if:
            # - Has better historical accuracy
            # - Or hasn't been tried much yet
            if perf.total_predictions < 5:
                alternatives.append((name, 0.5))  # Give untried strategies a chance
            elif perf.accuracy > self._performance.get(current_strategy, StrategyPerformance(current_strategy)).accuracy:
                alternatives.append((name, perf.accuracy))

        if not alternatives:
            return None

        # Sort by accuracy and return best
        alternatives.sort(key=lambda x: -x[1])
        return alternatives[0][0]

    async def _apply_decision(
        self,
        session: Session,
        decision: TuningDecision,
    ) -> None:
        """Apply a tuning decision."""
        if decision.action == "keep":
            return

        if decision.action == "switch" and decision.new_strategy:
            session.change_strategy(decision.new_strategy)

            # Publish event
            if self._event_bus:
                await self._event_bus.publish(StrategyChangedEvent(
                    aggregate_id=session.id,
                    payload={
                        "from_strategy": decision.current_strategy,
                        "to_strategy": decision.new_strategy,
                        "reasoning": decision.reasoning,
                    },
                ))

            self._logger.info(
                "Strategy switched",
                session_id=session.id,
                from_strategy=decision.current_strategy,
                to_strategy=decision.new_strategy,
            )

        elif decision.action == "adapt":
            strategy = self._strategies.get(decision.current_strategy)
            if strategy and isinstance(strategy, IAdaptiveStrategy):
                # Get recent predictions for adaptation
                # In real implementation, this would get actual feedback
                await strategy.adapt([])

                self._performance[decision.current_strategy].record_adaptation()

                self._logger.info(
                    "Strategy adapted",
                    session_id=session.id,
                    strategy=decision.current_strategy,
                )

    def get_strategy_rankings(self) -> list[tuple[str, float]]:
        """Get strategies ranked by performance.

        Returns:
            List of (strategy_name, accuracy) sorted by accuracy.
        """
        rankings = [
            (name, perf.accuracy)
            for name, perf in self._performance.items()
        ]
        rankings.sort(key=lambda x: -x[1])
        return rankings

    def get_session_accuracy(self, session_id: str) -> Optional[float]:
        """Get current accuracy for a session."""
        window = self._accuracy_windows.get(session_id)
        if window:
            return window.average
        return None

    def reset_session(self, session_id: str) -> None:
        """Reset accuracy tracking for a session."""
        if session_id in self._accuracy_windows:
            del self._accuracy_windows[session_id]

    def get_decision_history(
        self,
        limit: int = 50,
    ) -> list[TuningDecision]:
        """Get recent tuning decisions."""
        return self._decisions[-limit:]
