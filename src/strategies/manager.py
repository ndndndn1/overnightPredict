"""Strategy management module for adaptive prediction strategy selection."""

from __future__ import annotations

import random
from typing import Any

import structlog

from src.core.config import Settings
from src.core.models import AccuracyMetrics, PredictionStrategy, StrategyConfig

logger = structlog.get_logger(__name__)


class StrategyManager:
    """
    Manages prediction strategies and handles adaptive strategy adjustment.

    Features:
    - Multi-strategy support
    - Performance-based strategy selection
    - Adaptive weight adjustment
    - Strategy recommendation based on context
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the strategy manager."""
        self.settings = settings

        # Initialize strategy configurations
        self._strategies: dict[PredictionStrategy, StrategyConfig] = {}
        self._initialize_strategies()

        # Track overall performance
        self._global_performance: dict[PredictionStrategy, list[float]] = {}
        for strategy in PredictionStrategy:
            self._global_performance[strategy] = []

    def _initialize_strategies(self) -> None:
        """Initialize all available strategies with default configurations."""
        strategy_defaults = {
            PredictionStrategy.CONTEXT_BASED: {
                "weight": 0.35,
                "parameters": {
                    "context_depth": 5,
                    "include_project_state": True,
                    "include_recent_qa": True,
                },
            },
            PredictionStrategy.PATTERN_MATCHING: {
                "weight": 0.25,
                "parameters": {
                    "min_pattern_confidence": 0.6,
                    "max_patterns": 10,
                    "use_transitions": True,
                },
            },
            PredictionStrategy.SEMANTIC_SIMILARITY: {
                "weight": 0.25,
                "parameters": {
                    "similarity_threshold": 0.7,
                    "max_candidates": 20,
                    "use_clustering": False,
                },
            },
            PredictionStrategy.HYBRID: {
                "weight": 0.15,
                "parameters": {
                    "strategy_weights": {
                        "context_based": 0.4,
                        "pattern_matching": 0.3,
                        "semantic_similarity": 0.3,
                    },
                    "voting_method": "weighted",
                },
            },
        }

        for strategy, defaults in strategy_defaults.items():
            self._strategies[strategy] = StrategyConfig(
                strategy=strategy,
                weight=defaults["weight"],
                parameters=defaults["parameters"],
            )

    async def adjust_strategy(
        self,
        current_strategy: PredictionStrategy,
        metrics: AccuracyMetrics,
        context: dict[str, Any],
    ) -> PredictionStrategy:
        """
        Adjust the prediction strategy based on performance metrics.

        Args:
            current_strategy: Currently active strategy
            metrics: Accuracy metrics for current strategy
            context: Current context for decision making

        Returns:
            Recommended new strategy (may be same as current)
        """
        logger.info(
            "Evaluating strategy adjustment",
            current_strategy=current_strategy,
            accuracy_rate=metrics.accuracy_rate,
            total_predictions=metrics.total_predictions,
        )

        # Update global performance tracking
        self._global_performance[current_strategy].append(metrics.accuracy_rate)

        # Record performance in strategy config
        self._strategies[current_strategy].update_performance(metrics.accuracy_rate)

        # Check if current strategy is performing well enough
        threshold = self.settings.prediction.accuracy_threshold
        if metrics.accuracy_rate >= threshold:
            logger.info(
                "Current strategy meeting threshold",
                strategy=current_strategy,
                accuracy=metrics.accuracy_rate,
            )
            return current_strategy

        # Current strategy underperforming, evaluate alternatives
        best_strategy = await self._select_best_strategy(
            current_strategy=current_strategy,
            context=context,
            exclude_current=metrics.strategy_adjustments > 0,
        )

        if best_strategy != current_strategy:
            logger.info(
                "Recommending strategy change",
                from_strategy=current_strategy,
                to_strategy=best_strategy,
                reason=self._get_change_reason(current_strategy, best_strategy, context),
            )

        return best_strategy

    async def _select_best_strategy(
        self,
        current_strategy: PredictionStrategy,
        context: dict[str, Any],
        exclude_current: bool = False,
    ) -> PredictionStrategy:
        """
        Select the best strategy based on historical performance and context.

        Uses a combination of:
        - Historical performance data
        - Context suitability scoring
        - Exploration/exploitation balance
        """
        candidates = list(PredictionStrategy)
        if exclude_current:
            candidates = [s for s in candidates if s != current_strategy]

        # Score each candidate
        scores: dict[PredictionStrategy, float] = {}

        for strategy in candidates:
            score = await self._score_strategy(strategy, context)
            scores[strategy] = score

        # Apply exploration factor (occasionally try underused strategies)
        exploration_rate = self.settings.prediction.adaptation_rate
        if random.random() < exploration_rate:
            # Find least-used strategy
            usage_counts = {
                s: len(self._global_performance[s]) for s in candidates
            }
            least_used = min(usage_counts, key=lambda s: usage_counts[s])

            logger.info(
                "Exploration: trying least-used strategy",
                strategy=least_used,
            )
            return least_used

        # Return highest-scoring strategy
        best = max(scores, key=lambda s: scores[s])
        return best

    async def _score_strategy(
        self,
        strategy: PredictionStrategy,
        context: dict[str, Any],
    ) -> float:
        """
        Score a strategy based on historical performance and context fit.

        Args:
            strategy: Strategy to score
            context: Current context

        Returns:
            Score between 0 and 1
        """
        config = self._strategies[strategy]
        score = 0.0

        # Historical performance score (50% weight)
        if config.performance_history:
            recent_performance = config.performance_history[-10:]
            avg_performance = sum(recent_performance) / len(recent_performance)
            score += 0.5 * avg_performance
        else:
            # No history, use prior weight as estimate
            score += 0.5 * config.weight

        # Context suitability score (50% weight)
        context_score = self._compute_context_suitability(strategy, context)
        score += 0.5 * context_score

        return score

    def _compute_context_suitability(
        self,
        strategy: PredictionStrategy,
        context: dict[str, Any],
    ) -> float:
        """
        Compute how suitable a strategy is for the current context.

        Different strategies work better in different situations.
        """
        project = context.get("project", {})
        recent_qa = context.get("recent_qa", [])
        phase = project.get("current_phase", "implementation")

        suitability = 0.5  # Base suitability

        if strategy == PredictionStrategy.CONTEXT_BASED:
            # Better when project context is rich
            if project.get("pending_components"):
                suitability += 0.2
            if project.get("requirements"):
                suitability += 0.1
            if phase in ["planning", "architecture"]:
                suitability += 0.1

        elif strategy == PredictionStrategy.PATTERN_MATCHING:
            # Better with more history
            if len(recent_qa) >= 3:
                suitability += 0.2
            if phase in ["implementation", "debugging"]:
                suitability += 0.1
            # Better when in repetitive phases
            if self._detect_repetitive_pattern(recent_qa):
                suitability += 0.15

        elif strategy == PredictionStrategy.SEMANTIC_SIMILARITY:
            # Better with extensive history
            if len(recent_qa) >= 5:
                suitability += 0.2
            # Better for complex, varied questions
            if self._detect_question_variety(recent_qa):
                suitability += 0.15

        elif strategy == PredictionStrategy.HYBRID:
            # Generally good, especially with mixed contexts
            suitability += 0.1
            # Better when other strategies have mixed performance
            if self._has_mixed_performance():
                suitability += 0.2

        return min(1.0, suitability)

    def _detect_repetitive_pattern(self, recent_qa: list[dict[str, Any]]) -> bool:
        """Detect if recent Q&A shows repetitive patterns."""
        if len(recent_qa) < 3:
            return False

        questions = [qa.get("question", "").lower() for qa in recent_qa]

        # Check for similar question structures
        common_starts = ["how", "what", "why", "implement", "create"]
        start_counts: dict[str, int] = {}

        for q in questions:
            for start in common_starts:
                if q.startswith(start):
                    start_counts[start] = start_counts.get(start, 0) + 1

        # Repetitive if one start dominates
        if start_counts:
            max_count = max(start_counts.values())
            return max_count >= len(questions) * 0.6

        return False

    def _detect_question_variety(self, recent_qa: list[dict[str, Any]]) -> bool:
        """Detect if recent questions show high variety."""
        if len(recent_qa) < 3:
            return False

        questions = [qa.get("question", "").lower() for qa in recent_qa]

        # Simple variety check based on word overlap
        all_words: set[str] = set()
        unique_ratios = []

        for q in questions:
            words = set(q.split())
            if all_words:
                overlap = len(words & all_words) / len(words) if words else 0
                unique_ratios.append(1 - overlap)
            all_words.update(words)

        if unique_ratios:
            avg_uniqueness = sum(unique_ratios) / len(unique_ratios)
            return avg_uniqueness > 0.5

        return False

    def _has_mixed_performance(self) -> bool:
        """Check if strategies have mixed/varied performance."""
        performances = []

        for strategy in PredictionStrategy:
            history = self._global_performance[strategy]
            if history:
                performances.append(sum(history[-5:]) / len(history[-5:]))

        if len(performances) < 2:
            return False

        # Check variance in performances
        avg = sum(performances) / len(performances)
        variance = sum((p - avg) ** 2 for p in performances) / len(performances)

        return variance > 0.05  # Moderate variance indicates mixed results

    def _get_change_reason(
        self,
        from_strategy: PredictionStrategy,
        to_strategy: PredictionStrategy,
        context: dict[str, Any],
    ) -> str:
        """Generate human-readable reason for strategy change."""
        reasons = {
            (PredictionStrategy.CONTEXT_BASED, PredictionStrategy.PATTERN_MATCHING): (
                "Switching to pattern matching due to repetitive question patterns"
            ),
            (PredictionStrategy.CONTEXT_BASED, PredictionStrategy.SEMANTIC_SIMILARITY): (
                "Switching to semantic similarity for better handling of varied questions"
            ),
            (PredictionStrategy.CONTEXT_BASED, PredictionStrategy.HYBRID): (
                "Switching to hybrid approach for more robust predictions"
            ),
            (PredictionStrategy.PATTERN_MATCHING, PredictionStrategy.CONTEXT_BASED): (
                "Switching to context-based for better project awareness"
            ),
            (PredictionStrategy.PATTERN_MATCHING, PredictionStrategy.SEMANTIC_SIMILARITY): (
                "Switching to semantic similarity for non-repetitive questions"
            ),
            (PredictionStrategy.PATTERN_MATCHING, PredictionStrategy.HYBRID): (
                "Switching to hybrid for balanced prediction"
            ),
            (PredictionStrategy.SEMANTIC_SIMILARITY, PredictionStrategy.CONTEXT_BASED): (
                "Switching to context-based for project-specific predictions"
            ),
            (PredictionStrategy.SEMANTIC_SIMILARITY, PredictionStrategy.PATTERN_MATCHING): (
                "Switching to pattern matching for common question sequences"
            ),
            (PredictionStrategy.SEMANTIC_SIMILARITY, PredictionStrategy.HYBRID): (
                "Switching to hybrid for improved accuracy"
            ),
            (PredictionStrategy.HYBRID, PredictionStrategy.CONTEXT_BASED): (
                "Simplifying to context-based strategy"
            ),
            (PredictionStrategy.HYBRID, PredictionStrategy.PATTERN_MATCHING): (
                "Focusing on pattern matching for current phase"
            ),
            (PredictionStrategy.HYBRID, PredictionStrategy.SEMANTIC_SIMILARITY): (
                "Focusing on semantic matching for diverse questions"
            ),
        }

        key = (from_strategy, to_strategy)
        return reasons.get(key, f"Adjusting from {from_strategy} to {to_strategy} based on performance")

    def get_strategy_config(self, strategy: PredictionStrategy) -> StrategyConfig:
        """Get configuration for a specific strategy."""
        return self._strategies[strategy]

    def update_strategy_parameters(
        self,
        strategy: PredictionStrategy,
        parameters: dict[str, Any],
    ) -> None:
        """Update parameters for a strategy."""
        if strategy in self._strategies:
            self._strategies[strategy].parameters.update(parameters)
            logger.info(
                "Strategy parameters updated",
                strategy=strategy,
                parameters=parameters,
            )

    def get_strategy_stats(self) -> dict[str, Any]:
        """Get statistics for all strategies."""
        stats = {}

        for strategy in PredictionStrategy:
            config = self._strategies[strategy]
            history = self._global_performance[strategy]

            stats[strategy.value] = {
                "weight": config.weight,
                "is_active": config.is_active,
                "total_uses": len(history),
                "avg_performance": (
                    sum(history) / len(history) if history else 0.0
                ),
                "recent_performance": (
                    sum(history[-10:]) / len(history[-10:])
                    if len(history) >= 10
                    else (sum(history) / len(history) if history else 0.0)
                ),
                "parameters": config.parameters,
            }

        return stats

    def recommend_initial_strategy(
        self, context: dict[str, Any]
    ) -> PredictionStrategy:
        """
        Recommend an initial strategy based on context.

        Used when starting a new session.
        """
        project = context.get("project", {})

        # New project with clear structure -> context-based
        if project.get("pending_components") and project.get("architecture_type"):
            return PredictionStrategy.CONTEXT_BASED

        # Debugging or fixing -> pattern matching
        if project.get("current_phase") in ["debugging", "maintenance"]:
            return PredictionStrategy.PATTERN_MATCHING

        # Default to hybrid for balanced approach
        return PredictionStrategy.HYBRID
