"""Accuracy evaluation module for comparing predictions with actual questions."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog

from src.core.config import Settings
from src.core.models import Prediction, PredictionResult, Question
from src.predictors.embeddings import EmbeddingService, get_embedding_service

logger = structlog.get_logger(__name__)


class AccuracyEvaluator:
    """
    Evaluates the accuracy of question predictions.

    Uses multiple methods:
    - Semantic similarity using embeddings
    - Keyword/entity matching
    - Question type matching
    - Intent analysis
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the accuracy evaluator."""
        self.settings = settings
        self._embedding_service: EmbeddingService | None = None

        # Weights for different evaluation components
        self._weights = {
            "semantic_similarity": 0.5,
            "keyword_match": 0.2,
            "type_match": 0.15,
            "intent_match": 0.15,
        }

    @property
    def embedding_service(self) -> EmbeddingService:
        """Get the embedding service (lazy init)."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service(
                self.settings.prediction.embedding_model
            )
        return self._embedding_service

    async def evaluate(
        self,
        prediction: Prediction,
        actual_question: Question,
    ) -> PredictionResult:
        """
        Evaluate how well a prediction matches an actual question.

        Args:
            prediction: The predicted question
            actual_question: The actual question received

        Returns:
            Evaluation result with similarity score and accuracy determination
        """
        logger.info(
            "Evaluating prediction",
            prediction_id=prediction.id,
            actual_question_id=actual_question.id,
        )

        # Compute individual scores in parallel
        scores = await asyncio.gather(
            self._compute_semantic_similarity(
                prediction.predicted_question,
                actual_question.content,
            ),
            self._compute_keyword_match(
                prediction.predicted_question,
                actual_question.content,
            ),
            self._compute_type_match(
                prediction.question_type,
                actual_question.question_type,
            ),
            self._compute_intent_match(
                prediction.predicted_question,
                actual_question.content,
            ),
        )

        semantic_score, keyword_score, type_score, intent_score = scores

        # Compute weighted final score
        final_score = (
            semantic_score * self._weights["semantic_similarity"]
            + keyword_score * self._weights["keyword_match"]
            + type_score * self._weights["type_match"]
            + intent_score * self._weights["intent_match"]
        )

        # Determine if accurate based on threshold
        is_accurate = final_score >= self.settings.prediction.accuracy_threshold

        # Generate feedback
        feedback = self._generate_feedback(
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            type_score=type_score,
            intent_score=intent_score,
            final_score=final_score,
            is_accurate=is_accurate,
        )

        result = PredictionResult(
            prediction_id=prediction.id,
            actual_question_id=actual_question.id,
            similarity_score=final_score,
            is_accurate=is_accurate,
            strategy_used=prediction.strategy_used,
            feedback=feedback,
        )

        logger.info(
            "Prediction evaluated",
            prediction_id=prediction.id,
            similarity_score=final_score,
            is_accurate=is_accurate,
            semantic=semantic_score,
            keyword=keyword_score,
            type=type_score,
            intent=intent_score,
        )

        return result

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute overall similarity between two texts.

        Public method for external use.
        """
        return await self._compute_semantic_similarity(text1, text2)

    async def evaluate_batch(
        self,
        predictions: list[Prediction],
        actual_question: Question,
    ) -> list[PredictionResult]:
        """
        Evaluate multiple predictions against an actual question.

        Args:
            predictions: List of predictions to evaluate
            actual_question: The actual question received

        Returns:
            List of evaluation results
        """
        results = await asyncio.gather(
            *[self.evaluate(pred, actual_question) for pred in predictions]
        )
        return list(results)

    async def _compute_semantic_similarity(
        self, predicted: str, actual: str
    ) -> float:
        """
        Compute semantic similarity using embeddings.

        This captures whether the questions have similar meanings
        even if expressed differently.
        """
        await self.embedding_service.initialize()
        return await self.embedding_service.compute_similarity(predicted, actual)

    async def _compute_keyword_match(
        self, predicted: str, actual: str
    ) -> float:
        """
        Compute keyword overlap between questions.

        Extracts important keywords and measures overlap.
        """
        # Extract keywords (non-stopwords with length > 3)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "this", "that", "these", "those", "what",
            "which", "who", "whom", "whose",
        }

        def extract_keywords(text: str) -> set[str]:
            words = re.findall(r"\b\w+\b", text.lower())
            return {w for w in words if len(w) > 3 and w not in stopwords}

        predicted_keywords = extract_keywords(predicted)
        actual_keywords = extract_keywords(actual)

        if not predicted_keywords or not actual_keywords:
            return 0.0

        # Compute Jaccard similarity
        intersection = predicted_keywords & actual_keywords
        union = predicted_keywords | actual_keywords

        return len(intersection) / len(union) if union else 0.0

    async def _compute_type_match(
        self, predicted_type: Any, actual_type: Any
    ) -> float:
        """
        Compute whether question types match.

        Returns 1.0 for exact match, partial scores for related types.
        """
        if predicted_type == actual_type:
            return 1.0

        # Define related question types
        related_types = {
            "implementation": {"debugging", "optimization"},
            "debugging": {"implementation", "testing"},
            "testing": {"debugging", "implementation"},
            "architecture": {"implementation", "optimization"},
            "optimization": {"implementation", "architecture"},
            "documentation": {"clarification"},
            "clarification": {"documentation", "architecture"},
        }

        # Convert to string value for comparison
        pred_str = predicted_type.value if hasattr(predicted_type, 'value') else str(predicted_type).lower()
        actual_str = actual_type.value if hasattr(actual_type, 'value') else str(actual_type).lower()

        # Check if types are related
        related = related_types.get(pred_str, set())
        if actual_str in related:
            return 0.5

        return 0.0

    async def _compute_intent_match(
        self, predicted: str, actual: str
    ) -> float:
        """
        Compute whether the questions have similar intent.

        Analyzes question structure and action words.
        """
        # Extract intent indicators
        intent_patterns = {
            "how_to": (r"\bhow\s+(do|should|can|to)\b", r"\bhow\s+(do|should|can|to)\b"),
            "what_is": (r"\bwhat\s+(is|are)\b", r"\bwhat\s+(is|are)\b"),
            "why": (r"\bwhy\b", r"\bwhy\b"),
            "implement": (r"\b(implement|create|build|add|write)\b", r"\b(implement|create|build|add|write)\b"),
            "fix": (r"\b(fix|debug|solve|resolve)\b", r"\b(fix|debug|solve|resolve)\b"),
            "test": (r"\b(test|verify|validate|check)\b", r"\b(test|verify|validate|check)\b"),
            "optimize": (r"\b(optimize|improve|performance|faster)\b", r"\b(optimize|improve|performance|faster)\b"),
            "design": (r"\b(design|architecture|structure)\b", r"\b(design|architecture|structure)\b"),
        }

        predicted_lower = predicted.lower()
        actual_lower = actual.lower()

        matches = 0
        total = 0

        for intent, (pred_pattern, actual_pattern) in intent_patterns.items():
            pred_match = bool(re.search(pred_pattern, predicted_lower))
            actual_match = bool(re.search(actual_pattern, actual_lower))

            if pred_match or actual_match:
                total += 1
                if pred_match and actual_match:
                    matches += 1

        return matches / total if total > 0 else 0.5

    def _generate_feedback(
        self,
        semantic_score: float,
        keyword_score: float,
        type_score: float,
        intent_score: float,
        final_score: float,
        is_accurate: bool,
    ) -> str:
        """Generate human-readable feedback about the evaluation."""
        feedback_parts = []

        if is_accurate:
            feedback_parts.append("Prediction was accurate.")
        else:
            feedback_parts.append("Prediction did not meet accuracy threshold.")

        # Identify strongest and weakest components
        scores = {
            "semantic similarity": semantic_score,
            "keyword overlap": keyword_score,
            "question type": type_score,
            "intent matching": intent_score,
        }

        best = max(scores, key=lambda k: scores[k])
        worst = min(scores, key=lambda k: scores[k])

        if scores[best] > 0.7:
            feedback_parts.append(f"Strong {best} ({scores[best]:.2f}).")

        if scores[worst] < 0.3:
            feedback_parts.append(f"Weak {worst} ({scores[worst]:.2f}).")

        # Suggest improvements
        if not is_accurate:
            if semantic_score < 0.5:
                feedback_parts.append(
                    "Consider improving context understanding for better semantic matching."
                )
            if keyword_score < 0.3:
                feedback_parts.append(
                    "Focus on capturing key domain terms."
                )
            if type_score < 0.5:
                feedback_parts.append(
                    "Question type classification needs improvement."
                )

        return " ".join(feedback_parts)

    async def get_evaluation_stats(
        self, results: list[PredictionResult]
    ) -> dict[str, Any]:
        """
        Compute aggregate statistics from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {
                "total": 0,
                "accurate": 0,
                "accuracy_rate": 0.0,
                "avg_similarity": 0.0,
                "strategy_performance": {},
            }

        accurate_count = sum(1 for r in results if r.is_accurate)
        avg_similarity = sum(r.similarity_score for r in results) / len(results)

        # Group by strategy
        strategy_results: dict[str, list[PredictionResult]] = {}
        for r in results:
            strategy = r.strategy_used.value if hasattr(r.strategy_used, 'value') else str(r.strategy_used)
            if strategy not in strategy_results:
                strategy_results[strategy] = []
            strategy_results[strategy].append(r)

        strategy_performance = {}
        for strategy, strat_results in strategy_results.items():
            strat_accurate = sum(1 for r in strat_results if r.is_accurate)
            strat_avg = sum(r.similarity_score for r in strat_results) / len(strat_results)
            strategy_performance[strategy] = {
                "count": len(strat_results),
                "accurate": strat_accurate,
                "accuracy_rate": strat_accurate / len(strat_results),
                "avg_similarity": strat_avg,
            }

        return {
            "total": len(results),
            "accurate": accurate_count,
            "accuracy_rate": accurate_count / len(results),
            "avg_similarity": avg_similarity,
            "strategy_performance": strategy_performance,
        }
