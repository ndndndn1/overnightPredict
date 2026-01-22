"""Tests for accuracy evaluation components."""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.core.models import Prediction, PredictionStrategy, Question, QuestionType
from src.evaluators.accuracy import AccuracyEvaluator


class TestAccuracyEvaluator:
    """Test accuracy evaluator."""

    @pytest.fixture
    def evaluator(self, settings: Settings) -> AccuracyEvaluator:
        """Create accuracy evaluator."""
        return AccuracyEvaluator(settings)

    @pytest.mark.asyncio
    async def test_evaluate_exact_match(self, evaluator: AccuracyEvaluator) -> None:
        """Test evaluation with exact matching questions."""
        prediction = Prediction(
            predicted_question="How do I implement authentication?",
            question_type=QuestionType.IMPLEMENTATION,
            confidence=0.8,
            strategy_used=PredictionStrategy.CONTEXT_BASED,
        )

        actual = Question(
            content="How do I implement authentication?",
            question_type=QuestionType.IMPLEMENTATION,
        )

        result = await evaluator.evaluate(prediction, actual)

        assert result.similarity_score > 0.9
        assert result.is_accurate is True

    @pytest.mark.asyncio
    async def test_evaluate_similar_questions(
        self, evaluator: AccuracyEvaluator
    ) -> None:
        """Test evaluation with similar but not identical questions."""
        prediction = Prediction(
            predicted_question="How should I implement user authentication?",
            question_type=QuestionType.IMPLEMENTATION,
            confidence=0.8,
            strategy_used=PredictionStrategy.CONTEXT_BASED,
        )

        actual = Question(
            content="What is the best way to add login functionality?",
            question_type=QuestionType.IMPLEMENTATION,
        )

        result = await evaluator.evaluate(prediction, actual)

        # Should have moderate similarity
        assert 0.3 <= result.similarity_score <= 0.9

    @pytest.mark.asyncio
    async def test_evaluate_different_questions(
        self, evaluator: AccuracyEvaluator
    ) -> None:
        """Test evaluation with very different questions."""
        prediction = Prediction(
            predicted_question="How do I optimize database queries?",
            question_type=QuestionType.OPTIMIZATION,
            confidence=0.8,
            strategy_used=PredictionStrategy.CONTEXT_BASED,
        )

        actual = Question(
            content="What testing framework should I use?",
            question_type=QuestionType.TESTING,
        )

        result = await evaluator.evaluate(prediction, actual)

        assert result.similarity_score < 0.5
        assert result.is_accurate is False

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, evaluator: AccuracyEvaluator) -> None:
        """Test batch evaluation."""
        predictions = [
            Prediction(
                predicted_question="Question 1",
                question_type=QuestionType.IMPLEMENTATION,
                confidence=0.8,
                strategy_used=PredictionStrategy.CONTEXT_BASED,
            ),
            Prediction(
                predicted_question="Question 2",
                question_type=QuestionType.TESTING,
                confidence=0.7,
                strategy_used=PredictionStrategy.PATTERN_MATCHING,
            ),
        ]

        actual = Question(
            content="Question 1",
            question_type=QuestionType.IMPLEMENTATION,
        )

        results = await evaluator.evaluate_batch(predictions, actual)

        assert len(results) == 2
        # First prediction should be more accurate
        assert results[0].similarity_score > results[1].similarity_score

    @pytest.mark.asyncio
    async def test_keyword_match_scoring(self, evaluator: AccuracyEvaluator) -> None:
        """Test keyword match component of scoring."""
        score = await evaluator._compute_keyword_match(
            "How do I implement authentication?",
            "Implementing authentication in the system",
        )

        # Should have some keyword overlap
        assert score > 0.2

    @pytest.mark.asyncio
    async def test_type_match_scoring(self, evaluator: AccuracyEvaluator) -> None:
        """Test question type matching."""
        # Exact match
        exact_score = await evaluator._compute_type_match(
            QuestionType.IMPLEMENTATION,
            QuestionType.IMPLEMENTATION,
        )
        assert exact_score == 1.0

        # Related types
        related_score = await evaluator._compute_type_match(
            QuestionType.IMPLEMENTATION,
            QuestionType.DEBUGGING,
        )
        assert related_score == 0.5

        # Unrelated types
        unrelated_score = await evaluator._compute_type_match(
            QuestionType.DOCUMENTATION,
            QuestionType.OPTIMIZATION,
        )
        assert unrelated_score == 0.0

    @pytest.mark.asyncio
    async def test_evaluation_stats(self, evaluator: AccuracyEvaluator) -> None:
        """Test aggregate statistics computation."""
        from src.core.models import PredictionResult

        results = [
            PredictionResult(
                prediction_id="p1",
                actual_question_id="q1",
                similarity_score=0.85,
                is_accurate=True,
                strategy_used=PredictionStrategy.CONTEXT_BASED,
            ),
            PredictionResult(
                prediction_id="p2",
                actual_question_id="q2",
                similarity_score=0.45,
                is_accurate=False,
                strategy_used=PredictionStrategy.CONTEXT_BASED,
            ),
            PredictionResult(
                prediction_id="p3",
                actual_question_id="q3",
                similarity_score=0.75,
                is_accurate=True,
                strategy_used=PredictionStrategy.PATTERN_MATCHING,
            ),
        ]

        stats = await evaluator.get_evaluation_stats(results)

        assert stats["total"] == 3
        assert stats["accurate"] == 2
        assert stats["accuracy_rate"] == pytest.approx(2 / 3)
        assert "context_based" in stats["strategy_performance"]
