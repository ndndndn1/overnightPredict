"""
Evaluator service (Critic) - Measures prediction accuracy.

Computes semantic similarity between predicted and actual questions
to evaluate the quality of predictions.
"""

import time
from typing import Optional

import numpy as np

from src.core.config.settings import PredictionConfig
from src.core.utils.logging import get_logger
from src.domain.entities.prediction import Prediction
from src.domain.entities.question import Question
from src.domain.interfaces.evaluator import (
    IEvaluator,
    EvaluationResult,
    BatchEvaluationResult,
)
from src.domain.value_objects.accuracy import AccuracyScore


class SemanticEvaluator(IEvaluator):
    """
    Evaluator using semantic similarity via sentence transformers.

    Uses pre-trained models to compute embeddings and measure
    semantic similarity between texts.
    """

    def __init__(self, config: PredictionConfig):
        """Initialize the evaluator.

        Args:
            config: Prediction configuration.
        """
        self._config = config
        self._logger = get_logger("evaluator.semantic")
        self._model = None
        self._model_name = config.similarity_model

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._logger.info("Loading similarity model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._logger.info("Model loaded successfully")
        except ImportError:
            self._logger.warning(
                "sentence-transformers not installed, using fallback"
            )
            self._model = None

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._model = None

    @property
    def evaluation_method(self) -> str:
        """Get the evaluation method name."""
        return "semantic_similarity"

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        if self._model is None:
            await self.initialize()

        if self._model is None:
            # Fallback: simple hash-based "embedding"
            return self._fallback_embedding(text)

        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _fallback_embedding(self, text: str) -> list[float]:
        """Fallback embedding using character frequencies."""
        # Simple character frequency vector
        embedding = [0.0] * 128
        for char in text.lower():
            if ord(char) < 128:
                embedding[ord(char)] += 1

        # Normalize
        total = sum(embedding) or 1
        return [v / total for v in embedding]

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if self._model is None:
            await self.initialize()

        if self._model is None:
            return self._fallback_similarity(text1, text2)

        # Get embeddings
        embeddings = self._model.encode([text1, text2], convert_to_numpy=True)

        # Cosine similarity
        similarity = float(np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        ))

        # Normalize to [0, 1]
        return (similarity + 1) / 2

    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Fallback similarity using Jaccard index."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    async def evaluate(
        self,
        prediction: Prediction,
        actual_question: Question,
        threshold: float = 0.6,
    ) -> EvaluationResult:
        """Evaluate a single prediction."""
        start_time = time.time()

        # Compute semantic similarity
        semantic_similarity = await self.compute_similarity(
            prediction.predicted_question,
            actual_question.content,
        )

        # Compute keyword overlap
        keyword_overlap = self._compute_keyword_overlap(
            prediction.predicted_question,
            actual_question.content,
        )

        # Create accuracy score
        accuracy_score = AccuracyScore(
            value=semantic_similarity,
            predicted=prediction.predicted_question,
            actual=actual_question.content,
            method=self.evaluation_method,
        )

        # Determine if it's a match
        is_match = semantic_similarity >= threshold

        # Determine if we should use the pre-computed answer
        should_use_prediction = (
            is_match
            and prediction.predicted_answer is not None
            and prediction.confidence >= 0.8
        )

        evaluation_time = (time.time() - start_time) * 1000

        result = EvaluationResult(
            prediction=prediction,
            actual_question=actual_question,
            accuracy_score=accuracy_score,
            semantic_similarity=semantic_similarity,
            keyword_overlap=keyword_overlap,
            intent_match=is_match,
            is_match=is_match,
            should_use_prediction=should_use_prediction,
            evaluation_time_ms=evaluation_time,
            reasoning=self._generate_reasoning(
                semantic_similarity, keyword_overlap, is_match
            ),
        )

        # Update prediction status
        if is_match:
            prediction.match(actual_question.content, semantic_similarity)
        else:
            prediction.unmatch(actual_question.content, semantic_similarity)

        self._logger.debug(
            "Evaluation complete",
            prediction_id=prediction.id,
            similarity=semantic_similarity,
            is_match=is_match,
        )

        return result

    async def evaluate_batch(
        self,
        predictions: list[Prediction],
        actual_questions: list[Question],
        threshold: float = 0.6,
    ) -> BatchEvaluationResult:
        """Evaluate multiple predictions against actual questions."""
        start_time = time.time()
        batch_result = BatchEvaluationResult()

        # Match predictions to questions by finding best matches
        for actual in actual_questions:
            best_match: Optional[EvaluationResult] = None
            best_similarity = 0.0

            for prediction in predictions:
                if prediction.is_evaluated:
                    continue

                result = await self.evaluate(
                    prediction, actual, threshold
                )

                if result.semantic_similarity > best_similarity:
                    best_similarity = result.semantic_similarity
                    best_match = result

            if best_match:
                batch_result.add_result(best_match)

        batch_result.total_time_ms = (time.time() - start_time) * 1000

        self._logger.info(
            "Batch evaluation complete",
            predictions=len(predictions),
            questions=len(actual_questions),
            matches=batch_result.accurate_count,
            accuracy=batch_result.average_accuracy,
        )

        return batch_result

    def _compute_keyword_overlap(self, text1: str, text2: str) -> float:
        """Compute keyword overlap ratio."""
        # Extract keywords (simple: non-stop words longer than 3 chars)
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "have", "has", "had", "do", "does", "did", "will", "would",
                      "could", "should", "may", "might", "must", "and", "or", "but",
                      "if", "then", "else", "when", "where", "why", "how", "what",
                      "this", "that", "these", "those", "it", "its", "to", "for",
                      "of", "in", "on", "at", "by", "with", "from"}

        def get_keywords(text: str) -> set[str]:
            words = text.lower().split()
            return {w for w in words if len(w) > 3 and w not in stop_words}

        kw1 = get_keywords(text1)
        kw2 = get_keywords(text2)

        if not kw1 or not kw2:
            return 0.0

        intersection = len(kw1 & kw2)
        union = len(kw1 | kw2)

        return intersection / union if union > 0 else 0.0

    def _generate_reasoning(
        self,
        similarity: float,
        keyword_overlap: float,
        is_match: bool,
    ) -> str:
        """Generate human-readable reasoning for the evaluation."""
        parts = []

        if is_match:
            parts.append(f"Match found with {similarity:.1%} semantic similarity")
        else:
            parts.append(f"No match: {similarity:.1%} similarity below threshold")

        if keyword_overlap > 0.5:
            parts.append(f"Strong keyword overlap ({keyword_overlap:.1%})")
        elif keyword_overlap > 0.2:
            parts.append(f"Moderate keyword overlap ({keyword_overlap:.1%})")
        else:
            parts.append(f"Low keyword overlap ({keyword_overlap:.1%})")

        return ". ".join(parts) + "."


# Alias for backward compatibility
Evaluator = SemanticEvaluator
