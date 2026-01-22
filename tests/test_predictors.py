"""Tests for question prediction components."""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.core.models import PredictionStrategy, Question, QuestionType
from src.predictors.embeddings import EmbeddingService
from src.predictors.question import QuestionPredictor


class TestEmbeddingService:
    """Test embedding service."""

    @pytest.fixture
    def embedding_service(self) -> EmbeddingService:
        """Create embedding service."""
        return EmbeddingService(model_name="all-MiniLM-L6-v2")

    @pytest.mark.asyncio
    async def test_simple_embedding(self, embedding_service: EmbeddingService) -> None:
        """Test simple embedding generation (fallback)."""
        # This will use the fallback since model may not be loaded
        embedding = await embedding_service.get_embedding("Hello world")

        assert embedding is not None
        assert len(embedding) == 384  # Default dimension

    @pytest.mark.asyncio
    async def test_similarity_computation(
        self, embedding_service: EmbeddingService
    ) -> None:
        """Test similarity computation between texts."""
        similarity = await embedding_service.compute_similarity(
            "How do I implement authentication?",
            "What is the best way to add login functionality?",
        )

        assert 0 <= similarity <= 1

    @pytest.mark.asyncio
    async def test_batch_embeddings(self, embedding_service: EmbeddingService) -> None:
        """Test batch embedding generation."""
        texts = [
            "First text",
            "Second text",
            "Third text",
        ]

        embeddings = await embedding_service.get_embeddings(texts)

        assert len(embeddings) == 3
        assert embeddings.shape[1] == 384

    @pytest.mark.asyncio
    async def test_find_most_similar(self, embedding_service: EmbeddingService) -> None:
        """Test finding most similar texts."""
        query = "authentication"
        candidates = [
            "login and security",
            "database operations",
            "user authentication",
            "file handling",
        ]

        results = await embedding_service.find_most_similar(query, candidates, top_k=2)

        assert len(results) == 2
        # Results should be sorted by similarity
        assert results[0][1] >= results[1][1]


class TestQuestionPredictor:
    """Test question predictor."""

    @pytest.fixture
    def predictor(self, settings: Settings) -> QuestionPredictor:
        """Create question predictor."""
        return QuestionPredictor(settings)

    @pytest.mark.asyncio
    async def test_context_based_prediction(
        self, predictor: QuestionPredictor
    ) -> None:
        """Test context-based prediction."""
        context = {
            "project": {
                "name": "TestProject",
                "pending_components": ["auth", "api", "database"],
                "current_phase": "implementation",
            }
        }

        predictions = await predictor.predict(
            context=context,
            history=[],
            strategy=PredictionStrategy.CONTEXT_BASED,
            count=3,
        )

        assert len(predictions) == 3
        for pred in predictions:
            assert pred.strategy_used == PredictionStrategy.CONTEXT_BASED
            assert 0 <= pred.confidence <= 1

    @pytest.mark.asyncio
    async def test_pattern_based_prediction(
        self, predictor: QuestionPredictor
    ) -> None:
        """Test pattern-based prediction."""
        context = {"project": {"current_phase": "implementation"}}

        history = [
            Question(
                content="How should I design the architecture?",
                question_type=QuestionType.ARCHITECTURE,
            ),
            Question(
                content="How do I implement the user service?",
                question_type=QuestionType.IMPLEMENTATION,
            ),
        ]

        predictions = await predictor.predict(
            context=context,
            history=history,
            strategy=PredictionStrategy.PATTERN_MATCHING,
            count=3,
        )

        assert len(predictions) == 3
        for pred in predictions:
            assert pred.strategy_used == PredictionStrategy.PATTERN_MATCHING

    @pytest.mark.asyncio
    async def test_hybrid_prediction(self, predictor: QuestionPredictor) -> None:
        """Test hybrid prediction combining strategies."""
        context = {
            "project": {
                "name": "TestProject",
                "pending_components": ["auth"],
                "current_phase": "implementation",
            }
        }

        predictions = await predictor.predict(
            context=context,
            history=[],
            strategy=PredictionStrategy.HYBRID,
            count=3,
        )

        assert len(predictions) == 3
        for pred in predictions:
            assert pred.strategy_used == PredictionStrategy.HYBRID

    def test_question_type_inference(self, predictor: QuestionPredictor) -> None:
        """Test question type inference from text."""
        test_cases = [
            ("How do I implement the login feature?", QuestionType.IMPLEMENTATION),
            ("What is the architecture for this system?", QuestionType.ARCHITECTURE),
            ("How should I test this component?", QuestionType.TESTING),
            ("Why is this code not working?", QuestionType.DEBUGGING),
            ("How can I optimize this algorithm?", QuestionType.OPTIMIZATION),
        ]

        for question, expected_type in test_cases:
            inferred = predictor._infer_question_type(question)
            assert inferred == expected_type, f"Failed for: {question}"
