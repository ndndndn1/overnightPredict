"""Tests for core components."""

from __future__ import annotations

import pytest

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


class TestModels:
    """Test data models."""

    def test_question_creation(self) -> None:
        """Test Question model creation."""
        question = Question(
            content="How do I implement authentication?",
            question_type=QuestionType.IMPLEMENTATION,
        )

        assert question.content == "How do I implement authentication?"
        assert question.question_type == QuestionType.IMPLEMENTATION
        assert question.id is not None

    def test_answer_creation(self) -> None:
        """Test Answer model creation."""
        answer = Answer(
            question_id="test-q-id",
            content="Here is how to implement authentication...",
            code_snippets=["def authenticate(): pass"],
            confidence=0.85,
        )

        assert answer.question_id == "test-q-id"
        assert answer.confidence == 0.85
        assert len(answer.code_snippets) == 1

    def test_prediction_creation(self) -> None:
        """Test Prediction model creation."""
        prediction = Prediction(
            predicted_question="What tests are needed?",
            question_type=QuestionType.TESTING,
            confidence=0.75,
            strategy_used=PredictionStrategy.CONTEXT_BASED,
        )

        assert prediction.confidence == 0.75
        assert prediction.strategy_used == PredictionStrategy.CONTEXT_BASED

    def test_accuracy_metrics_update(self) -> None:
        """Test AccuracyMetrics update logic."""
        metrics = AccuracyMetrics(
            session_id="test-session",
            strategy=PredictionStrategy.HYBRID,
        )

        # Update with accurate prediction
        metrics.update_accuracy(is_accurate=True, similarity=0.85)
        assert metrics.total_predictions == 1
        assert metrics.accurate_predictions == 1
        assert metrics.accuracy_rate == 1.0

        # Update with inaccurate prediction
        metrics.update_accuracy(is_accurate=False, similarity=0.45)
        assert metrics.total_predictions == 2
        assert metrics.accurate_predictions == 1
        assert metrics.accuracy_rate == 0.5

    def test_session_state_operations(self) -> None:
        """Test SessionState operations."""
        session = SessionState(topic="Test topic")

        # Add question
        question = Question(
            content="Test question",
            question_type=QuestionType.IMPLEMENTATION,
        )
        session.add_question(question)
        assert len(session.questions) == 1

        # Add answer
        answer = Answer(
            question_id=question.id,
            content="Test answer",
        )
        session.add_answer(answer)
        assert len(session.answers) == 1

        # Add prediction
        prediction = Prediction(
            predicted_question="Predicted question",
            question_type=QuestionType.TESTING,
            confidence=0.8,
            strategy_used=PredictionStrategy.PATTERN_MATCHING,
        )
        session.add_prediction(prediction)
        assert len(session.pending_predictions) == 1

    def test_project_context_creation(self) -> None:
        """Test ProjectContext model."""
        project = ProjectContext(
            name="MyProject",
            description="A test project",
            target_languages=["python", "typescript"],
            architecture_type="microservices",
            pending_components=["auth", "api"],
        )

        assert project.name == "MyProject"
        assert len(project.target_languages) == 2
        assert len(project.pending_components) == 2


class TestSettings:
    """Test settings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings creation."""
        settings = Settings()

        assert settings.environment == "development"
        assert settings.ai.primary_provider == "anthropic"
        assert settings.prediction.accuracy_threshold == 0.7

    def test_settings_override(self) -> None:
        """Test nested settings can be configured."""
        from src.core.config import AIConfig, PredictionConfig

        # Test nested config override
        ai_config = AIConfig(primary_provider="openai", anthropic_model="claude-3-opus")
        assert ai_config.primary_provider == "openai"
        assert ai_config.anthropic_model == "claude-3-opus"

        pred_config = PredictionConfig(accuracy_threshold=0.8, lookahead_count=10)
        assert pred_config.accuracy_threshold == 0.8
        assert pred_config.lookahead_count == 10


class TestCodeArtifact:
    """Test CodeArtifact model."""

    def test_artifact_creation(self) -> None:
        """Test artifact creation."""
        artifact = CodeArtifact(
            session_id="test-session",
            file_path="src/auth/handler.py",
            content="def authenticate(): pass",
            language="python",
        )

        assert artifact.file_path == "src/auth/handler.py"
        assert artifact.language == "python"
        assert artifact.is_valid is True
        assert artifact.version == 1

    def test_artifact_with_errors(self) -> None:
        """Test artifact with lint errors."""
        artifact = CodeArtifact(
            session_id="test-session",
            file_path="src/broken.py",
            content="def broken(",
            language="python",
            is_valid=False,
            lint_errors=["Syntax error: unexpected EOF"],
        )

        assert artifact.is_valid is False
        assert len(artifact.lint_errors) == 1
