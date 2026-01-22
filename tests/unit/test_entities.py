"""Unit tests for domain entities."""

import pytest
from datetime import datetime, timedelta

from src.domain.entities.session import Session, SessionStatus, SessionMetrics
from src.domain.entities.prediction import Prediction, PredictionStatus, PredictionBatch
from src.domain.entities.question import Question, QuestionType, QuestionSource
from src.domain.entities.task import Task, TaskStatus, TaskPriority, TaskType, TaskResult
from src.domain.value_objects.accuracy import AccuracyScore, AccuracyLevel


class TestSession:
    """Tests for Session entity."""

    def test_session_creation(self):
        """Test creating a new session."""
        session = Session(
            name="Test Session",
            provider="openai",
            model="gpt-4",
        )

        assert session.id.startswith("sess_")
        assert session.name == "Test Session"
        assert session.status == SessionStatus.INITIALIZING
        assert session.metrics.questions_processed == 0

    def test_session_lifecycle(self):
        """Test session state transitions."""
        session = Session()

        # Start
        session.start()
        assert session.status == SessionStatus.RUNNING
        assert session.started_at is not None

        # Pause
        session.pause()
        assert session.status == SessionStatus.PAUSED

        # Resume
        session.resume()
        assert session.status == SessionStatus.RUNNING

        # Complete
        session.complete()
        assert session.status == SessionStatus.COMPLETED
        assert session.completed_at is not None

    def test_session_fail(self):
        """Test session failure handling."""
        session = Session()
        session.start()
        session.fail("Test error")

        assert session.status == SessionStatus.FAILED
        assert session.last_error == "Test error"
        assert session.error_count == 1

    def test_session_rate_limiting(self):
        """Test rate limit handling."""
        session = Session()
        session.start()

        limit_time = datetime.utcnow() + timedelta(minutes=5)
        session.set_rate_limited(limit_time)

        assert session.status == SessionStatus.WAITING_RATE_LIMIT
        assert session.rate_limited_until == limit_time

        session.clear_rate_limit()
        assert session.status == SessionStatus.RUNNING
        assert session.rate_limited_until is None

    def test_session_metrics(self):
        """Test session metrics tracking."""
        session = Session()
        session.start()

        session.add_question("q1")
        session.add_prediction("p1")
        session.record_accurate_prediction()
        session.record_task_completion(success=True)

        assert session.metrics.questions_processed == 1
        assert session.metrics.predictions_made == 1
        assert session.metrics.predictions_accurate == 1
        assert session.metrics.tasks_completed == 1
        assert session.metrics.prediction_accuracy == 1.0


class TestPrediction:
    """Tests for Prediction entity."""

    def test_prediction_creation(self):
        """Test creating a prediction."""
        pred = Prediction(
            session_id="sess_123",
            predicted_question="How do I implement X?",
            confidence=0.85,
        )

        assert pred.id.startswith("pred_")
        assert pred.status == PredictionStatus.PENDING
        assert pred.confidence == 0.85

    def test_prediction_match(self):
        """Test matching a prediction."""
        pred = Prediction(predicted_question="Test question")

        pred.match("Actual question", 0.9)

        assert pred.status == PredictionStatus.MATCHED
        assert pred.actual_question == "Actual question"
        assert pred.similarity_score == 0.9
        assert pred.evaluated_at is not None

    def test_prediction_unmatch(self):
        """Test unmatching a prediction."""
        pred = Prediction(predicted_question="Test question")

        pred.unmatch("Different question", 0.3)

        assert pred.status == PredictionStatus.UNMATCHED
        assert pred.similarity_score == 0.3

    def test_prediction_batch(self):
        """Test prediction batch operations."""
        batch = PredictionBatch(session_id="sess_123")

        for i in range(3):
            pred = Prediction(predicted_question=f"Question {i}")
            batch.add(pred)

        assert len(batch.predictions) == 3
        assert batch.predictions[0].sequence_number == 0
        assert batch.predictions[2].sequence_number == 2

        # Match first prediction
        batch.predictions[0].match("Q0", 0.9)

        assert batch.accuracy == 1 / 1  # 1 evaluated, 1 matched
        assert len(batch.get_pending()) == 2


class TestQuestion:
    """Tests for Question entity."""

    def test_question_creation(self):
        """Test creating a question."""
        q = Question(
            content="How do I implement feature X?",
            question_type=QuestionType.FEATURE,
            source=QuestionSource.USER,
        )

        assert q.id.startswith("q_")
        assert not q.processed
        assert q.answer is None

    def test_question_processing(self):
        """Test marking question as processed."""
        q = Question(content="Test question")

        q.mark_processed("Test answer")

        assert q.processed
        assert q.answer == "Test answer"
        assert q.answer_at is not None


class TestTask:
    """Tests for Task entity."""

    def test_task_creation(self):
        """Test creating a task."""
        task = Task(
            description="Implement feature",
            instructions="Add the feature to module X",
            task_type=TaskType.CODE_GENERATION,
            priority=TaskPriority.HIGH,
        )

        assert task.id.startswith("task_")
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.HIGH

    def test_task_execution(self):
        """Test task execution flow."""
        task = Task(description="Test task")

        task.start()
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None

        result = TaskResult(success=True, output="Done")
        task.complete(result)

        assert task.status == TaskStatus.COMPLETED
        assert task.result.success

    def test_task_retry(self):
        """Test task retry logic."""
        task = Task(description="Test", max_retries=3)

        task.fail("Error 1")
        assert task.status == TaskStatus.FAILED
        assert task.can_retry()

        task.retry()
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 1


class TestAccuracyScore:
    """Tests for AccuracyScore value object."""

    def test_accuracy_score_creation(self):
        """Test creating an accuracy score."""
        score = AccuracyScore(
            value=0.85,
            predicted="Predicted Q",
            actual="Actual Q",
        )

        assert score.value == 0.85
        assert score.level == AccuracyLevel.GOOD
        assert score.is_acceptable

    def test_accuracy_score_levels(self):
        """Test accuracy level determination."""
        assert AccuracyScore(0.95, "", "").level == AccuracyLevel.EXCELLENT
        assert AccuracyScore(0.75, "", "").level == AccuracyLevel.GOOD
        assert AccuracyScore(0.55, "", "").level == AccuracyLevel.MODERATE
        assert AccuracyScore(0.35, "", "").level == AccuracyLevel.POOR
        assert AccuracyScore(0.15, "", "").level == AccuracyLevel.VERY_POOR

    def test_accuracy_score_validation(self):
        """Test accuracy score validation."""
        with pytest.raises(ValueError):
            AccuracyScore(1.5, "", "")

        with pytest.raises(ValueError):
            AccuracyScore(-0.5, "", "")
