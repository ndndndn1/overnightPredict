"""SQLite implementation of session repository."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from src.core.utils.logging import get_logger
from src.domain.entities.session import Session, SessionStatus, SessionMetrics
from src.domain.entities.prediction import Prediction, PredictionStatus, PredictionBatch
from src.domain.entities.question import Question, QuestionType, QuestionSource
from src.domain.entities.task import Task, TaskStatus, TaskPriority, TaskType, TaskResult
from src.domain.interfaces.session_repository import ISessionRepository


class SQLiteSessionRepository(ISessionRepository):
    """
    SQLite implementation of session repository.

    Provides persistent storage for sessions, predictions, questions, and tasks.
    """

    def __init__(self, db_path: Path):
        """Initialize the repository.

        Args:
            db_path: Path to SQLite database file.
        """
        self._db_path = db_path
        self._logger = get_logger("storage.sqlite", db_path=str(db_path))
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return

        async with aiosqlite.connect(self._db_path) as db:
            # Sessions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    provider TEXT,
                    status TEXT,
                    model TEXT,
                    initial_prompt TEXT,
                    working_directory TEXT,
                    group_id TEXT,
                    created_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    last_activity_at TEXT,
                    rate_limited_until TEXT,
                    current_task_id TEXT,
                    current_prediction_id TEXT,
                    current_strategy TEXT,
                    metrics_json TEXT,
                    last_error TEXT,
                    error_count INTEGER DEFAULT 0,
                    metadata_json TEXT
                )
            """)

            # Predictions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    predicted_question TEXT,
                    predicted_answer TEXT,
                    predicted_context TEXT,
                    strategy_used TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    status TEXT,
                    actual_question TEXT,
                    similarity_score REAL,
                    created_at TEXT,
                    evaluated_at TEXT,
                    expires_at TEXT,
                    sequence_number INTEGER,
                    lookahead_position INTEGER,
                    metadata_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)

            # Questions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    content TEXT,
                    context TEXT,
                    expected_output TEXT,
                    question_type TEXT,
                    source TEXT,
                    parent_question_id TEXT,
                    prediction_id TEXT,
                    processed INTEGER DEFAULT 0,
                    answer TEXT,
                    answer_at TEXT,
                    created_at TEXT,
                    priority INTEGER DEFAULT 0,
                    sequence_number INTEGER,
                    embedding_json TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)

            # Tasks table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    question_id TEXT,
                    description TEXT,
                    instructions TEXT,
                    task_type TEXT,
                    priority INTEGER,
                    status TEXT,
                    result_json TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    depends_on_json TEXT,
                    blocked_by TEXT,
                    created_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    context TEXT,
                    working_directory TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)

            # Indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_group ON sessions(group_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(session_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_questions_session ON questions(session_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id)")

            await db.commit()

        self._initialized = True
        self._logger.info("Database initialized")

    async def _ensure_initialized(self) -> None:
        """Ensure database is initialized."""
        if not self._initialized:
            await self.initialize()

    # Session operations

    async def save_session(self, session: Session) -> str:
        """Save or update a session."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO sessions (
                    id, name, provider, status, model, initial_prompt, working_directory,
                    group_id, created_at, started_at, completed_at, last_activity_at,
                    rate_limited_until, current_task_id, current_prediction_id,
                    current_strategy, metrics_json, last_error, error_count, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.id, session.name, session.provider, session.status.value,
                session.model, session.initial_prompt, session.working_directory,
                session.group_id,
                session.created_at.isoformat(),
                session.started_at.isoformat() if session.started_at else None,
                session.completed_at.isoformat() if session.completed_at else None,
                session.last_activity_at.isoformat() if session.last_activity_at else None,
                session.rate_limited_until.isoformat() if session.rate_limited_until else None,
                session.current_task_id, session.current_prediction_id,
                session.current_strategy,
                json.dumps({
                    "questions_processed": session.metrics.questions_processed,
                    "predictions_made": session.metrics.predictions_made,
                    "predictions_accurate": session.metrics.predictions_accurate,
                    "tasks_completed": session.metrics.tasks_completed,
                    "tasks_failed": session.metrics.tasks_failed,
                    "total_tokens_used": session.metrics.total_tokens_used,
                    "total_api_calls": session.metrics.total_api_calls,
                    "strategy_changes": session.metrics.strategy_changes,
                }),
                session.last_error, session.error_count,
                json.dumps(session.metadata),
            ))
            await db.commit()

        return session.id

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_session(row)

    def _row_to_session(self, row: aiosqlite.Row) -> Session:
        """Convert database row to Session entity."""
        metrics_data = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
        metrics = SessionMetrics(
            questions_processed=metrics_data.get("questions_processed", 0),
            predictions_made=metrics_data.get("predictions_made", 0),
            predictions_accurate=metrics_data.get("predictions_accurate", 0),
            tasks_completed=metrics_data.get("tasks_completed", 0),
            tasks_failed=metrics_data.get("tasks_failed", 0),
            total_tokens_used=metrics_data.get("total_tokens_used", 0),
            total_api_calls=metrics_data.get("total_api_calls", 0),
            strategy_changes=metrics_data.get("strategy_changes", 0),
        )

        return Session(
            id=row["id"],
            name=row["name"] or "",
            provider=row["provider"] or "",
            status=SessionStatus(row["status"]),
            model=row["model"] or "",
            initial_prompt=row["initial_prompt"] or "",
            working_directory=row["working_directory"] or "",
            group_id=row["group_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            last_activity_at=datetime.fromisoformat(row["last_activity_at"]) if row["last_activity_at"] else None,
            rate_limited_until=datetime.fromisoformat(row["rate_limited_until"]) if row["rate_limited_until"] else None,
            current_task_id=row["current_task_id"],
            current_prediction_id=row["current_prediction_id"],
            current_strategy=row["current_strategy"] or "default",
            metrics=metrics,
            last_error=row["last_error"],
            error_count=row["error_count"] or 0,
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )

    async def get_sessions_by_status(
        self, status: SessionStatus, limit: int = 100
    ) -> list[Session]:
        """Get sessions by status."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM sessions WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status.value, limit),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_session(row) for row in rows]

    async def get_sessions_by_group(self, group_id: str) -> list[Session]:
        """Get sessions in a group."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM sessions WHERE group_id = ? ORDER BY created_at",
                (group_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_session(row) for row in rows]

    async def get_active_sessions(self) -> list[Session]:
        """Get all active sessions."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM sessions
                   WHERE status IN (?, ?)
                   ORDER BY last_activity_at DESC""",
                (SessionStatus.RUNNING.value, SessionStatus.WAITING_RATE_LIMIT.value),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_session(row) for row in rows]

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and related data."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM tasks WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM questions WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM predictions WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await db.commit()
            return True

    # Prediction operations

    async def save_prediction(self, prediction: Prediction) -> str:
        """Save a prediction."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO predictions (
                    id, session_id, predicted_question, predicted_answer, predicted_context,
                    strategy_used, confidence, reasoning, status, actual_question,
                    similarity_score, created_at, evaluated_at, expires_at,
                    sequence_number, lookahead_position, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.id, prediction.session_id, prediction.predicted_question,
                prediction.predicted_answer, prediction.predicted_context,
                prediction.strategy_used, prediction.confidence, prediction.reasoning,
                prediction.status.value, prediction.actual_question, prediction.similarity_score,
                prediction.created_at.isoformat(),
                prediction.evaluated_at.isoformat() if prediction.evaluated_at else None,
                prediction.expires_at.isoformat() if prediction.expires_at else None,
                prediction.sequence_number, prediction.lookahead_position,
                json.dumps(prediction.metadata),
            ))
            await db.commit()

        return prediction.id

    async def get_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """Get a prediction by ID."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_prediction(row)

    def _row_to_prediction(self, row: aiosqlite.Row) -> Prediction:
        """Convert database row to Prediction entity."""
        return Prediction(
            id=row["id"],
            session_id=row["session_id"],
            predicted_question=row["predicted_question"] or "",
            predicted_answer=row["predicted_answer"],
            predicted_context=row["predicted_context"],
            strategy_used=row["strategy_used"] or "default",
            confidence=row["confidence"] or 0.0,
            reasoning=row["reasoning"],
            status=PredictionStatus(row["status"]),
            actual_question=row["actual_question"],
            similarity_score=row["similarity_score"],
            created_at=datetime.fromisoformat(row["created_at"]),
            evaluated_at=datetime.fromisoformat(row["evaluated_at"]) if row["evaluated_at"] else None,
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            sequence_number=row["sequence_number"] or 0,
            lookahead_position=row["lookahead_position"] or 0,
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )

    async def get_session_predictions(
        self, session_id: str, pending_only: bool = False, limit: int = 100
    ) -> list[Prediction]:
        """Get predictions for a session."""
        await self._ensure_initialized()

        query = "SELECT * FROM predictions WHERE session_id = ?"
        params: list[Any] = [session_id]

        if pending_only:
            query += " AND status = ?"
            params.append(PredictionStatus.PENDING.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_prediction(row) for row in rows]

    async def save_prediction_batch(self, batch: PredictionBatch) -> list[str]:
        """Save a batch of predictions."""
        ids = []
        for prediction in batch.predictions:
            await self.save_prediction(prediction)
            ids.append(prediction.id)
        return ids

    # Question operations

    async def save_question(self, question: Question) -> str:
        """Save a question."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO questions (
                    id, session_id, content, context, expected_output, question_type,
                    source, parent_question_id, prediction_id, processed, answer,
                    answer_at, created_at, priority, sequence_number, embedding_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                question.id, question.session_id, question.content, question.context,
                question.expected_output, question.question_type.value, question.source.value,
                question.parent_question_id, question.prediction_id, int(question.processed),
                question.answer,
                question.answer_at.isoformat() if question.answer_at else None,
                question.created_at.isoformat(), question.priority, question.sequence_number,
                json.dumps(question.embedding) if question.embedding else None,
                json.dumps(question.metadata),
            ))
            await db.commit()

        return question.id

    async def get_question(self, question_id: str) -> Optional[Question]:
        """Get a question by ID."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM questions WHERE id = ?", (question_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_question(row)

    def _row_to_question(self, row: aiosqlite.Row) -> Question:
        """Convert database row to Question entity."""
        return Question(
            id=row["id"],
            session_id=row["session_id"],
            content=row["content"] or "",
            context=row["context"],
            expected_output=row["expected_output"],
            question_type=QuestionType(row["question_type"]),
            source=QuestionSource(row["source"]),
            parent_question_id=row["parent_question_id"],
            prediction_id=row["prediction_id"],
            processed=bool(row["processed"]),
            answer=row["answer"],
            answer_at=datetime.fromisoformat(row["answer_at"]) if row["answer_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            priority=row["priority"] or 0,
            sequence_number=row["sequence_number"] or 0,
            embedding=json.loads(row["embedding_json"]) if row["embedding_json"] else None,
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )

    async def get_session_questions(self, session_id: str, limit: int = 100) -> list[Question]:
        """Get questions for a session."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM questions WHERE session_id = ? ORDER BY sequence_number LIMIT ?",
                (session_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_question(row) for row in rows]

    # Task operations

    async def save_task(self, task: Task) -> str:
        """Save a task."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO tasks (
                    id, session_id, question_id, description, instructions, task_type,
                    priority, status, result_json, retry_count, max_retries,
                    depends_on_json, blocked_by, created_at, started_at, completed_at,
                    context, working_directory, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id, task.session_id, task.question_id, task.description,
                task.instructions, task.task_type.value, task.priority.value,
                task.status.value,
                json.dumps(task.result.to_dict()) if task.result else None,
                task.retry_count, task.max_retries,
                json.dumps(task.depends_on), task.blocked_by,
                task.created_at.isoformat(),
                task.started_at.isoformat() if task.started_at else None,
                task.completed_at.isoformat() if task.completed_at else None,
                task.context, task.working_directory,
                json.dumps(task.metadata),
            ))
            await db.commit()

        return task.id

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_task(row)

    def _row_to_task(self, row: aiosqlite.Row) -> Task:
        """Convert database row to Task entity."""
        result_data = json.loads(row["result_json"]) if row["result_json"] else None
        result = None
        if result_data:
            result = TaskResult(
                success=result_data.get("success", False),
                output=result_data.get("output", ""),
                error=result_data.get("error"),
                files_modified=result_data.get("files_modified", []),
                files_created=result_data.get("files_created", []),
                tokens_used=result_data.get("tokens_used", 0),
                execution_time=result_data.get("execution_time", 0.0),
            )

        return Task(
            id=row["id"],
            session_id=row["session_id"],
            question_id=row["question_id"] or "",
            description=row["description"] or "",
            instructions=row["instructions"] or "",
            task_type=TaskType(row["task_type"]),
            priority=TaskPriority(row["priority"]),
            status=TaskStatus(row["status"]),
            result=result,
            retry_count=row["retry_count"] or 0,
            max_retries=row["max_retries"] or 3,
            depends_on=json.loads(row["depends_on_json"]) if row["depends_on_json"] else [],
            blocked_by=row["blocked_by"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            context=row["context"],
            working_directory=row["working_directory"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )

    async def get_session_tasks(self, session_id: str, limit: int = 100) -> list[Task]:
        """Get tasks for a session."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tasks WHERE session_id = ? ORDER BY created_at LIMIT ?",
                (session_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_task(row) for row in rows]

    # Metrics

    async def get_session_metrics(self, session_id: str) -> dict[str, Any]:
        """Get comprehensive metrics for a session."""
        await self._ensure_initialized()

        session = await self.get_session(session_id)
        if not session:
            return {}

        predictions = await self.get_session_predictions(session_id)
        questions = await self.get_session_questions(session_id)
        tasks = await self.get_session_tasks(session_id)

        return {
            "session_id": session_id,
            "status": session.status.value,
            "duration": session.duration,
            "questions": {
                "total": len(questions),
                "processed": sum(1 for q in questions if q.processed),
            },
            "predictions": {
                "total": len(predictions),
                "matched": sum(1 for p in predictions if p.status == PredictionStatus.MATCHED),
                "accuracy": session.metrics.prediction_accuracy,
            },
            "tasks": {
                "total": len(tasks),
                "completed": sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
                "failed": sum(1 for t in tasks if t.status == TaskStatus.FAILED),
            },
            "tokens_used": session.metrics.total_tokens_used,
            "strategy_changes": session.metrics.strategy_changes,
        }

    async def get_strategy_performance(
        self, strategy_name: str, since: Optional[datetime] = None
    ) -> dict[str, Any]:
        """Get performance metrics for a strategy."""
        await self._ensure_initialized()

        query = "SELECT * FROM predictions WHERE strategy_used = ?"
        params: list[Any] = [strategy_name]

        if since:
            query += " AND created_at >= ?"
            params.append(since.isoformat())

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        predictions = [self._row_to_prediction(row) for row in rows]

        if not predictions:
            return {
                "strategy_name": strategy_name,
                "total_predictions": 0,
                "accuracy": 0.0,
            }

        matched = sum(1 for p in predictions if p.status == PredictionStatus.MATCHED)
        evaluated = sum(1 for p in predictions if p.is_evaluated)
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
        avg_similarity = sum(
            p.similarity_score for p in predictions if p.similarity_score
        ) / max(1, sum(1 for p in predictions if p.similarity_score))

        return {
            "strategy_name": strategy_name,
            "total_predictions": len(predictions),
            "evaluated": evaluated,
            "matched": matched,
            "accuracy": matched / evaluated if evaluated else 0.0,
            "average_confidence": avg_confidence,
            "average_similarity": avg_similarity,
        }
