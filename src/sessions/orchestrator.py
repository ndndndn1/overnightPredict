"""Session orchestrator for managing parallel coding sessions."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable

import structlog

from src.core.config import Settings
from src.core.engine import OvernightEngine
from src.core.models import (
    OrchestratorState,
    ProjectContext,
    SessionState,
    SessionStatus,
)
from src.sessions.checkpoint import CheckpointManager

logger = structlog.get_logger(__name__)


class SessionOrchestrator:
    """
    Orchestrates multiple parallel coding sessions.

    Features:
    - Parallel session management
    - Auto-scaling based on workload
    - Load balancing across sessions
    - Session health monitoring
    - Coordinated project decomposition
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the session orchestrator."""
        self.settings = settings
        self._engines: dict[str, OvernightEngine] = {}
        self._session_tasks: dict[str, asyncio.Task[None]] = {}
        self._state = OrchestratorState()
        self._checkpoint_manager = CheckpointManager()

        # Control
        self._is_running = False
        self._shutdown_event = asyncio.Event()
        self._scale_lock = asyncio.Lock()

        # Callbacks
        self._on_session_complete: list[Callable[[str, SessionState], Any]] = []
        self._on_all_complete: list[Callable[[ProjectContext], Any]] = []

        # Monitoring
        self._health_check_interval = 30  # seconds
        self._health_check_task: asyncio.Task[None] | None = None

    async def initialize_project(
        self,
        project_context: ProjectContext,
    ) -> str:
        """
        Initialize an enterprise project for parallel development.

        Args:
            project_context: Project context with requirements and components

        Returns:
            Project ID
        """
        self._state.project_context = project_context

        logger.info(
            "Project initialized",
            project_name=project_context.name,
            components=len(project_context.pending_components),
        )

        return project_context.id

    async def start(
        self,
        initial_sessions: int | None = None,
    ) -> None:
        """
        Start the orchestrator and begin parallel sessions.

        Args:
            initial_sessions: Number of initial sessions to spawn
        """
        if self._is_running:
            logger.warning("Orchestrator already running")
            return

        self._is_running = True
        self._shutdown_event.clear()

        # Determine initial session count
        if initial_sessions is None:
            initial_sessions = self.settings.sessions.min_sessions

        initial_sessions = max(
            self.settings.sessions.min_sessions,
            min(initial_sessions, self.settings.sessions.max_sessions),
        )

        logger.info(
            "Starting orchestrator",
            initial_sessions=initial_sessions,
            project=self._state.project_context.name if self._state.project_context else None,
        )

        # Start initial sessions
        await self._spawn_sessions(initial_sessions)

        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        # Start auto-scaling if enabled
        if self.settings.sessions.auto_scale_enabled:
            asyncio.create_task(self._auto_scale_loop())

    async def stop(self, graceful: bool = True) -> None:
        """
        Stop the orchestrator and all sessions.

        Args:
            graceful: If True, wait for sessions to complete current work
        """
        logger.info("Stopping orchestrator", graceful=graceful)

        self._is_running = False
        self._shutdown_event.set()

        if graceful:
            # Wait for sessions to complete current iteration
            for session_id, task in list(self._session_tasks.items()):
                engine = self._engines.get(session_id)
                if engine:
                    await engine.shutdown()

                # Wait with timeout
                try:
                    await asyncio.wait_for(task, timeout=30)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Session shutdown timeout",
                        session_id=session_id,
                    )
                    task.cancel()
        else:
            # Cancel all tasks immediately
            for task in self._session_tasks.values():
                task.cancel()

        # Cleanup
        self._session_tasks.clear()
        self._engines.clear()

        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()

        # Wait for pending checkpoints
        await self._checkpoint_manager.wait_for_pending_saves()

        logger.info("Orchestrator stopped")

    async def add_session(
        self,
        topic: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a new session to the orchestrator.

        Args:
            topic: Optional topic for the session
            context: Optional initial context

        Returns:
            Session ID
        """
        async with self._scale_lock:
            if len(self._session_tasks) >= self.settings.sessions.max_sessions:
                raise RuntimeError("Maximum session limit reached")

            session_id = await self._create_session(topic, context)
            return session_id

    async def remove_session(self, session_id: str) -> bool:
        """
        Remove a session from the orchestrator.

        Args:
            session_id: ID of session to remove

        Returns:
            True if session was removed
        """
        async with self._scale_lock:
            if session_id not in self._session_tasks:
                return False

            # Stop the session
            engine = self._engines.get(session_id)
            if engine:
                await engine.shutdown()

            # Cancel task
            task = self._session_tasks.pop(session_id, None)
            if task:
                task.cancel()

            # Cleanup
            self._engines.pop(session_id, None)
            self._state.active_sessions.pop(session_id, None)

            logger.info("Session removed", session_id=session_id)
            return True

    async def get_session(self, session_id: str) -> SessionState | None:
        """Get the state of a specific session."""
        return self._state.active_sessions.get(session_id)

    def get_all_sessions(self) -> list[SessionState]:
        """Get all active session states."""
        return list(self._state.active_sessions.values())

    async def get_orchestrator_status(self) -> dict[str, Any]:
        """Get current orchestrator status."""
        active_count = len(self._session_tasks)
        running_count = sum(
            1
            for s in self._state.active_sessions.values()
            if s.status == SessionStatus.RUNNING
        )

        return {
            "is_running": self._is_running,
            "project": (
                self._state.project_context.model_dump()
                if self._state.project_context
                else None
            ),
            "sessions": {
                "total": active_count,
                "running": running_count,
                "paused": active_count - running_count,
                "completed": len(self._state.completed_sessions),
            },
            "auto_scaling": self.settings.sessions.auto_scale_enabled,
            "started_at": self._state.started_at.isoformat(),
        }

    async def get_global_metrics(self) -> dict[str, Any]:
        """Get aggregate metrics across all sessions."""
        total_questions = 0
        total_answers = 0
        total_artifacts = 0
        total_predictions = 0
        accurate_predictions = 0

        for session in self._state.active_sessions.values():
            total_questions += len(session.questions)
            total_answers += len(session.answers)
            total_artifacts += len(session.artifacts)

            if session.accuracy_metrics:
                total_predictions += session.accuracy_metrics.total_predictions
                accurate_predictions += session.accuracy_metrics.accurate_predictions

        accuracy_rate = (
            accurate_predictions / total_predictions if total_predictions > 0 else 0.0
        )

        return {
            "total_questions": total_questions,
            "total_answers": total_answers,
            "total_artifacts": total_artifacts,
            "total_predictions": total_predictions,
            "accurate_predictions": accurate_predictions,
            "overall_accuracy": accuracy_rate,
            "sessions_completed": len(self._state.completed_sessions),
        }

    async def _spawn_sessions(self, count: int) -> list[str]:
        """Spawn multiple sessions."""
        session_ids = []

        # Decompose project into session topics
        topics = await self._decompose_project_topics(count)

        for i, topic in enumerate(topics):
            context = self._build_session_context(i, topic)
            session_id = await self._create_session(topic, context)
            session_ids.append(session_id)

        return session_ids

    async def _create_session(
        self,
        topic: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Create and start a single session."""
        # Create engine for this session
        engine = OvernightEngine(settings=self.settings)

        # Generate topic if not provided
        if topic is None and self._state.project_context:
            pending = self._state.project_context.pending_components
            if pending:
                topic = f"Implement {pending[0]}"
            else:
                topic = f"Work on {self._state.project_context.name}"
        elif topic is None:
            topic = "General development session"

        # Create session
        session = await engine.create_session(
            topic=topic,
            project_context=self._state.project_context,
            initial_context=context,
        )

        # Store references
        self._engines[session.id] = engine
        self._state.active_sessions[session.id] = session

        # Start session task
        task = asyncio.create_task(self._run_session(session.id))
        self._session_tasks[session.id] = task

        logger.info(
            "Session created and started",
            session_id=session.id,
            topic=topic,
        )

        return session.id

    async def _run_session(self, session_id: str) -> None:
        """Run a session's main loop."""
        engine = self._engines.get(session_id)
        if not engine:
            return

        try:
            await engine.start_session(session_id)
        except asyncio.CancelledError:
            logger.info("Session cancelled", session_id=session_id)
        except Exception as e:
            logger.error(
                "Session error",
                session_id=session_id,
                error=str(e),
            )
        finally:
            # Update state
            session = engine.get_session(session_id)
            if session:
                self._state.active_sessions[session_id] = session

                # Checkpoint final state
                await self._checkpoint_manager.save_checkpoint(session, sync=True)

                # Mark as completed
                if session.status == SessionStatus.COMPLETED:
                    self._state.completed_sessions.append(session_id)
                    await self._notify_session_complete(session_id, session)

                    # Check if all sessions complete
                    await self._check_project_completion()

    async def _decompose_project_topics(self, count: int) -> list[str]:
        """Decompose project into topics for parallel sessions."""
        topics = []

        if self._state.project_context:
            pending = self._state.project_context.pending_components

            # Distribute components across sessions
            for i in range(count):
                if i < len(pending):
                    topics.append(f"Implement {pending[i]}")
                else:
                    # Additional sessions for testing, docs, etc.
                    extra_tasks = [
                        "Write unit tests",
                        "Integration testing",
                        "Documentation",
                        "Performance optimization",
                        "Security review",
                    ]
                    extra_idx = (i - len(pending)) % len(extra_tasks)
                    topics.append(extra_tasks[extra_idx])
        else:
            # No project context, use generic topics
            topics = [f"Development session {i + 1}" for i in range(count)]

        return topics

    def _build_session_context(
        self,
        session_index: int,
        topic: str,
    ) -> dict[str, Any]:
        """Build initial context for a session."""
        context: dict[str, Any] = {
            "session_index": session_index,
            "topic": topic,
            "autonomous_mode": True,
        }

        if self._state.project_context:
            project = self._state.project_context
            context["project"] = {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "architecture_type": project.architecture_type,
                "current_phase": project.current_phase,
                "pending_components": project.pending_components,
                "completed_components": project.completed_components,
                "target_languages": project.target_languages,
            }

        return context

    async def _health_check_loop(self) -> None:
        """Periodically check session health."""
        while self._is_running:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._check_session_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error", error=str(e))

    async def _check_session_health(self) -> None:
        """Check health of all sessions and restart unhealthy ones."""
        timeout = timedelta(minutes=self.settings.sessions.session_timeout_minutes)
        now = datetime.utcnow()

        for session_id, session in list(self._state.active_sessions.items()):
            # Check for timeout
            if session.status == SessionStatus.RUNNING:
                inactive_time = now - session.last_activity
                if inactive_time > timeout:
                    logger.warning(
                        "Session timed out",
                        session_id=session_id,
                        inactive_minutes=inactive_time.total_seconds() / 60,
                    )
                    await self.remove_session(session_id)

            # Check for failed sessions
            if session.status == SessionStatus.FAILED:
                logger.warning(
                    "Removing failed session",
                    session_id=session_id,
                )
                await self.remove_session(session_id)

    async def _auto_scale_loop(self) -> None:
        """Auto-scale sessions based on workload."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._evaluate_scaling()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Auto-scale error", error=str(e))

    async def _evaluate_scaling(self) -> None:
        """Evaluate and apply auto-scaling decisions."""
        async with self._scale_lock:
            current_count = len(self._session_tasks)
            min_sessions = self.settings.sessions.min_sessions
            max_sessions = self.settings.sessions.max_sessions

            # Calculate load
            running_sessions = sum(
                1
                for s in self._state.active_sessions.values()
                if s.status == SessionStatus.RUNNING
            )

            load = running_sessions / current_count if current_count > 0 else 0

            # Scale up if load is high and pending work exists
            if load >= self.settings.sessions.scale_up_threshold:
                if current_count < max_sessions and self._has_pending_work():
                    new_count = min(current_count + 2, max_sessions)
                    to_spawn = new_count - current_count

                    logger.info(
                        "Scaling up",
                        current=current_count,
                        target=new_count,
                        load=load,
                    )

                    await self._spawn_sessions(to_spawn)

            # Scale down if load is low
            elif load <= self.settings.sessions.scale_down_threshold:
                if current_count > min_sessions:
                    # Remove idle sessions
                    idle_sessions = [
                        sid
                        for sid, s in self._state.active_sessions.items()
                        if s.status in [SessionStatus.PAUSED, SessionStatus.WAITING_INPUT]
                    ]

                    for sid in idle_sessions[: current_count - min_sessions]:
                        await self.remove_session(sid)
                        logger.info(
                            "Scaled down",
                            removed_session=sid,
                            load=load,
                        )

    def _has_pending_work(self) -> bool:
        """Check if there's pending work for new sessions."""
        if self._state.project_context:
            return len(self._state.project_context.pending_components) > 0
        return False

    async def _check_project_completion(self) -> None:
        """Check if the entire project is complete."""
        if not self._state.project_context:
            return

        # All sessions completed
        active_running = sum(
            1
            for s in self._state.active_sessions.values()
            if s.status == SessionStatus.RUNNING
        )

        if active_running == 0:
            logger.info(
                "Project completed",
                project_name=self._state.project_context.name,
                sessions_completed=len(self._state.completed_sessions),
            )

            for callback in self._on_all_complete:
                try:
                    result = callback(self._state.project_context)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error("Project completion callback error", error=str(e))

    async def _notify_session_complete(
        self, session_id: str, session: SessionState
    ) -> None:
        """Notify callbacks about session completion."""
        for callback in self._on_session_complete:
            try:
                result = callback(session_id, session)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    "Session completion callback error",
                    session_id=session_id,
                    error=str(e),
                )

    def on_session_complete(
        self, callback: Callable[[str, SessionState], Any]
    ) -> None:
        """Register callback for session completion."""
        self._on_session_complete.append(callback)

    def on_all_complete(
        self, callback: Callable[[ProjectContext], Any]
    ) -> None:
        """Register callback for project completion."""
        self._on_all_complete.append(callback)
