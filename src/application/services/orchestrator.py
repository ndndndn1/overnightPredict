"""
Orchestrator service - Manages multiple sessions in parallel.

The central coordinator that:
- Creates and manages multiple parallel sessions
- Distributes work across providers
- Handles cross-session coordination
- Provides unified monitoring and control
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Optional

from src.core.config.settings import Settings, OrchestratorConfig, LLMProvider
from src.core.utils.id_generator import generate_group_id, generate_session_id
from src.core.utils.logging import get_logger
from src.domain.entities.session import Session, SessionStatus
from src.domain.entities.question import Question, QuestionSource, QuestionType
from src.domain.interfaces.llm_provider import ILLMProvider
from src.domain.interfaces.session_repository import ISessionRepository
from src.domain.interfaces.event_bus import IEventBus
from src.domain.interfaces.context_store import IContextStore, IContextSynchronizer
from src.domain.value_objects.context import Context, ContextType, SharedContext
from src.application.services.session_manager import SessionManager, QuestionProcessingResult
from src.application.services.forecaster import Forecaster, LLMPredictionStrategy
from src.application.services.evaluator import Evaluator
from src.application.services.meta_tuner import MetaTuner
from src.infrastructure.llm_providers.provider_factory import LLMProviderFactory


class Orchestrator:
    """
    Orchestrator for managing multiple parallel sessions.

    Provides:
    - Multi-session creation and management
    - Cross-session context sharing
    - Unified monitoring dashboard
    - Provider-agnostic coordination
    """

    def __init__(
        self,
        settings: Settings,
        repository: ISessionRepository,
        context_store: IContextStore,
        event_bus: IEventBus,
        context_sync: Optional[IContextSynchronizer] = None,
    ):
        """Initialize the orchestrator.

        Args:
            settings: Application settings.
            repository: Session repository.
            context_store: Context store.
            event_bus: Event bus.
            context_sync: Optional context synchronizer.
        """
        self._settings = settings
        self._repository = repository
        self._context_store = context_store
        self._event_bus = event_bus
        self._context_sync = context_sync
        self._config = settings.orchestrator

        self._logger = get_logger("orchestrator")

        # Session management
        self._sessions: dict[str, SessionManager] = {}
        self._groups: dict[str, list[str]] = {}  # group_id -> [session_ids]

        # LLM providers
        self._providers: dict[str, ILLMProvider] = {}

        # Shared services
        self._forecaster: Optional[Forecaster] = None
        self._evaluator: Optional[Evaluator] = None
        self._meta_tuner: Optional[MetaTuner] = None

        # Running state
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        self._logger.info("Initializing orchestrator")

        # Create providers
        self._providers = LLMProviderFactory.create_all_enabled(self._settings)
        self._logger.info(f"Created {len(self._providers)} providers")

        # Initialize shared services
        self._evaluator = Evaluator(self._settings.prediction)
        await self._evaluator.initialize()

        self._forecaster = Forecaster(self._settings.prediction)
        self._meta_tuner = MetaTuner(self._settings.prediction, self._event_bus)

        # Register prediction strategies for each provider
        for name, provider in self._providers.items():
            strategy = LLMPredictionStrategy(provider, self._settings.prediction)
            self._forecaster.register_strategy(strategy)
            self._meta_tuner.register_strategy(strategy)

        self._logger.info("Orchestrator initialized")

    async def start(self) -> None:
        """Start the orchestrator."""
        if self._running:
            return

        self._running = True

        # Start health check loop
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        # Recover active sessions
        await self._recover_sessions()

        self._logger.info("Orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        self._running = False

        # Stop health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop all sessions
        for manager in self._sessions.values():
            await manager.stop()

        self._sessions.clear()

        if self._evaluator:
            await self._evaluator.cleanup()

        self._logger.info("Orchestrator stopped")

    async def create_session(
        self,
        provider: LLMProvider,
        name: str = "",
        initial_prompt: str = "",
        working_directory: str = "",
        group_id: Optional[str] = None,
    ) -> str:
        """Create a new session.

        Args:
            provider: LLM provider to use.
            name: Session name.
            initial_prompt: Initial prompt/context.
            working_directory: Working directory for file operations.
            group_id: Optional group for context sharing.

        Returns:
            Session ID.
        """
        # Check session limit
        if len(self._sessions) >= self._config.max_sessions:
            raise RuntimeError(f"Maximum sessions ({self._config.max_sessions}) reached")

        # Get provider
        llm_provider = self._providers.get(provider.value)
        if not llm_provider:
            raise ValueError(f"Provider {provider.value} not available")

        # Create session
        session = Session(
            name=name or f"Session-{len(self._sessions) + 1}",
            provider=provider.value,
            model=llm_provider.model_name,
            initial_prompt=initial_prompt,
            working_directory=working_directory,
            group_id=group_id,
        )

        # Create session manager
        manager = SessionManager(
            session=session,
            llm_provider=llm_provider,
            repository=self._repository,
            forecaster=self._forecaster,
            evaluator=self._evaluator,
            meta_tuner=self._meta_tuner,
            event_bus=self._event_bus,
        )

        # Add initial context if provided
        if initial_prompt:
            await manager.add_context(initial_prompt, ContextType.REQUIREMENT)

        # Register session
        self._sessions[session.id] = manager
        await self._repository.save_session(session)

        # Add to group if specified
        if group_id:
            if group_id not in self._groups:
                self._groups[group_id] = []
            self._groups[group_id].append(session.id)

        self._logger.info(
            "Session created",
            session_id=session.id,
            provider=provider.value,
            group=group_id,
        )

        return session.id

    async def create_session_group(
        self,
        providers: list[LLMProvider],
        initial_prompt: str,
        working_directory: str = "",
    ) -> str:
        """Create a group of sessions with shared context.

        Args:
            providers: List of providers to use.
            initial_prompt: Shared initial prompt.
            working_directory: Shared working directory.

        Returns:
            Group ID.
        """
        group_id = generate_group_id()

        for provider in providers:
            await self.create_session(
                provider=provider,
                initial_prompt=initial_prompt,
                working_directory=working_directory,
                group_id=group_id,
            )

        self._logger.info(
            "Session group created",
            group_id=group_id,
            session_count=len(providers),
        )

        return group_id

    async def start_session(self, session_id: str) -> None:
        """Start a specific session.

        Args:
            session_id: Session to start.
        """
        manager = self._sessions.get(session_id)
        if manager:
            await manager.start()

    async def stop_session(self, session_id: str) -> None:
        """Stop a specific session.

        Args:
            session_id: Session to stop.
        """
        manager = self._sessions.get(session_id)
        if manager:
            await manager.stop()

    async def start_all_sessions(self) -> None:
        """Start all sessions concurrently."""
        await asyncio.gather(*[
            manager.start() for manager in self._sessions.values()
        ])

    async def stop_all_sessions(self) -> None:
        """Stop all sessions concurrently."""
        await asyncio.gather(*[
            manager.stop() for manager in self._sessions.values()
        ])

    async def process_question(
        self,
        session_id: str,
        content: str,
        question_type: QuestionType = QuestionType.OTHER,
    ) -> QuestionProcessingResult:
        """Process a question in a specific session.

        Args:
            session_id: Target session.
            content: Question content.
            question_type: Type of question.

        Returns:
            QuestionProcessingResult with answer and metadata.
        """
        manager = self._sessions.get(session_id)
        if not manager:
            raise ValueError(f"Session {session_id} not found")

        question = Question(
            session_id=session_id,
            content=content,
            question_type=question_type,
            source=QuestionSource.USER,
        )

        return await manager.process_question(question)

    async def broadcast_context(
        self,
        group_id: str,
        content: str,
        context_type: ContextType,
        from_session: Optional[str] = None,
    ) -> None:
        """Broadcast context to all sessions in a group.

        Args:
            group_id: Target group.
            content: Context content.
            context_type: Type of context.
            from_session: Optional source session.
        """
        if group_id not in self._groups:
            return

        context = Context(
            content=content,
            context_type=context_type,
            source=from_session or "orchestrator",
        )

        # Share through context store
        await self._context_store.share_context(
            context=context,
            session_id=from_session or "orchestrator",
            group_id=group_id,
        )

        # Sync if synchronizer available
        if self._context_sync:
            shared = SharedContext(
                context=context,
                shared_by=from_session or "orchestrator",
                group_id=group_id,
            )
            await self._context_sync.push(shared)

        # Add to each session's context
        for session_id in self._groups[group_id]:
            if session_id != from_session:
                manager = self._sessions.get(session_id)
                if manager:
                    await manager.add_context(content, context_type)

        self._logger.info(
            "Context broadcast",
            group_id=group_id,
            context_type=context_type.value,
        )

    def get_session_status(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get status for a specific session."""
        manager = self._sessions.get(session_id)
        if manager:
            return manager.get_status()
        return None

    def get_all_sessions_status(self) -> list[dict[str, Any]]:
        """Get status for all sessions."""
        return [manager.get_status() for manager in self._sessions.values()]

    def get_group_status(self, group_id: str) -> dict[str, Any]:
        """Get status for a session group."""
        if group_id not in self._groups:
            return {"error": "Group not found"}

        session_ids = self._groups[group_id]
        sessions = [
            self.get_session_status(sid)
            for sid in session_ids
        ]

        # Aggregate metrics
        total_questions = sum(s["metrics"]["questions_processed"] for s in sessions if s)
        total_predictions = sum(s["metrics"]["predictions_made"] for s in sessions if s)
        avg_accuracy = sum(s["metrics"]["prediction_accuracy"] for s in sessions if s) / len(sessions) if sessions else 0

        return {
            "group_id": group_id,
            "session_count": len(session_ids),
            "sessions": sessions,
            "aggregate": {
                "total_questions": total_questions,
                "total_predictions": total_predictions,
                "average_accuracy": avg_accuracy,
            },
        }

    async def adjust_session_strategy(
        self,
        session_id: str,
        strategy_name: str,
    ) -> bool:
        """Manually adjust a session's strategy.

        Args:
            session_id: Target session.
            strategy_name: New strategy name.

        Returns:
            True if successful.
        """
        manager = self._sessions.get(session_id)
        if not manager:
            return False

        manager.session.change_strategy(strategy_name)
        await self._repository.save_session(manager.session)

        self._logger.info(
            "Strategy adjusted",
            session_id=session_id,
            strategy=strategy_name,
        )

        return True

    async def _recover_sessions(self) -> None:
        """Recover active sessions from storage."""
        active_sessions = await self._repository.get_active_sessions()

        for session in active_sessions:
            provider = self._providers.get(session.provider)
            if not provider:
                self._logger.warning(
                    f"Provider {session.provider} not available for session {session.id}"
                )
                continue

            manager = SessionManager(
                session=session,
                llm_provider=provider,
                repository=self._repository,
                forecaster=self._forecaster,
                evaluator=self._evaluator,
                meta_tuner=self._meta_tuner,
                event_bus=self._event_bus,
            )

            self._sessions[session.id] = manager

            # Add to group if applicable
            if session.group_id:
                if session.group_id not in self._groups:
                    self._groups[session.group_id] = []
                self._groups[session.group_id].append(session.id)

        self._logger.info(f"Recovered {len(active_sessions)} sessions")

    async def _health_check_loop(self) -> None:
        """Periodic health check for all sessions."""
        while self._running:
            try:
                await asyncio.sleep(self._config.health_check_interval)

                for session_id, manager in list(self._sessions.items()):
                    session = manager.session

                    # Check for timeout
                    if session.last_activity_at:
                        elapsed = (datetime.utcnow() - session.last_activity_at).total_seconds()
                        if elapsed > self._config.session_timeout:
                            self._logger.warning(
                                "Session timed out",
                                session_id=session_id,
                            )
                            await manager.stop()

                    # Check provider health
                    provider = self._providers.get(session.provider)
                    if provider and not await provider.health_check():
                        self._logger.warning(
                            "Provider unhealthy",
                            session_id=session_id,
                            provider=session.provider,
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Health check error", error=str(e))

    @property
    def session_count(self) -> int:
        """Get current session count."""
        return len(self._sessions)

    @property
    def group_count(self) -> int:
        """Get current group count."""
        return len(self._groups)

    @property
    def available_providers(self) -> list[str]:
        """Get list of available providers."""
        return list(self._providers.keys())
