"""
Session Controller - Interactive session management.

Provides high-level control over sessions with user-friendly methods.
"""

from typing import Any, Optional

from src.core.config.settings import LLMProvider
from src.domain.entities.question import QuestionType
from src.domain.value_objects.context import ContextType
from src.application.services.orchestrator import Orchestrator


class SessionController:
    """
    Controller for managing sessions through a simplified interface.

    Wraps the Orchestrator with user-friendly methods.
    """

    def __init__(self, orchestrator: Orchestrator):
        """Initialize the controller.

        Args:
            orchestrator: The orchestrator instance.
        """
        self._orchestrator = orchestrator

    @property
    def available_providers(self) -> list[str]:
        """Get available providers."""
        return self._orchestrator.available_providers

    @property
    def session_count(self) -> int:
        """Get current session count."""
        return self._orchestrator.session_count

    async def create_session(
        self,
        provider: str,
        name: str = "",
        initial_prompt: str = "",
        working_directory: str = "",
    ) -> str:
        """Create a new session.

        Args:
            provider: Provider name (openai, deepseek, claude).
            name: Session name.
            initial_prompt: Initial requirements/context.
            working_directory: Working directory for file operations.

        Returns:
            Session ID.
        """
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "deepseek": LLMProvider.DEEPSEEK,
            "claude": LLMProvider.CLAUDE,
            "anthropic": LLMProvider.ANTHROPIC,
        }

        llm_provider = provider_map.get(provider.lower())
        if not llm_provider:
            raise ValueError(f"Unknown provider: {provider}")

        return await self._orchestrator.create_session(
            provider=llm_provider,
            name=name,
            initial_prompt=initial_prompt,
            working_directory=working_directory,
        )

    async def create_parallel_sessions(
        self,
        providers: list[str],
        initial_prompt: str,
        working_directory: str = "",
    ) -> str:
        """Create a group of parallel sessions with shared context.

        Args:
            providers: List of provider names.
            initial_prompt: Shared initial prompt.
            working_directory: Shared working directory.

        Returns:
            Group ID.
        """
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "deepseek": LLMProvider.DEEPSEEK,
            "claude": LLMProvider.CLAUDE,
            "anthropic": LLMProvider.ANTHROPIC,
        }

        llm_providers = []
        for p in providers:
            if p.lower() in provider_map:
                llm_providers.append(provider_map[p.lower()])

        return await self._orchestrator.create_session_group(
            providers=llm_providers,
            initial_prompt=initial_prompt,
            working_directory=working_directory,
        )

    async def start(self, session_id: str) -> None:
        """Start a session."""
        await self._orchestrator.start_session(session_id)

    async def stop(self, session_id: str) -> None:
        """Stop a session."""
        await self._orchestrator.stop_session(session_id)

    async def start_all(self) -> None:
        """Start all sessions."""
        await self._orchestrator.start_all_sessions()

    async def stop_all(self) -> None:
        """Stop all sessions."""
        await self._orchestrator.stop_all_sessions()

    async def ask(
        self,
        session_id: str,
        question: str,
        question_type: str = "other",
    ) -> str:
        """Ask a question in a session.

        Args:
            session_id: Target session.
            question: Question text.
            question_type: Type (initial, followup, error, feature, etc.).

        Returns:
            Answer string.
        """
        type_map = {
            "initial": QuestionType.INITIAL,
            "followup": QuestionType.FOLLOWUP,
            "clarification": QuestionType.CLARIFICATION,
            "error": QuestionType.ERROR,
            "feature": QuestionType.FEATURE,
            "refactor": QuestionType.REFACTOR,
            "review": QuestionType.REVIEW,
            "test": QuestionType.TEST,
            "documentation": QuestionType.DOCUMENTATION,
            "other": QuestionType.OTHER,
        }

        q_type = type_map.get(question_type.lower(), QuestionType.OTHER)

        return await self._orchestrator.process_question(
            session_id,
            question,
            q_type,
        )

    async def share_context(
        self,
        group_id: str,
        content: str,
        context_type: str = "code",
    ) -> None:
        """Share context with a session group.

        Args:
            group_id: Target group.
            content: Context content.
            context_type: Type (code, conversation, error, requirement).
        """
        type_map = {
            "code": ContextType.CODE,
            "conversation": ContextType.CONVERSATION,
            "error": ContextType.ERROR,
            "requirement": ContextType.REQUIREMENT,
            "file_system": ContextType.FILE_SYSTEM,
            "analysis": ContextType.ANALYSIS,
        }

        c_type = type_map.get(context_type.lower(), ContextType.CODE)

        await self._orchestrator.broadcast_context(
            group_id,
            content,
            c_type,
        )

    def get_status(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get session status."""
        return self._orchestrator.get_session_status(session_id)

    def get_all_status(self) -> list[dict[str, Any]]:
        """Get all sessions status."""
        return self._orchestrator.get_all_sessions_status()

    def get_group_status(self, group_id: str) -> dict[str, Any]:
        """Get group status."""
        return self._orchestrator.get_group_status(group_id)

    async def change_strategy(
        self,
        session_id: str,
        strategy: str,
    ) -> bool:
        """Change a session's prediction strategy.

        Args:
            session_id: Target session.
            strategy: New strategy name.

        Returns:
            True if successful.
        """
        return await self._orchestrator.adjust_session_strategy(
            session_id,
            strategy,
        )
