"""Context store interface."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from src.domain.value_objects.context import Context, ContextSnapshot, SharedContext


class IContextStore(ABC):
    """
    Interface for context storage and retrieval.

    Handles both local and shared context across sessions.
    """

    @abstractmethod
    async def save_context(
        self,
        session_id: str,
        context: Context,
    ) -> str:
        """
        Save a context.

        Args:
            session_id: Session identifier.
            context: Context to save.

        Returns:
            Context ID.
        """
        ...

    @abstractmethod
    async def get_context(
        self,
        context_id: str,
    ) -> Optional[Context]:
        """
        Get a context by ID.

        Args:
            context_id: Context identifier.

        Returns:
            Context or None if not found.
        """
        ...

    @abstractmethod
    async def get_session_contexts(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[Context]:
        """
        Get contexts for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of contexts.

        Returns:
            List of contexts.
        """
        ...

    @abstractmethod
    async def save_snapshot(
        self,
        snapshot: ContextSnapshot,
    ) -> str:
        """
        Save a context snapshot.

        Args:
            snapshot: Snapshot to save.

        Returns:
            Snapshot ID.
        """
        ...

    @abstractmethod
    async def get_snapshot(
        self,
        snapshot_id: str,
    ) -> Optional[ContextSnapshot]:
        """
        Get a snapshot by ID.

        Args:
            snapshot_id: Snapshot identifier.

        Returns:
            Snapshot or None if not found.
        """
        ...

    @abstractmethod
    async def get_latest_snapshot(
        self,
        session_id: str,
    ) -> Optional[ContextSnapshot]:
        """
        Get the latest snapshot for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Latest snapshot or None.
        """
        ...

    # Shared context operations

    @abstractmethod
    async def share_context(
        self,
        context: Context,
        session_id: str,
        group_id: str,
        expires_at: Optional[datetime] = None,
        priority: int = 0,
    ) -> str:
        """
        Share a context with a group.

        Args:
            context: Context to share.
            session_id: Session sharing the context.
            group_id: Group to share with.
            expires_at: Optional expiration time.
            priority: Sharing priority.

        Returns:
            Shared context ID.
        """
        ...

    @abstractmethod
    async def get_shared_contexts(
        self,
        group_id: str,
        limit: int = 50,
        min_priority: int = 0,
    ) -> list[SharedContext]:
        """
        Get shared contexts for a group.

        Args:
            group_id: Group identifier.
            limit: Maximum number of contexts.
            min_priority: Minimum priority filter.

        Returns:
            List of shared contexts.
        """
        ...

    @abstractmethod
    async def delete_shared_context(
        self,
        shared_context_id: str,
    ) -> bool:
        """
        Delete a shared context.

        Args:
            shared_context_id: Shared context ID.

        Returns:
            True if deleted.
        """
        ...

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """
        Clean up expired shared contexts.

        Returns:
            Number of contexts cleaned up.
        """
        ...

    # Search operations

    @abstractmethod
    async def search_contexts(
        self,
        query: str,
        session_id: Optional[str] = None,
        group_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[tuple[Context, float]]:
        """
        Search contexts by semantic similarity.

        Args:
            query: Search query.
            session_id: Optional session filter.
            group_id: Optional group filter.
            limit: Maximum results.

        Returns:
            List of (context, similarity_score) tuples.
        """
        ...


class IContextSynchronizer(ABC):
    """
    Interface for synchronizing context across distributed systems.

    Used when sessions run on different machines.
    """

    @abstractmethod
    async def sync(self, group_id: str) -> None:
        """
        Synchronize context for a group.

        Args:
            group_id: Group to synchronize.
        """
        ...

    @abstractmethod
    async def push(
        self,
        shared_context: SharedContext,
    ) -> bool:
        """
        Push context to remote.

        Args:
            shared_context: Context to push.

        Returns:
            True if successful.
        """
        ...

    @abstractmethod
    async def pull(
        self,
        group_id: str,
        since: Optional[datetime] = None,
    ) -> list[SharedContext]:
        """
        Pull contexts from remote.

        Args:
            group_id: Group to pull from.
            since: Only get contexts since this time.

        Returns:
            List of shared contexts.
        """
        ...

    @abstractmethod
    async def subscribe(
        self,
        group_id: str,
        callback: Any,  # Callable[[SharedContext], Awaitable[None]]
    ) -> str:
        """
        Subscribe to context updates.

        Args:
            group_id: Group to subscribe to.
            callback: Callback for new contexts.

        Returns:
            Subscription ID.
        """
        ...

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from updates.

        Args:
            subscription_id: Subscription to cancel.

        Returns:
            True if unsubscribed.
        """
        ...
