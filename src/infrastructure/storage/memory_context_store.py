"""In-memory context store implementation."""

from datetime import datetime
from typing import Optional

from src.core.utils.id_generator import generate_id
from src.core.utils.logging import get_logger
from src.domain.interfaces.context_store import IContextStore
from src.domain.value_objects.context import Context, ContextSnapshot, SharedContext


class InMemoryContextStore(IContextStore):
    """
    In-memory implementation of context store.

    Provides fast context storage for single-process deployments.
    """

    def __init__(self):
        """Initialize the context store."""
        self._logger = get_logger("contextstore.memory")

        # Storage
        self._contexts: dict[str, Context] = {}
        self._context_by_session: dict[str, list[str]] = {}  # session_id -> [context_ids]
        self._snapshots: dict[str, ContextSnapshot] = {}
        self._snapshot_by_session: dict[str, list[str]] = {}  # session_id -> [snapshot_ids]
        self._shared_contexts: dict[str, SharedContext] = {}
        self._shared_by_group: dict[str, list[str]] = {}  # group_id -> [shared_context_ids]

    async def save_context(self, session_id: str, context: Context) -> str:
        """Save a context."""
        context_id = generate_id(prefix="ctx")

        self._contexts[context_id] = context

        if session_id not in self._context_by_session:
            self._context_by_session[session_id] = []
        self._context_by_session[session_id].append(context_id)

        return context_id

    async def get_context(self, context_id: str) -> Optional[Context]:
        """Get a context by ID."""
        return self._contexts.get(context_id)

    async def get_session_contexts(
        self, session_id: str, limit: int = 100
    ) -> list[Context]:
        """Get contexts for a session."""
        context_ids = self._context_by_session.get(session_id, [])
        contexts = []

        for ctx_id in context_ids[-limit:]:
            ctx = self._contexts.get(ctx_id)
            if ctx:
                contexts.append(ctx)

        return contexts

    async def save_snapshot(self, snapshot: ContextSnapshot) -> str:
        """Save a context snapshot."""
        snapshot_id = generate_id(prefix="snap")

        self._snapshots[snapshot_id] = snapshot

        session_id = snapshot.session_id
        if session_id not in self._snapshot_by_session:
            self._snapshot_by_session[session_id] = []
        self._snapshot_by_session[session_id].append(snapshot_id)

        return snapshot_id

    async def get_snapshot(self, snapshot_id: str) -> Optional[ContextSnapshot]:
        """Get a snapshot by ID."""
        return self._snapshots.get(snapshot_id)

    async def get_latest_snapshot(self, session_id: str) -> Optional[ContextSnapshot]:
        """Get the latest snapshot for a session."""
        snapshot_ids = self._snapshot_by_session.get(session_id, [])
        if not snapshot_ids:
            return None

        return self._snapshots.get(snapshot_ids[-1])

    async def share_context(
        self,
        context: Context,
        session_id: str,
        group_id: str,
        expires_at: Optional[datetime] = None,
        priority: int = 0,
    ) -> str:
        """Share a context with a group."""
        shared_id = generate_id(prefix="shared")

        shared = SharedContext(
            context=context,
            shared_by=session_id,
            group_id=group_id,
            expires_at=expires_at,
            priority=priority,
        )

        self._shared_contexts[shared_id] = shared

        if group_id not in self._shared_by_group:
            self._shared_by_group[group_id] = []
        self._shared_by_group[group_id].append(shared_id)

        self._logger.debug(
            "Context shared",
            shared_id=shared_id,
            group_id=group_id,
            session_id=session_id,
        )

        return shared_id

    async def get_shared_contexts(
        self, group_id: str, limit: int = 50, min_priority: int = 0
    ) -> list[SharedContext]:
        """Get shared contexts for a group."""
        shared_ids = self._shared_by_group.get(group_id, [])
        contexts = []

        for shared_id in shared_ids[-limit:]:
            shared = self._shared_contexts.get(shared_id)
            if shared and shared.priority >= min_priority:
                if not shared.is_expired():
                    contexts.append(shared)

        # Sort by priority (higher first)
        contexts.sort(key=lambda x: -x.priority)
        return contexts

    async def delete_shared_context(self, shared_context_id: str) -> bool:
        """Delete a shared context."""
        if shared_context_id not in self._shared_contexts:
            return False

        shared = self._shared_contexts.pop(shared_context_id)

        # Remove from group index
        if shared.group_id in self._shared_by_group:
            self._shared_by_group[shared.group_id] = [
                sid for sid in self._shared_by_group[shared.group_id]
                if sid != shared_context_id
            ]

        return True

    async def cleanup_expired(self) -> int:
        """Clean up expired shared contexts."""
        to_remove = []

        for shared_id, shared in self._shared_contexts.items():
            if shared.is_expired():
                to_remove.append(shared_id)

        for shared_id in to_remove:
            await self.delete_shared_context(shared_id)

        if to_remove:
            self._logger.info("Cleaned up expired contexts", count=len(to_remove))

        return len(to_remove)

    async def search_contexts(
        self,
        query: str,
        session_id: Optional[str] = None,
        group_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[tuple[Context, float]]:
        """Search contexts by keyword match (simple implementation)."""
        results: list[tuple[Context, float]] = []
        query_lower = query.lower()

        # Search session contexts
        if session_id:
            for ctx in await self.get_session_contexts(session_id, limit=1000):
                if query_lower in ctx.content.lower():
                    # Simple relevance based on occurrence count
                    score = ctx.content.lower().count(query_lower) / len(ctx.content)
                    results.append((ctx, min(score * 10, 1.0)))

        # Search shared contexts
        if group_id:
            for shared in await self.get_shared_contexts(group_id, limit=1000):
                if query_lower in shared.context.content.lower():
                    score = shared.context.content.lower().count(query_lower) / len(shared.context.content)
                    results.append((shared.context, min(score * 10, 1.0)))

        # Sort by score and return top results
        results.sort(key=lambda x: -x[1])
        return results[:limit]

    def clear(self) -> None:
        """Clear all stored contexts."""
        self._contexts.clear()
        self._context_by_session.clear()
        self._snapshots.clear()
        self._snapshot_by_session.clear()
        self._shared_contexts.clear()
        self._shared_by_group.clear()
