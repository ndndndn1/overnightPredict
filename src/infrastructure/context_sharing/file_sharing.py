"""File-based context sharing implementation."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import aiofiles

from src.core.exceptions import ContextSharingError
from src.core.utils.id_generator import generate_id
from src.core.utils.logging import get_logger
from src.domain.interfaces.context_store import IContextSynchronizer
from src.domain.value_objects.context import Context, ContextType, SharedContext


class FileContextSharing(IContextSynchronizer):
    """
    File-based context sharing.

    Shares context through a shared file system directory.
    Suitable for sessions running on the same machine or shared NFS.
    """

    def __init__(self, shared_path: Path):
        """Initialize file-based sharing.

        Args:
            shared_path: Path to shared directory.
        """
        self._shared_path = Path(shared_path)
        self._logger = get_logger("sharing.file", path=str(shared_path))
        self._subscriptions: dict[str, tuple[str, Callable[[SharedContext], Awaitable[None]]]] = {}
        self._watch_tasks: dict[str, asyncio.Task] = {}

    async def initialize(self) -> None:
        """Initialize the shared directory structure."""
        self._shared_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organization
        (self._shared_path / "contexts").mkdir(exist_ok=True)
        (self._shared_path / "groups").mkdir(exist_ok=True)
        (self._shared_path / "metadata").mkdir(exist_ok=True)

        self._logger.info("File sharing initialized")

    def _get_group_path(self, group_id: str) -> Path:
        """Get path for group directory."""
        return self._shared_path / "groups" / group_id

    def _get_context_file(self, group_id: str, context_id: str) -> Path:
        """Get path for context file."""
        return self._get_group_path(group_id) / f"{context_id}.json"

    async def sync(self, group_id: str) -> None:
        """Synchronize context for a group."""
        group_path = self._get_group_path(group_id)
        group_path.mkdir(parents=True, exist_ok=True)
        self._logger.debug("Synced group", group_id=group_id)

    async def push(self, shared_context: SharedContext) -> bool:
        """Push context to file system."""
        try:
            group_path = self._get_group_path(shared_context.group_id)
            group_path.mkdir(parents=True, exist_ok=True)

            context_id = generate_id(prefix="ctx")
            file_path = self._get_context_file(shared_context.group_id, context_id)

            data = {
                "id": context_id,
                "content": shared_context.context.content,
                "context_type": shared_context.context.context_type.value,
                "source": shared_context.context.source,
                "shared_by": shared_context.shared_by,
                "shared_at": shared_context.shared_at.isoformat(),
                "group_id": shared_context.group_id,
                "priority": shared_context.priority,
                "expires_at": shared_context.expires_at.isoformat() if shared_context.expires_at else None,
            }

            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(data, indent=2))

            self._logger.debug(
                "Pushed context",
                context_id=context_id,
                group_id=shared_context.group_id,
            )
            return True

        except Exception as e:
            self._logger.error("Failed to push context", error=str(e))
            raise ContextSharingError(
                f"Failed to push context: {str(e)}",
                sharing_type="file",
                cause=e,
            )

    async def pull(
        self,
        group_id: str,
        since: Optional[datetime] = None,
    ) -> list[SharedContext]:
        """Pull contexts from file system."""
        group_path = self._get_group_path(group_id)

        if not group_path.exists():
            return []

        contexts = []

        for file_path in group_path.glob("*.json"):
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    data = json.loads(content)

                shared_at = datetime.fromisoformat(data["shared_at"])

                # Filter by time if specified
                if since and shared_at < since:
                    continue

                # Check expiration
                expires_at = None
                if data.get("expires_at"):
                    expires_at = datetime.fromisoformat(data["expires_at"])
                    if datetime.utcnow() > expires_at:
                        # Remove expired file
                        file_path.unlink(missing_ok=True)
                        continue

                context = Context(
                    content=data["content"],
                    context_type=ContextType(data["context_type"]),
                    source=data.get("source", ""),
                    timestamp=shared_at,
                )

                shared = SharedContext(
                    context=context,
                    shared_by=data["shared_by"],
                    shared_at=shared_at,
                    group_id=data["group_id"],
                    priority=data.get("priority", 0),
                    expires_at=expires_at,
                )

                contexts.append(shared)

            except Exception as e:
                self._logger.warning(f"Failed to read context file {file_path}", error=str(e))

        # Sort by shared_at
        contexts.sort(key=lambda x: x.shared_at)
        return contexts

    async def subscribe(
        self,
        group_id: str,
        callback: Callable[[SharedContext], Awaitable[None]],
    ) -> str:
        """Subscribe to context updates via file watching."""
        subscription_id = generate_id(prefix="sub")
        self._subscriptions[subscription_id] = (group_id, callback)

        # Start watching if not already
        if group_id not in self._watch_tasks:
            self._watch_tasks[group_id] = asyncio.create_task(
                self._watch_group(group_id)
            )

        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from updates."""
        if subscription_id not in self._subscriptions:
            return False

        group_id, _ = self._subscriptions.pop(subscription_id)

        # Stop watching if no more subscriptions for group
        remaining = [gid for gid, _ in self._subscriptions.values() if gid == group_id]
        if not remaining and group_id in self._watch_tasks:
            self._watch_tasks[group_id].cancel()
            del self._watch_tasks[group_id]

        return True

    async def _watch_group(self, group_id: str) -> None:
        """Watch group directory for new contexts."""
        group_path = self._get_group_path(group_id)
        group_path.mkdir(parents=True, exist_ok=True)

        seen_files: set[str] = set()

        # Initial scan
        for file_path in group_path.glob("*.json"):
            seen_files.add(str(file_path))

        while True:
            try:
                await asyncio.sleep(1.0)  # Poll interval

                current_files = set(str(p) for p in group_path.glob("*.json"))
                new_files = current_files - seen_files

                for file_path_str in new_files:
                    file_path = Path(file_path_str)
                    try:
                        async with aiofiles.open(file_path, "r") as f:
                            content = await f.read()
                            data = json.loads(content)

                        context = Context(
                            content=data["content"],
                            context_type=ContextType(data["context_type"]),
                            source=data.get("source", ""),
                        )

                        shared = SharedContext(
                            context=context,
                            shared_by=data["shared_by"],
                            shared_at=datetime.fromisoformat(data["shared_at"]),
                            group_id=data["group_id"],
                            priority=data.get("priority", 0),
                        )

                        # Notify subscribers
                        for sub_group_id, callback in self._subscriptions.values():
                            if sub_group_id == group_id:
                                await callback(shared)

                    except Exception as e:
                        self._logger.warning(f"Failed to process new file", error=str(e))

                seen_files = current_files

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Watch error", error=str(e))
                await asyncio.sleep(5.0)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Cancel all watch tasks
        for task in self._watch_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._watch_tasks.clear()
        self._subscriptions.clear()
