"""Checkpoint management for session persistence and recovery."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import structlog

from src.core.models import SessionState

logger = structlog.get_logger(__name__)


class CheckpointManager:
    """
    Manages session checkpoints for persistence and recovery.

    Features:
    - Automatic checkpoint saving
    - Session recovery from checkpoints
    - Checkpoint rotation and cleanup
    """

    def __init__(
        self,
        checkpoint_dir: str = "./data/checkpoints",
        max_checkpoints_per_session: int = 10,
    ) -> None:
        """Initialize the checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints_per_session = max_checkpoints_per_session

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track pending saves
        self._pending_saves: dict[str, asyncio.Task[None]] = {}

    async def save_checkpoint(
        self,
        session: SessionState,
        sync: bool = False,
    ) -> str:
        """
        Save a checkpoint for a session.

        Args:
            session: Session state to checkpoint
            sync: If True, wait for save to complete

        Returns:
            Path to checkpoint file
        """
        checkpoint_data = self._serialize_session(session)
        checkpoint_path = self._get_checkpoint_path(session.id, session.checkpoint_version)

        if sync:
            await self._write_checkpoint(checkpoint_path, checkpoint_data)
        else:
            # Save asynchronously
            task = asyncio.create_task(
                self._write_checkpoint(checkpoint_path, checkpoint_data)
            )
            self._pending_saves[session.id] = task

        # Cleanup old checkpoints
        await self._cleanup_old_checkpoints(session.id)

        logger.debug(
            "Checkpoint saved",
            session_id=session.id,
            version=session.checkpoint_version,
            path=str(checkpoint_path),
        )

        return str(checkpoint_path)

    async def load_checkpoint(
        self,
        session_id: str,
        version: int | None = None,
    ) -> SessionState | None:
        """
        Load a session from checkpoint.

        Args:
            session_id: ID of session to load
            version: Specific version to load, or latest if None

        Returns:
            Session state or None if not found
        """
        if version is not None:
            checkpoint_path = self._get_checkpoint_path(session_id, version)
            if checkpoint_path.exists():
                return await self._load_from_path(checkpoint_path)
            return None

        # Find latest checkpoint
        latest = await self._find_latest_checkpoint(session_id)
        if latest:
            return await self._load_from_path(latest)

        return None

    async def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        """List all checkpoints for a session."""
        session_dir = self.checkpoint_dir / session_id
        if not session_dir.exists():
            return []

        checkpoints = []
        for checkpoint_file in sorted(session_dir.glob("*.json")):
            try:
                stat = checkpoint_file.stat()
                checkpoints.append({
                    "path": str(checkpoint_file),
                    "version": int(checkpoint_file.stem.split("_v")[-1]),
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "size_bytes": stat.st_size,
                })
            except (ValueError, OSError):
                continue

        return sorted(checkpoints, key=lambda c: c["version"], reverse=True)

    async def delete_checkpoint(self, session_id: str, version: int) -> bool:
        """Delete a specific checkpoint."""
        checkpoint_path = self._get_checkpoint_path(session_id, version)
        try:
            checkpoint_path.unlink(missing_ok=True)
            logger.info(
                "Checkpoint deleted",
                session_id=session_id,
                version=version,
            )
            return True
        except OSError as e:
            logger.error(
                "Failed to delete checkpoint",
                session_id=session_id,
                version=version,
                error=str(e),
            )
            return False

    async def delete_all_checkpoints(self, session_id: str) -> int:
        """Delete all checkpoints for a session."""
        session_dir = self.checkpoint_dir / session_id
        if not session_dir.exists():
            return 0

        count = 0
        for checkpoint_file in session_dir.glob("*.json"):
            try:
                checkpoint_file.unlink()
                count += 1
            except OSError:
                continue

        # Remove directory if empty
        try:
            session_dir.rmdir()
        except OSError:
            pass

        logger.info(
            "All checkpoints deleted",
            session_id=session_id,
            count=count,
        )
        return count

    def _get_checkpoint_path(self, session_id: str, version: int) -> Path:
        """Get path for a specific checkpoint."""
        session_dir = self.checkpoint_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / f"checkpoint_v{version:06d}.json"

    async def _find_latest_checkpoint(self, session_id: str) -> Path | None:
        """Find the latest checkpoint for a session."""
        session_dir = self.checkpoint_dir / session_id
        if not session_dir.exists():
            return None

        checkpoints = list(session_dir.glob("*.json"))
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    async def _write_checkpoint(self, path: Path, data: dict[str, Any]) -> None:
        """Write checkpoint data to file."""
        try:
            async with aiofiles.open(path, "w") as f:
                await f.write(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(
                "Failed to write checkpoint",
                path=str(path),
                error=str(e),
            )
            raise

    async def _load_from_path(self, path: Path) -> SessionState | None:
        """Load session state from checkpoint file."""
        try:
            async with aiofiles.open(path, "r") as f:
                content = await f.read()
                data = json.loads(content)
                return self._deserialize_session(data)
        except Exception as e:
            logger.error(
                "Failed to load checkpoint",
                path=str(path),
                error=str(e),
            )
            return None

    async def _cleanup_old_checkpoints(self, session_id: str) -> None:
        """Remove old checkpoints beyond the limit."""
        checkpoints = await self.list_checkpoints(session_id)

        if len(checkpoints) > self.max_checkpoints_per_session:
            to_delete = checkpoints[self.max_checkpoints_per_session:]
            for checkpoint in to_delete:
                await self.delete_checkpoint(session_id, checkpoint["version"])

    def _serialize_session(self, session: SessionState) -> dict[str, Any]:
        """Serialize session state to dictionary."""
        return session.model_dump(mode="json")

    def _deserialize_session(self, data: dict[str, Any]) -> SessionState:
        """Deserialize session state from dictionary."""
        return SessionState.model_validate(data)

    async def wait_for_pending_saves(self) -> None:
        """Wait for all pending checkpoint saves to complete."""
        if self._pending_saves:
            await asyncio.gather(*self._pending_saves.values(), return_exceptions=True)
            self._pending_saves.clear()
