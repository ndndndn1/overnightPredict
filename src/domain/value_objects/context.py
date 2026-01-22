"""Context value objects."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ContextType(str, Enum):
    """Type of context."""

    CODE = "code"
    CONVERSATION = "conversation"
    FILE_SYSTEM = "file_system"
    ERROR = "error"
    REQUIREMENT = "requirement"
    ANALYSIS = "analysis"
    COMPOSITE = "composite"


@dataclass(frozen=True)
class Context:
    """
    Immutable context value object.

    Represents the context for prediction or execution.
    """

    content: str
    context_type: ContextType
    source: str = ""  # File path, session ID, etc.
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Embeddings for semantic operations
    embedding: Optional[tuple[float, ...]] = None

    def __str__(self) -> str:
        """String representation."""
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"Context({self.context_type.value}): {preview}"

    def truncate(self, max_length: int) -> "Context":
        """Return a truncated copy of context."""
        if len(self.content) <= max_length:
            return self
        truncated = self.content[:max_length] + "...[truncated]"
        return Context(
            content=truncated,
            context_type=self.context_type,
            source=self.source,
            timestamp=self.timestamp,
        )


@dataclass
class ContextSnapshot:
    """
    Mutable snapshot of multiple contexts.

    Used to capture the full state at a point in time.
    """

    contexts: list[Context] = field(default_factory=list)
    snapshot_at: datetime = field(default_factory=datetime.utcnow)
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, context: Context) -> None:
        """Add a context to the snapshot."""
        self.contexts.append(context)

    def get_by_type(self, context_type: ContextType) -> list[Context]:
        """Get contexts of a specific type."""
        return [c for c in self.contexts if c.context_type == context_type]

    def merge(self, other: "ContextSnapshot") -> "ContextSnapshot":
        """Merge with another snapshot."""
        merged = ContextSnapshot(
            contexts=self.contexts + other.contexts,
            snapshot_at=datetime.utcnow(),
            session_id=self.session_id or other.session_id,
        )
        merged.metadata = {**self.metadata, **other.metadata}
        return merged

    @property
    def total_length(self) -> int:
        """Get total content length."""
        return sum(len(c.content) for c in self.contexts)

    def to_string(self, separator: str = "\n---\n") -> str:
        """Convert all contexts to a single string."""
        return separator.join(c.content for c in self.contexts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "snapshot_at": self.snapshot_at.isoformat(),
            "context_count": len(self.contexts),
            "total_length": self.total_length,
            "types": [c.context_type.value for c in self.contexts],
        }


@dataclass(frozen=True)
class SharedContext:
    """
    Context that can be shared across sessions.

    Used for inter-session communication and coordination.
    """

    context: Context
    shared_by: str  # Session ID that shared this
    shared_at: datetime = field(default_factory=datetime.utcnow)
    group_id: str = ""
    expires_at: Optional[datetime] = None
    priority: int = 0

    def is_expired(self) -> bool:
        """Check if context has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.context.content,
            "context_type": self.context.context_type.value,
            "shared_by": self.shared_by,
            "shared_at": self.shared_at.isoformat(),
            "group_id": self.group_id,
            "priority": self.priority,
        }
