"""Event bus interface for domain events."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from src.core.utils.id_generator import generate_id


@dataclass
class DomainEvent:
    """Base class for domain events."""

    id: str = field(default_factory=lambda: generate_id(prefix="evt"))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""
    aggregate_id: str = ""  # Session ID, etc.
    aggregate_type: str = ""  # "session", "prediction", etc.
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "payload": self.payload,
            "metadata": self.metadata,
        }


# Specific event types

@dataclass
class SessionStartedEvent(DomainEvent):
    """Event when a session starts."""

    def __post_init__(self) -> None:
        self.event_type = "session.started"
        self.aggregate_type = "session"


@dataclass
class SessionCompletedEvent(DomainEvent):
    """Event when a session completes."""

    def __post_init__(self) -> None:
        self.event_type = "session.completed"
        self.aggregate_type = "session"


@dataclass
class SessionFailedEvent(DomainEvent):
    """Event when a session fails."""

    def __post_init__(self) -> None:
        self.event_type = "session.failed"
        self.aggregate_type = "session"


@dataclass
class PredictionMadeEvent(DomainEvent):
    """Event when a prediction is made."""

    def __post_init__(self) -> None:
        self.event_type = "prediction.made"
        self.aggregate_type = "prediction"


@dataclass
class PredictionEvaluatedEvent(DomainEvent):
    """Event when a prediction is evaluated."""

    def __post_init__(self) -> None:
        self.event_type = "prediction.evaluated"
        self.aggregate_type = "prediction"


@dataclass
class StrategyChangedEvent(DomainEvent):
    """Event when prediction strategy is changed."""

    def __post_init__(self) -> None:
        self.event_type = "strategy.changed"
        self.aggregate_type = "session"


@dataclass
class TaskCompletedEvent(DomainEvent):
    """Event when a task is completed."""

    def __post_init__(self) -> None:
        self.event_type = "task.completed"
        self.aggregate_type = "task"


@dataclass
class RateLimitHitEvent(DomainEvent):
    """Event when rate limit is hit."""

    def __post_init__(self) -> None:
        self.event_type = "rate_limit.hit"
        self.aggregate_type = "session"


@dataclass
class ContextSharedEvent(DomainEvent):
    """Event when context is shared."""

    def __post_init__(self) -> None:
        self.event_type = "context.shared"
        self.aggregate_type = "context"


# Event handler type
EventHandler = Callable[[DomainEvent], Awaitable[None]]


class IEventBus(ABC):
    """
    Interface for the event bus.

    Enables loose coupling through pub/sub messaging.
    """

    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """
        Publish an event.

        Args:
            event: Event to publish.
        """
        ...

    @abstractmethod
    async def publish_batch(self, events: list[DomainEvent]) -> None:
        """
        Publish multiple events.

        Args:
            events: Events to publish.
        """
        ...

    @abstractmethod
    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        priority: int = 0,
    ) -> str:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to (e.g., "session.started").
            handler: Async handler function.
            priority: Handler priority (higher = called first).

        Returns:
            Subscription ID.
        """
        ...

    @abstractmethod
    def subscribe_all(
        self,
        handler: EventHandler,
        priority: int = 0,
    ) -> str:
        """
        Subscribe to all events.

        Args:
            handler: Async handler function.
            priority: Handler priority.

        Returns:
            Subscription ID.
        """
        ...

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: Subscription to cancel.

        Returns:
            True if unsubscribed.
        """
        ...

    @abstractmethod
    async def wait_for(
        self,
        event_type: str,
        predicate: Optional[Callable[[DomainEvent], bool]] = None,
        timeout: float = 30.0,
    ) -> Optional[DomainEvent]:
        """
        Wait for a specific event.

        Args:
            event_type: Event type to wait for.
            predicate: Optional filter function.
            timeout: Maximum wait time in seconds.

        Returns:
            The event, or None if timeout.
        """
        ...

    @abstractmethod
    async def get_history(
        self,
        event_type: Optional[str] = None,
        aggregate_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[DomainEvent]:
        """
        Get event history.

        Args:
            event_type: Optional event type filter.
            aggregate_id: Optional aggregate filter.
            limit: Maximum events to return.

        Returns:
            List of events.
        """
        ...
