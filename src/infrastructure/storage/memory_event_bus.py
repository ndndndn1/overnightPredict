"""In-memory event bus implementation."""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Optional

from src.core.utils.id_generator import generate_id
from src.core.utils.logging import get_logger
from src.domain.interfaces.event_bus import DomainEvent, EventHandler, IEventBus


class InMemoryEventBus(IEventBus):
    """
    In-memory implementation of event bus.

    Provides pub/sub messaging for domain events within a process.
    """

    def __init__(self, history_limit: int = 1000):
        """Initialize the event bus.

        Args:
            history_limit: Maximum events to keep in history.
        """
        self._logger = get_logger("eventbus.memory")
        self._history_limit = history_limit

        # Handlers: event_type -> [(priority, subscription_id, handler)]
        self._handlers: dict[str, list[tuple[int, str, EventHandler]]] = defaultdict(list)
        self._global_handlers: list[tuple[int, str, EventHandler]] = []

        # Event history
        self._history: list[DomainEvent] = []

        # Waiting subscribers
        self._waiters: dict[str, list[tuple[str, Optional[Callable], asyncio.Future]]] = defaultdict(list)

    async def publish(self, event: DomainEvent) -> None:
        """Publish an event."""
        self._logger.debug(
            "Publishing event",
            event_type=event.event_type,
            aggregate_id=event.aggregate_id,
        )

        # Add to history
        self._add_to_history(event)

        # Notify waiters
        await self._notify_waiters(event)

        # Call handlers
        await self._dispatch(event)

    async def publish_batch(self, events: list[DomainEvent]) -> None:
        """Publish multiple events."""
        for event in events:
            await self.publish(event)

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        priority: int = 0,
    ) -> str:
        """Subscribe to an event type."""
        subscription_id = generate_id(prefix="sub")

        self._handlers[event_type].append((priority, subscription_id, handler))
        # Sort by priority (higher first)
        self._handlers[event_type].sort(key=lambda x: -x[0])

        self._logger.debug(
            "Subscribed to event",
            event_type=event_type,
            subscription_id=subscription_id,
        )

        return subscription_id

    def subscribe_all(
        self,
        handler: EventHandler,
        priority: int = 0,
    ) -> str:
        """Subscribe to all events."""
        subscription_id = generate_id(prefix="sub_all")

        self._global_handlers.append((priority, subscription_id, handler))
        self._global_handlers.sort(key=lambda x: -x[0])

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        # Check global handlers
        for i, (_, sub_id, _) in enumerate(self._global_handlers):
            if sub_id == subscription_id:
                self._global_handlers.pop(i)
                return True

        # Check type-specific handlers
        for event_type, handlers in self._handlers.items():
            for i, (_, sub_id, _) in enumerate(handlers):
                if sub_id == subscription_id:
                    handlers.pop(i)
                    return True

        return False

    async def wait_for(
        self,
        event_type: str,
        predicate: Optional[Callable[[DomainEvent], bool]] = None,
        timeout: float = 30.0,
    ) -> Optional[DomainEvent]:
        """Wait for a specific event."""
        future: asyncio.Future[DomainEvent] = asyncio.get_event_loop().create_future()
        waiter_id = generate_id(prefix="wait")

        self._waiters[event_type].append((waiter_id, predicate, future))

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            # Clean up waiter
            self._waiters[event_type] = [
                w for w in self._waiters[event_type] if w[0] != waiter_id
            ]

    async def get_history(
        self,
        event_type: Optional[str] = None,
        aggregate_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[DomainEvent]:
        """Get event history."""
        events = self._history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if aggregate_id:
            events = [e for e in events if e.aggregate_id == aggregate_id]

        return events[-limit:]

    def _add_to_history(self, event: DomainEvent) -> None:
        """Add event to history."""
        self._history.append(event)

        # Trim if over limit
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

    async def _notify_waiters(self, event: DomainEvent) -> None:
        """Notify waiting subscribers."""
        waiters = self._waiters.get(event.event_type, [])
        to_remove = []

        for waiter_id, predicate, future in waiters:
            if future.done():
                to_remove.append(waiter_id)
                continue

            if predicate is None or predicate(event):
                future.set_result(event)
                to_remove.append(waiter_id)

        # Clean up completed waiters
        self._waiters[event.event_type] = [
            w for w in waiters if w[0] not in to_remove
        ]

    async def _dispatch(self, event: DomainEvent) -> None:
        """Dispatch event to handlers."""
        handlers: list[EventHandler] = []

        # Global handlers
        handlers.extend(h for _, _, h in self._global_handlers)

        # Type-specific handlers
        if event.event_type in self._handlers:
            handlers.extend(h for _, _, h in self._handlers[event.event_type])

        # Call all handlers
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                self._logger.error(
                    "Handler failed",
                    event_type=event.event_type,
                    error=str(e),
                )

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()

    def get_subscription_count(self, event_type: Optional[str] = None) -> int:
        """Get number of subscriptions."""
        if event_type:
            return len(self._handlers.get(event_type, []))
        return sum(len(h) for h in self._handlers.values()) + len(self._global_handlers)
