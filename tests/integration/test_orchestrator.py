"""Integration tests for the Orchestrator."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.core.config.settings import Settings, LLMProvider
from src.application.services.orchestrator import Orchestrator
from src.infrastructure.storage.sqlite_repository import SQLiteSessionRepository
from src.infrastructure.storage.memory_event_bus import InMemoryEventBus
from src.infrastructure.storage.memory_context_store import InMemoryContextStore


@pytest.fixture
def repository(tmp_path):
    """Create a test repository."""
    import asyncio
    db_path = tmp_path / "test.db"
    repo = SQLiteSessionRepository(db_path)
    asyncio.get_event_loop().run_until_complete(repo.initialize())
    return repo


@pytest.fixture
def event_bus():
    """Create a test event bus."""
    return InMemoryEventBus()


@pytest.fixture
def context_store():
    """Create a test context store."""
    return InMemoryContextStore()


@pytest.fixture
def mock_settings(tmp_path):
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.work_dir = tmp_path
    settings.prediction = MagicMock()
    settings.prediction.accuracy_threshold = 0.7
    settings.prediction.lookahead_count = 5
    settings.prediction.min_accuracy_for_keep = 0.6
    settings.prediction.evaluation_window = 10
    settings.prediction.similarity_model = "all-MiniLM-L6-v2"
    settings.orchestrator = MagicMock()
    settings.orchestrator.max_sessions = 10
    settings.orchestrator.session_timeout = 3600
    settings.orchestrator.health_check_interval = 30
    settings.get_enabled_providers = MagicMock(return_value=[])
    return settings


class TestOrchestrator:
    """Tests for Orchestrator service."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(
        self,
        mock_settings,
        repository,
        event_bus,
        context_store,
    ):
        """Test orchestrator initialization."""
        orchestrator = Orchestrator(
            settings=mock_settings,
            repository=repository,
            context_store=context_store,
            event_bus=event_bus,
        )

        await orchestrator.initialize()
        assert orchestrator.session_count == 0

    @pytest.mark.asyncio
    async def test_get_all_sessions_empty(
        self,
        mock_settings,
        repository,
        event_bus,
        context_store,
    ):
        """Test getting all sessions when empty."""
        orchestrator = Orchestrator(
            settings=mock_settings,
            repository=repository,
            context_store=context_store,
            event_bus=event_bus,
        )

        await orchestrator.initialize()
        statuses = orchestrator.get_all_sessions_status()

        assert statuses == []


class TestEventBus:
    """Tests for InMemoryEventBus."""

    @pytest.mark.asyncio
    async def test_publish_subscribe(self, event_bus):
        """Test event publishing and subscription."""
        from src.domain.interfaces.event_bus import DomainEvent

        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe("test.event", handler)

        event = DomainEvent(event_type="test.event", aggregate_id="123")
        await event_bus.publish(event)

        assert len(received_events) == 1
        assert received_events[0].aggregate_id == "123"

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        from src.domain.interfaces.event_bus import DomainEvent

        received = []

        async def handler(event):
            received.append(event)

        sub_id = event_bus.subscribe("test.event", handler)
        event_bus.unsubscribe(sub_id)

        await event_bus.publish(DomainEvent(event_type="test.event"))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_wait_for_event(self, event_bus):
        """Test waiting for a specific event."""
        from src.domain.interfaces.event_bus import DomainEvent

        async def delayed_publish():
            await asyncio.sleep(0.1)
            await event_bus.publish(DomainEvent(event_type="test.event", aggregate_id="waited"))

        asyncio.create_task(delayed_publish())

        event = await event_bus.wait_for("test.event", timeout=1.0)

        assert event is not None
        assert event.aggregate_id == "waited"


class TestContextStore:
    """Tests for InMemoryContextStore."""

    @pytest.mark.asyncio
    async def test_save_and_get_context(self, context_store):
        """Test saving and retrieving context."""
        from src.domain.value_objects.context import Context, ContextType

        ctx = Context(
            content="Test content",
            context_type=ContextType.CODE,
            source="test",
        )

        ctx_id = await context_store.save_context("session1", ctx)
        retrieved = await context_store.get_context(ctx_id)

        assert retrieved is not None
        assert retrieved.content == "Test content"

    @pytest.mark.asyncio
    async def test_share_context(self, context_store):
        """Test sharing context across groups."""
        from src.domain.value_objects.context import Context, ContextType

        ctx = Context(
            content="Shared content",
            context_type=ContextType.CODE,
        )

        await context_store.share_context(
            context=ctx,
            session_id="session1",
            group_id="group1",
            priority=5,
        )

        shared = await context_store.get_shared_contexts("group1")

        assert len(shared) == 1
        assert shared[0].context.content == "Shared content"
        assert shared[0].priority == 5
