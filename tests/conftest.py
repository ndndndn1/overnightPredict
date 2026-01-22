"""Pytest configuration and fixtures."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio

from src.core.config import Settings
from src.core.engine import OvernightEngine
from src.core.models import ProjectContext, SessionState
from src.sessions.orchestrator import SessionOrchestrator
from src.utils.ai_client import MockAIClient


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        environment="test",
        debug=True,
        anthropic_api_key="test-key",
        openai_api_key="test-key",
    )


@pytest.fixture
def mock_ai_client(settings: Settings) -> MockAIClient:
    """Create a mock AI client."""
    client = MockAIClient(settings)
    client.set_responses([
        """### Answer
This is a test answer.

### Code
```python
def test_function():
    pass
```

### Follow-up Questions
1. What tests should be added?
2. How should errors be handled?
""",
    ])
    return client


@pytest.fixture
def project_context() -> ProjectContext:
    """Create a test project context."""
    return ProjectContext(
        name="TestProject",
        description="A test project for unit tests",
        target_languages=["python"],
        architecture_type="monolithic",
        requirements=["Fast", "Reliable"],
        pending_components=["auth", "api", "database"],
    )


@pytest_asyncio.fixture
async def engine(settings: Settings) -> AsyncGenerator[OvernightEngine, None]:
    """Create a test engine."""
    engine = OvernightEngine(settings=settings)
    yield engine
    await engine.shutdown()


@pytest_asyncio.fixture
async def session(
    engine: OvernightEngine,
    project_context: ProjectContext,
) -> SessionState:
    """Create a test session."""
    return await engine.create_session(
        topic="Test session",
        project_context=project_context,
    )


@pytest_asyncio.fixture
async def orchestrator(settings: Settings) -> AsyncGenerator[SessionOrchestrator, None]:
    """Create a test orchestrator."""
    orchestrator = SessionOrchestrator(settings)
    yield orchestrator
    if orchestrator._is_running:
        await orchestrator.stop(graceful=False)
