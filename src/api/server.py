"""FastAPI server for OvernightPredict."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.api.websocket import websocket_router
from src.core.config import Settings, get_settings
from src.sessions.orchestrator import SessionOrchestrator
from src.utils.logging import setup_logging

logger = structlog.get_logger(__name__)

# Global orchestrator instance
_orchestrator: SessionOrchestrator | None = None


def get_orchestrator() -> SessionOrchestrator:
    """Get the global orchestrator instance."""
    if _orchestrator is None:
        raise RuntimeError("Orchestrator not initialized")
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan."""
    global _orchestrator

    settings = get_settings()
    setup_logging(settings)

    logger.info("Starting OvernightPredict API server")

    # Initialize orchestrator
    _orchestrator = SessionOrchestrator(settings)

    yield

    # Cleanup
    if _orchestrator:
        await _orchestrator.stop(graceful=True)

    logger.info("OvernightPredict API server stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        settings: Optional settings override

    Returns:
        Configured FastAPI application
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="OvernightPredict API",
        description="AI-powered autonomous code generation with question prediction",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS
    if settings.api.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include routers
    app.include_router(router, prefix="/api/v1")
    app.include_router(websocket_router, prefix="/ws")

    # Error handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # Health check
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "healthy"}

    return app


async def run_server(
    settings: Settings | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """
    Run the API server.

    Args:
        settings: Optional settings override
        host: Optional host override
        port: Optional port override
    """
    if settings is None:
        settings = get_settings()

    host = host or settings.api.host
    port = port or settings.api.port

    app = create_app(settings)

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        workers=settings.api.workers,
        log_level="info",
    )

    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(run_server())
