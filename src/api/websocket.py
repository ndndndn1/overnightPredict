"""WebSocket support for real-time updates."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.core.models import Answer, CodeArtifact, Prediction, PredictionStrategy

logger = structlog.get_logger(__name__)

websocket_router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self._connections: dict[str, list[WebSocket]] = {}
        self._global_connections: list[WebSocket] = []

    async def connect(
        self, websocket: WebSocket, session_id: str | None = None
    ) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()

        if session_id:
            if session_id not in self._connections:
                self._connections[session_id] = []
            self._connections[session_id].append(websocket)
        else:
            self._global_connections.append(websocket)

        logger.info(
            "WebSocket connected",
            session_id=session_id,
            is_global=session_id is None,
        )

    def disconnect(self, websocket: WebSocket, session_id: str | None = None) -> None:
        """Remove a WebSocket connection."""
        if session_id and session_id in self._connections:
            self._connections[session_id] = [
                ws for ws in self._connections[session_id] if ws != websocket
            ]
        else:
            self._global_connections = [
                ws for ws in self._global_connections if ws != websocket
            ]

        logger.info(
            "WebSocket disconnected",
            session_id=session_id,
        )

    async def broadcast_to_session(
        self, session_id: str, message: dict[str, Any]
    ) -> None:
        """Broadcast a message to all connections for a session."""
        connections = self._connections.get(session_id, [])
        data = json.dumps(message, default=str)

        for connection in connections:
            try:
                await connection.send_text(data)
            except Exception as e:
                logger.error(
                    "Failed to send message to WebSocket",
                    session_id=session_id,
                    error=str(e),
                )

    async def broadcast_global(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all global connections."""
        data = json.dumps(message, default=str)

        for connection in self._global_connections:
            try:
                await connection.send_text(data)
            except Exception as e:
                logger.error("Failed to send global message", error=str(e))


# Global connection manager
manager = ConnectionManager()


@websocket_router.websocket("/session/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for session-specific updates."""
    await manager.connect(websocket, session_id)

    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                await handle_session_message(session_id, message)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)


@websocket_router.websocket("/global")
async def global_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for global updates."""
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                await handle_global_message(message)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def handle_session_message(session_id: str, message: dict[str, Any]) -> None:
    """Handle incoming WebSocket message for a session."""
    msg_type = message.get("type")

    if msg_type == "subscribe":
        # Client wants to subscribe to specific events
        logger.info(
            "Session subscription",
            session_id=session_id,
            events=message.get("events"),
        )

    elif msg_type == "ping":
        await manager.broadcast_to_session(
            session_id, {"type": "pong", "session_id": session_id}
        )


async def handle_global_message(message: dict[str, Any]) -> None:
    """Handle incoming global WebSocket message."""
    msg_type = message.get("type")

    if msg_type == "ping":
        await manager.broadcast_global({"type": "pong"})


# Event notification functions
async def notify_prediction(session_id: str, prediction: Prediction) -> None:
    """Notify about a new prediction."""
    await manager.broadcast_to_session(
        session_id,
        {
            "type": "prediction",
            "data": {
                "id": prediction.id,
                "predicted_question": prediction.predicted_question,
                "question_type": prediction.question_type.value,
                "confidence": prediction.confidence,
                "strategy": prediction.strategy_used.value,
            },
        },
    )

    await manager.broadcast_global(
        {
            "type": "prediction",
            "session_id": session_id,
            "data": {"id": prediction.id, "question": prediction.predicted_question[:100]},
        }
    )


async def notify_answer(session_id: str, answer: Answer) -> None:
    """Notify about a new answer."""
    await manager.broadcast_to_session(
        session_id,
        {
            "type": "answer",
            "data": {
                "id": answer.id,
                "question_id": answer.question_id,
                "content_preview": answer.content[:500],
                "has_code": bool(answer.code_snippets),
                "confidence": answer.confidence,
                "derived_questions": answer.derived_questions,
            },
        },
    )


async def notify_code_generated(session_id: str, artifact: CodeArtifact) -> None:
    """Notify about generated code."""
    await manager.broadcast_to_session(
        session_id,
        {
            "type": "code_generated",
            "data": {
                "id": artifact.id,
                "file_path": artifact.file_path,
                "language": artifact.language,
                "lines": len(artifact.content.split("\n")),
                "is_valid": artifact.is_valid,
                "version": artifact.version,
            },
        },
    )

    await manager.broadcast_global(
        {
            "type": "code_generated",
            "session_id": session_id,
            "data": {"file_path": artifact.file_path, "language": artifact.language},
        }
    )


async def notify_strategy_change(
    session_id: str,
    old_strategy: PredictionStrategy,
    new_strategy: PredictionStrategy,
) -> None:
    """Notify about strategy change."""
    await manager.broadcast_to_session(
        session_id,
        {
            "type": "strategy_changed",
            "data": {
                "old_strategy": old_strategy.value,
                "new_strategy": new_strategy.value,
            },
        },
    )


async def notify_session_status(session_id: str, status: str) -> None:
    """Notify about session status change."""
    await manager.broadcast_to_session(
        session_id,
        {"type": "status_changed", "data": {"status": status}},
    )

    await manager.broadcast_global(
        {
            "type": "session_status",
            "session_id": session_id,
            "data": {"status": status},
        }
    )


async def notify_metrics_update(metrics: dict[str, Any]) -> None:
    """Notify about global metrics update."""
    await manager.broadcast_global({"type": "metrics_update", "data": metrics})
