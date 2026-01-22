"""API routes for OvernightPredict."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.core.models import ProjectContext, SessionStatus

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["overnight"])


# Request/Response Models
class ProjectCreateRequest(BaseModel):
    """Request to create a new project."""

    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    target_languages: list[str] = Field(
        default_factory=lambda: ["python"],
        description="Target programming languages",
    )
    architecture_type: str = Field(
        default="microservices",
        description="Architecture type",
    )
    requirements: list[str] = Field(
        default_factory=list,
        description="Project requirements",
    )
    components: list[str] = Field(
        default_factory=list,
        description="Components to implement",
    )


class ProjectResponse(BaseModel):
    """Response containing project information."""

    id: str
    name: str
    description: str
    architecture_type: str
    current_phase: str
    pending_components: list[str]
    completed_components: list[str]


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""

    topic: str | None = Field(None, description="Session topic")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Initial context",
    )


class SessionResponse(BaseModel):
    """Response containing session information."""

    id: str
    topic: str
    status: str
    questions_count: int
    answers_count: int
    artifacts_count: int
    accuracy_rate: float
    active_strategy: str


class OrchestratorStartRequest(BaseModel):
    """Request to start the orchestrator."""

    initial_sessions: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of initial sessions",
    )


class StatusResponse(BaseModel):
    """Response containing orchestrator status."""

    is_running: bool
    project: ProjectResponse | None
    sessions: dict[str, int]
    auto_scaling: bool


class MetricsResponse(BaseModel):
    """Response containing global metrics."""

    total_questions: int
    total_answers: int
    total_artifacts: int
    total_predictions: int
    accurate_predictions: int
    overall_accuracy: float
    sessions_completed: int


# Helper to get orchestrator
def get_orchestrator():
    """Get the orchestrator instance."""
    from src.api.server import get_orchestrator as _get_orchestrator
    return _get_orchestrator()


# Project endpoints
@router.post("/projects", response_model=ProjectResponse)
async def create_project(request: ProjectCreateRequest) -> ProjectResponse:
    """Create a new project for overnight coding."""
    orchestrator = get_orchestrator()

    project = ProjectContext(
        name=request.name,
        description=request.description,
        target_languages=request.target_languages,
        architecture_type=request.architecture_type,
        requirements=request.requirements,
        pending_components=request.components,
    )

    await orchestrator.initialize_project(project)

    logger.info("Project created", project_id=project.id, name=project.name)

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        architecture_type=project.architecture_type,
        current_phase=project.current_phase,
        pending_components=project.pending_components,
        completed_components=project.completed_components,
    )


@router.get("/projects/current", response_model=ProjectResponse | None)
async def get_current_project() -> ProjectResponse | None:
    """Get the current project."""
    orchestrator = get_orchestrator()
    status = await orchestrator.get_orchestrator_status()

    if status["project"]:
        p = status["project"]
        return ProjectResponse(
            id=p["id"],
            name=p["name"],
            description=p["description"],
            architecture_type=p["architecture_type"],
            current_phase=p["current_phase"],
            pending_components=p["pending_components"],
            completed_components=p["completed_components"],
        )

    return None


# Orchestrator endpoints
@router.post("/orchestrator/start")
async def start_orchestrator(request: OrchestratorStartRequest) -> dict[str, str]:
    """Start the orchestrator with parallel sessions."""
    orchestrator = get_orchestrator()

    try:
        await orchestrator.start(initial_sessions=request.initial_sessions)
        return {"status": "started", "sessions": str(request.initial_sessions)}
    except Exception as e:
        logger.error("Failed to start orchestrator", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orchestrator/stop")
async def stop_orchestrator(
    graceful: bool = Query(default=True, description="Graceful shutdown"),
) -> dict[str, str]:
    """Stop the orchestrator."""
    orchestrator = get_orchestrator()

    try:
        await orchestrator.stop(graceful=graceful)
        return {"status": "stopped"}
    except Exception as e:
        logger.error("Failed to stop orchestrator", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orchestrator/status", response_model=StatusResponse)
async def get_orchestrator_status() -> StatusResponse:
    """Get orchestrator status."""
    orchestrator = get_orchestrator()
    status = await orchestrator.get_orchestrator_status()

    project_response = None
    if status["project"]:
        p = status["project"]
        project_response = ProjectResponse(
            id=p["id"],
            name=p["name"],
            description=p["description"],
            architecture_type=p["architecture_type"],
            current_phase=p["current_phase"],
            pending_components=p["pending_components"],
            completed_components=p["completed_components"],
        )

    return StatusResponse(
        is_running=status["is_running"],
        project=project_response,
        sessions=status["sessions"],
        auto_scaling=status["auto_scaling"],
    )


@router.get("/orchestrator/metrics", response_model=MetricsResponse)
async def get_global_metrics() -> MetricsResponse:
    """Get global metrics across all sessions."""
    orchestrator = get_orchestrator()
    metrics = await orchestrator.get_global_metrics()

    return MetricsResponse(
        total_questions=metrics["total_questions"],
        total_answers=metrics["total_answers"],
        total_artifacts=metrics["total_artifacts"],
        total_predictions=metrics["total_predictions"],
        accurate_predictions=metrics["accurate_predictions"],
        overall_accuracy=metrics["overall_accuracy"],
        sessions_completed=metrics["sessions_completed"],
    )


# Session endpoints
@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions() -> list[SessionResponse]:
    """List all active sessions."""
    orchestrator = get_orchestrator()
    sessions = orchestrator.get_all_sessions()

    return [
        SessionResponse(
            id=s.id,
            topic=s.topic,
            status=s.status.value if isinstance(s.status, SessionStatus) else s.status,
            questions_count=len(s.questions),
            answers_count=len(s.answers),
            artifacts_count=len(s.artifacts),
            accuracy_rate=(
                s.accuracy_metrics.accuracy_rate if s.accuracy_metrics else 0.0
            ),
            active_strategy=str(s.active_strategy),
        )
        for s in sessions
    ]


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest) -> SessionResponse:
    """Create a new coding session."""
    orchestrator = get_orchestrator()

    try:
        session_id = await orchestrator.add_session(
            topic=request.topic,
            context=request.context,
        )

        session = await orchestrator.get_session(session_id)
        if not session:
            raise HTTPException(status_code=500, detail="Failed to create session")

        return SessionResponse(
            id=session.id,
            topic=session.topic,
            status=session.status.value if isinstance(session.status, SessionStatus) else session.status,
            questions_count=len(session.questions),
            answers_count=len(session.answers),
            artifacts_count=len(session.artifacts),
            accuracy_rate=(
                session.accuracy_metrics.accuracy_rate
                if session.accuracy_metrics
                else 0.0
            ),
            active_strategy=str(session.active_strategy),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    """Get a specific session."""
    orchestrator = get_orchestrator()
    session = await orchestrator.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        id=session.id,
        topic=session.topic,
        status=session.status.value if isinstance(session.status, SessionStatus) else session.status,
        questions_count=len(session.questions),
        answers_count=len(session.answers),
        artifacts_count=len(session.artifacts),
        accuracy_rate=(
            session.accuracy_metrics.accuracy_rate if session.accuracy_metrics else 0.0
        ),
        active_strategy=str(session.active_strategy),
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete a session."""
    orchestrator = get_orchestrator()
    removed = await orchestrator.remove_session(session_id)

    if not removed:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "deleted", "session_id": session_id}


@router.get("/sessions/{session_id}/questions")
async def get_session_questions(
    session_id: str,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Get questions from a session."""
    orchestrator = get_orchestrator()
    session = await orchestrator.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    questions = session.questions[offset : offset + limit]

    return {
        "total": len(session.questions),
        "offset": offset,
        "limit": limit,
        "questions": [q.model_dump() for q in questions],
    }


@router.get("/sessions/{session_id}/answers")
async def get_session_answers(
    session_id: str,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Get answers from a session."""
    orchestrator = get_orchestrator()
    session = await orchestrator.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    answers = session.answers[offset : offset + limit]

    return {
        "total": len(session.answers),
        "offset": offset,
        "limit": limit,
        "answers": [a.model_dump() for a in answers],
    }


@router.get("/sessions/{session_id}/artifacts")
async def get_session_artifacts(
    session_id: str,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Get code artifacts from a session."""
    orchestrator = get_orchestrator()
    session = await orchestrator.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    artifacts = session.artifacts[offset : offset + limit]

    return {
        "total": len(session.artifacts),
        "offset": offset,
        "limit": limit,
        "artifacts": [a.model_dump() for a in artifacts],
    }


@router.get("/sessions/{session_id}/predictions")
async def get_session_predictions(
    session_id: str,
) -> dict[str, Any]:
    """Get prediction results from a session."""
    orchestrator = get_orchestrator()
    session = await orchestrator.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "pending_predictions": [p.model_dump() for p in session.pending_predictions],
        "prediction_results": [r.model_dump() for r in session.prediction_results],
        "accuracy_metrics": (
            session.accuracy_metrics.model_dump() if session.accuracy_metrics else None
        ),
    }
