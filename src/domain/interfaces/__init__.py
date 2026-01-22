"""Domain interfaces (ports) - Abstract contracts for infrastructure."""

from src.domain.interfaces.llm_provider import ILLMProvider, LLMResponse
from src.domain.interfaces.prediction_strategy import IPredictionStrategy, StrategyMetadata
from src.domain.interfaces.evaluator import IEvaluator, EvaluationResult
from src.domain.interfaces.context_store import IContextStore
from src.domain.interfaces.session_repository import ISessionRepository
from src.domain.interfaces.event_bus import IEventBus, DomainEvent

__all__ = [
    "ILLMProvider",
    "LLMResponse",
    "IPredictionStrategy",
    "StrategyMetadata",
    "IEvaluator",
    "EvaluationResult",
    "IContextStore",
    "ISessionRepository",
    "IEventBus",
    "DomainEvent",
]
