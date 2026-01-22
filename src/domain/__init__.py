"""
Domain Layer - Core business logic and entities.

This layer contains:
- Entities: Core business objects with identity
- Value Objects: Immutable objects without identity
- Interfaces: Abstract contracts for infrastructure
- Events: Domain events for communication
"""

from src.domain.entities.session import Session, SessionStatus
from src.domain.entities.prediction import Prediction, PredictionStatus
from src.domain.entities.question import Question, QuestionType
from src.domain.entities.task import Task, TaskStatus, TaskPriority
from src.domain.value_objects.accuracy import AccuracyScore
from src.domain.value_objects.context import Context, ContextType
from src.domain.interfaces.llm_provider import ILLMProvider
from src.domain.interfaces.prediction_strategy import IPredictionStrategy
from src.domain.interfaces.evaluator import IEvaluator
from src.domain.interfaces.context_store import IContextStore

__all__ = [
    # Entities
    "Session",
    "SessionStatus",
    "Prediction",
    "PredictionStatus",
    "Question",
    "QuestionType",
    "Task",
    "TaskStatus",
    "TaskPriority",
    # Value Objects
    "AccuracyScore",
    "Context",
    "ContextType",
    # Interfaces
    "ILLMProvider",
    "IPredictionStrategy",
    "IEvaluator",
    "IContextStore",
]
