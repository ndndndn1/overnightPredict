"""Question prediction components."""

from src.predictors.question import QuestionPredictor
from src.predictors.embeddings import EmbeddingService

__all__ = ["QuestionPredictor", "EmbeddingService"]
