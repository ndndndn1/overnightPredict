"""Embedding service for semantic similarity computations."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """
    Service for computing text embeddings and semantic similarity.

    Uses sentence-transformers for local embedding generation,
    with fallback to simpler methods if not available.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding service."""
        self.model_name = model_name
        self._model: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the embedding model asynchronously."""
        if self._initialized:
            return

        try:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(None, self._load_model)
            self._initialized = True
            logger.info("Embedding model loaded", model=self.model_name)
        except Exception as e:
            logger.warning(
                "Failed to load embedding model, using fallback",
                error=str(e),
            )
            self._initialized = True  # Mark as initialized to use fallback

    def _load_model(self) -> Any:
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(self.model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed")
            return None

    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for a text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as numpy array
        """
        if not self._initialized:
            await self.initialize()

        if self._model is not None:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self._model.encode(text, convert_to_numpy=True)
            )
            return embedding

        # Fallback: simple bag-of-words style embedding
        return self._simple_embedding(text)

    async def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Matrix of embeddings (n_texts, embedding_dim)
        """
        if not self._initialized:
            await self.initialize()

        if self._model is not None:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: self._model.encode(texts, convert_to_numpy=True)
            )
            return embeddings

        # Fallback
        return np.array([self._simple_embedding(t) for t in texts])

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = await self.get_embedding(text1)
        emb2 = await self.get_embedding(text2)

        return self._cosine_similarity(emb1, emb2)

    async def compute_similarities(
        self, query: str, candidates: list[str]
    ) -> list[float]:
        """
        Compute similarity between query and multiple candidates.

        Args:
            query: Query text
            candidates: List of candidate texts

        Returns:
            List of similarity scores
        """
        query_emb = await self.get_embedding(query)
        candidate_embs = await self.get_embeddings(candidates)

        return [
            self._cosine_similarity(query_emb, cand_emb)
            for cand_emb in candidate_embs
        ]

    async def find_most_similar(
        self, query: str, candidates: list[str], top_k: int = 5
    ) -> list[tuple[int, float]]:
        """
        Find most similar candidates to query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of (index, similarity) tuples sorted by similarity
        """
        similarities = await self.compute_similarities(query, candidates)
        indexed = list(enumerate(similarities))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _simple_embedding(self, text: str, dim: int = 384) -> np.ndarray:
        """
        Simple fallback embedding using character-level features.

        This is a basic fallback when sentence-transformers is not available.
        """
        text = text.lower()
        words = text.split()

        embedding = np.zeros(dim)

        # Simple bag-of-words style embedding
        for i, word in enumerate(words):
            # Hash word to position
            pos = hash(word) % dim
            embedding[pos] += 1.0 / (i + 1)  # Position-weighted

            # Add character n-gram features
            for j in range(len(word) - 2):
                trigram = word[j : j + 3]
                pos = hash(trigram) % dim
                embedding[pos] += 0.5 / (i + 1)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding


@lru_cache(maxsize=1)
def get_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    """Get a cached embedding service instance."""
    return EmbeddingService(model_name)
