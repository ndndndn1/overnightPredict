"""Question prediction module using multiple strategies."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from src.core.config import Settings
from src.core.models import (
    Prediction,
    PredictionStrategy,
    Question,
    QuestionType,
)
from src.predictors.embeddings import EmbeddingService, get_embedding_service

logger = structlog.get_logger(__name__)


class QuestionPredictor:
    """
    Predicts upcoming questions based on context using multiple strategies.

    Strategies:
    - context_based: Uses current context to predict next logical questions
    - pattern_matching: Matches against known question patterns
    - semantic_similarity: Uses embeddings to find similar past scenarios
    - hybrid: Combines multiple strategies with weighted voting
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the question predictor."""
        self.settings = settings
        self._embedding_service: EmbeddingService | None = None
        self._ai_client: Any = None

        # Pattern templates for common question types
        self._question_patterns = self._load_question_patterns()

        # History for pattern learning
        self._question_history: list[Question] = []
        self._context_history: list[dict[str, Any]] = []

    @property
    def embedding_service(self) -> EmbeddingService:
        """Get the embedding service (lazy init)."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service(
                self.settings.prediction.embedding_model
            )
        return self._embedding_service

    async def predict(
        self,
        context: dict[str, Any],
        history: list[Question],
        strategy: PredictionStrategy,
        count: int = 5,
    ) -> list[Prediction]:
        """
        Predict upcoming questions.

        Args:
            context: Current context dictionary
            history: History of previous questions
            strategy: Prediction strategy to use
            count: Number of predictions to generate

        Returns:
            List of predictions
        """
        logger.info(
            "Predicting questions",
            strategy=strategy,
            count=count,
            context_keys=list(context.keys()),
        )

        # Update history for learning
        self._question_history.extend(history)
        self._context_history.append(context)

        # Select strategy method
        strategy_methods = {
            PredictionStrategy.CONTEXT_BASED: self._predict_context_based,
            PredictionStrategy.PATTERN_MATCHING: self._predict_pattern_based,
            PredictionStrategy.SEMANTIC_SIMILARITY: self._predict_semantic,
            PredictionStrategy.HYBRID: self._predict_hybrid,
        }

        method = strategy_methods.get(strategy, self._predict_context_based)
        predictions = await method(context, history, count)

        logger.info(
            "Predictions generated",
            count=len(predictions),
            strategies=[p.strategy_used for p in predictions],
        )

        return predictions

    async def _predict_context_based(
        self,
        context: dict[str, Any],
        history: list[Question],
        count: int,
    ) -> list[Prediction]:
        """
        Predict questions based on current context analysis.

        Analyzes the context to determine what questions would logically
        follow based on project state, recent activity, and goals.
        """
        predictions = []

        # Extract relevant context elements
        project = context.get("project", {})
        recent_qa = context.get("recent_qa", [])
        pending_components = project.get("pending_components", [])
        current_phase = project.get("current_phase", "implementation")

        # Generate predictions based on pending work
        for i, component in enumerate(pending_components[:count]):
            question_templates = self._get_component_questions(component, current_phase)

            if question_templates:
                template = question_templates[i % len(question_templates)]
                question_text = template.format(component=component)

                prediction = Prediction(
                    predicted_question=question_text,
                    question_type=self._infer_question_type(question_text),
                    confidence=0.8 - (i * 0.1),  # Decrease confidence for later predictions
                    strategy_used=PredictionStrategy.CONTEXT_BASED,
                    context_snapshot={"component": component, "phase": current_phase},
                )
                predictions.append(prediction)

        # Fill remaining slots with phase-specific questions
        remaining = count - len(predictions)
        if remaining > 0:
            phase_questions = self._get_phase_questions(current_phase)
            for i, q_template in enumerate(phase_questions[:remaining]):
                prediction = Prediction(
                    predicted_question=q_template,
                    question_type=self._infer_question_type(q_template),
                    confidence=0.6 - (i * 0.1),
                    strategy_used=PredictionStrategy.CONTEXT_BASED,
                    context_snapshot={"phase": current_phase},
                )
                predictions.append(prediction)

        return predictions

    async def _predict_pattern_based(
        self,
        context: dict[str, Any],
        history: list[Question],
        count: int,
    ) -> list[Prediction]:
        """
        Predict questions based on learned patterns.

        Matches current context against known question patterns
        and sequences observed in history.
        """
        predictions = []

        # Analyze history for patterns
        if len(history) >= 2:
            recent_types = [q.question_type for q in history[-5:]]
            next_patterns = self._get_next_question_patterns(recent_types)

            for i, (q_type, pattern, confidence) in enumerate(next_patterns[:count]):
                # Fill in pattern with context
                question_text = self._fill_pattern(pattern, context)

                prediction = Prediction(
                    predicted_question=question_text,
                    question_type=q_type,
                    confidence=confidence * 0.9,  # Slight reduction for uncertainty
                    strategy_used=PredictionStrategy.PATTERN_MATCHING,
                    context_snapshot={"pattern": pattern, "recent_types": recent_types},
                )
                predictions.append(prediction)

        # If not enough patterns, fall back to common patterns
        remaining = count - len(predictions)
        if remaining > 0:
            common_patterns = self._question_patterns.get("common", [])
            for i, pattern in enumerate(common_patterns[:remaining]):
                prediction = Prediction(
                    predicted_question=self._fill_pattern(pattern["template"], context),
                    question_type=QuestionType(pattern["type"]),
                    confidence=0.5,
                    strategy_used=PredictionStrategy.PATTERN_MATCHING,
                    context_snapshot={"pattern_source": "common"},
                )
                predictions.append(prediction)

        return predictions

    async def _predict_semantic(
        self,
        context: dict[str, Any],
        history: list[Question],
        count: int,
    ) -> list[Prediction]:
        """
        Predict questions using semantic similarity.

        Uses embeddings to find similar past scenarios and
        predict what questions followed them.
        """
        await self.embedding_service.initialize()
        predictions = []

        # Build context representation
        context_text = self._context_to_text(context)

        # If we have history, find similar past contexts
        if self._context_history:
            past_texts = [self._context_to_text(c) for c in self._context_history[:-1]]

            if past_texts:
                # Find most similar past contexts
                similar_indices = await self.embedding_service.find_most_similar(
                    context_text, past_texts, top_k=min(count * 2, len(past_texts))
                )

                # Get questions that followed similar contexts
                for idx, similarity in similar_indices:
                    if idx < len(self._question_history):
                        past_question = self._question_history[idx]

                        prediction = Prediction(
                            predicted_question=past_question.content,
                            question_type=past_question.question_type,
                            confidence=similarity * 0.85,
                            strategy_used=PredictionStrategy.SEMANTIC_SIMILARITY,
                            context_snapshot={
                                "similar_context_idx": idx,
                                "similarity": similarity,
                            },
                        )
                        predictions.append(prediction)

                        if len(predictions) >= count:
                            break

        # Fill remaining with context-based predictions
        if len(predictions) < count:
            context_preds = await self._predict_context_based(
                context, history, count - len(predictions)
            )
            for pred in context_preds:
                pred.strategy_used = PredictionStrategy.SEMANTIC_SIMILARITY
                pred.confidence *= 0.7  # Reduce confidence for fallback
            predictions.extend(context_preds)

        return predictions

    async def _predict_hybrid(
        self,
        context: dict[str, Any],
        history: list[Question],
        count: int,
    ) -> list[Prediction]:
        """
        Combine multiple strategies with weighted voting.

        Runs all strategies in parallel and combines results
        using confidence-weighted voting.
        """
        # Run all strategies in parallel
        results = await asyncio.gather(
            self._predict_context_based(context, history, count),
            self._predict_pattern_based(context, history, count),
            self._predict_semantic(context, history, count),
        )

        context_preds, pattern_preds, semantic_preds = results

        # Weights for each strategy
        weights = {
            PredictionStrategy.CONTEXT_BASED: 0.4,
            PredictionStrategy.PATTERN_MATCHING: 0.3,
            PredictionStrategy.SEMANTIC_SIMILARITY: 0.3,
        }

        # Combine predictions using weighted confidence
        all_predictions: dict[str, Prediction] = {}

        for preds, weight in [
            (context_preds, weights[PredictionStrategy.CONTEXT_BASED]),
            (pattern_preds, weights[PredictionStrategy.PATTERN_MATCHING]),
            (semantic_preds, weights[PredictionStrategy.SEMANTIC_SIMILARITY]),
        ]:
            for pred in preds:
                key = pred.predicted_question.lower().strip()
                weighted_confidence = pred.confidence * weight

                if key in all_predictions:
                    # Combine confidences
                    existing = all_predictions[key]
                    existing.confidence = min(
                        1.0, existing.confidence + weighted_confidence
                    )
                else:
                    pred.confidence = weighted_confidence
                    pred.strategy_used = PredictionStrategy.HYBRID
                    all_predictions[key] = pred

        # Sort by confidence and return top count
        sorted_preds = sorted(
            all_predictions.values(), key=lambda p: p.confidence, reverse=True
        )

        return sorted_preds[:count]

    def _load_question_patterns(self) -> dict[str, list[dict[str, Any]]]:
        """Load question pattern templates."""
        return {
            "common": [
                {
                    "template": "How should I structure the {component} module?",
                    "type": "architecture",
                },
                {
                    "template": "What are the best practices for implementing {feature}?",
                    "type": "implementation",
                },
                {
                    "template": "How do I test the {component} functionality?",
                    "type": "testing",
                },
                {
                    "template": "What error handling should I add for {operation}?",
                    "type": "debugging",
                },
                {
                    "template": "How can I optimize the performance of {component}?",
                    "type": "optimization",
                },
            ],
            "implementation": [
                {
                    "template": "What dependencies are needed for {component}?",
                    "type": "implementation",
                },
                {
                    "template": "How should {component} interact with {related_component}?",
                    "type": "architecture",
                },
            ],
            "testing": [
                {"template": "What edge cases should I test for {component}?", "type": "testing"},
                {"template": "How do I mock {dependency} for testing?", "type": "testing"},
            ],
            "architecture": [
                {
                    "template": "Should I use {pattern} pattern for {component}?",
                    "type": "architecture",
                },
                {"template": "How should I organize the {module} module?", "type": "architecture"},
            ],
        }

    def _get_component_questions(self, component: str, phase: str) -> list[str]:
        """Get question templates for a specific component."""
        templates = [
            f"How should I implement the {component} component?",
            f"What is the API design for {component}?",
            f"How does {component} handle errors?",
            f"What tests are needed for {component}?",
            f"How should {component} be configured?",
        ]

        if phase == "testing":
            templates = [
                f"How do I write unit tests for {component}?",
                f"What integration tests are needed for {component}?",
                f"How do I mock dependencies for {component} tests?",
            ]
        elif phase == "optimization":
            templates = [
                f"How can I improve {component} performance?",
                f"What caching strategy should {component} use?",
                f"Are there memory leaks in {component}?",
            ]

        return templates

    def _get_phase_questions(self, phase: str) -> list[str]:
        """Get general questions for a project phase."""
        phase_questions = {
            "planning": [
                "What are the main requirements for this project?",
                "What architecture should we use?",
                "What are the key components needed?",
            ],
            "implementation": [
                "What should I implement next?",
                "How should I structure this code?",
                "What patterns should I follow?",
            ],
            "testing": [
                "What tests are missing?",
                "How do I improve test coverage?",
                "What scenarios should I test?",
            ],
            "optimization": [
                "What are the performance bottlenecks?",
                "How can I reduce memory usage?",
                "What can be parallelized?",
            ],
            "documentation": [
                "What documentation is needed?",
                "How should I document the API?",
                "What examples should I provide?",
            ],
        }
        return phase_questions.get(phase, phase_questions["implementation"])

    def _get_next_question_patterns(
        self, recent_types: list[QuestionType]
    ) -> list[tuple[QuestionType, str, float]]:
        """Get likely next question patterns based on recent types."""
        # Transition probabilities based on common patterns
        transitions: dict[QuestionType, list[tuple[QuestionType, str, float]]] = {
            QuestionType.ARCHITECTURE: [
                (QuestionType.IMPLEMENTATION, "How do I implement {component}?", 0.7),
                (QuestionType.CLARIFICATION, "What are the requirements for {feature}?", 0.5),
            ],
            QuestionType.IMPLEMENTATION: [
                (QuestionType.TESTING, "How should I test this implementation?", 0.6),
                (QuestionType.DEBUGGING, "Why is {component} not working?", 0.4),
                (QuestionType.IMPLEMENTATION, "What should I implement next?", 0.3),
            ],
            QuestionType.DEBUGGING: [
                (QuestionType.IMPLEMENTATION, "How do I fix this bug?", 0.7),
                (QuestionType.TESTING, "How do I prevent this bug in the future?", 0.4),
            ],
            QuestionType.TESTING: [
                (QuestionType.IMPLEMENTATION, "The tests found issues, how do I fix them?", 0.5),
                (QuestionType.OPTIMIZATION, "How can I improve performance?", 0.4),
            ],
            QuestionType.OPTIMIZATION: [
                (QuestionType.TESTING, "How do I verify the optimization worked?", 0.6),
                (QuestionType.DOCUMENTATION, "How should I document these changes?", 0.3),
            ],
            QuestionType.CLARIFICATION: [
                (QuestionType.ARCHITECTURE, "How should I design this?", 0.6),
                (QuestionType.IMPLEMENTATION, "How do I implement this?", 0.5),
            ],
            QuestionType.DOCUMENTATION: [
                (QuestionType.IMPLEMENTATION, "What feature should I work on next?", 0.5),
                (QuestionType.TESTING, "Are there missing test cases?", 0.3),
            ],
        }

        if recent_types:
            last_type = recent_types[-1]
            return transitions.get(last_type, transitions[QuestionType.IMPLEMENTATION])

        return transitions[QuestionType.IMPLEMENTATION]

    def _fill_pattern(self, pattern: str, context: dict[str, Any]) -> str:
        """Fill in a pattern template with context values."""
        project = context.get("project", {})

        replacements = {
            "component": project.get("pending_components", ["the component"])[0]
            if project.get("pending_components")
            else "the component",
            "feature": project.get("name", "this feature"),
            "operation": "this operation",
            "dependency": "external dependency",
            "related_component": "related module",
            "pattern": "recommended",
            "module": project.get("name", "main"),
        }

        result = pattern
        for key, value in replacements.items():
            result = result.replace(f"{{{key}}}", value)

        return result

    def _context_to_text(self, context: dict[str, Any]) -> str:
        """Convert context dictionary to text for embedding."""
        parts = []

        project = context.get("project", {})
        if project:
            parts.append(f"Project: {project.get('name', 'unknown')}")
            parts.append(f"Phase: {project.get('current_phase', 'unknown')}")
            if project.get("pending_components"):
                parts.append(f"Pending: {', '.join(project['pending_components'][:5])}")

        recent_qa = context.get("recent_qa", [])
        if recent_qa:
            for qa in recent_qa[-3:]:
                parts.append(f"Q: {qa.get('question', '')[:100]}")

        return " | ".join(parts)

    def _infer_question_type(self, question: str) -> QuestionType:
        """Infer the question type from the question text."""
        question_lower = question.lower()

        # Ordered by specificity - more specific patterns first
        type_keywords = [
            (QuestionType.DEBUGGING, ["fix", "bug", "error", "issue", "problem", "not working", "broken", "failing"]),
            (QuestionType.TESTING, ["test", "verify", "validate", "coverage", "mock"]),
            (QuestionType.OPTIMIZATION, ["optimize", "improve performance", "faster", "memory", "speed up"]),
            (QuestionType.ARCHITECTURE, ["design", "structure", "organize", "architecture", "pattern"]),
            (QuestionType.DOCUMENTATION, ["document", "readme", "comment", "explain"]),
            (QuestionType.IMPLEMENTATION, ["implement", "create", "build", "add", "write", "code"]),
            (QuestionType.CLARIFICATION, ["what", "which", "should", "requirement", "need"]),
        ]

        for q_type, keywords in type_keywords:
            if any(kw in question_lower for kw in keywords):
                return q_type

        return QuestionType.IMPLEMENTATION
