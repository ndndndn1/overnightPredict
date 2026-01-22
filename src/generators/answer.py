"""Answer generation module using AI models."""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from src.core.config import Settings
from src.core.models import Answer, Question, QuestionType
from src.utils.ai_client import AIClient, get_ai_client

logger = structlog.get_logger(__name__)


class AnswerGenerator:
    """
    Generates comprehensive answers to questions using AI models.

    Features:
    - Context-aware answer generation
    - Code snippet extraction
    - Follow-up question derivation
    - Confidence scoring
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the answer generator."""
        self.settings = settings
        self._ai_client: AIClient | None = None

    @property
    def ai_client(self) -> AIClient:
        """Get the AI client (lazy init)."""
        if self._ai_client is None:
            self._ai_client = get_ai_client(self.settings)
        return self._ai_client

    async def generate(
        self,
        question: Question,
        context: dict[str, Any],
        history: list[Answer] | None = None,
    ) -> Answer:
        """
        Generate a comprehensive answer to a question.

        Args:
            question: The question to answer
            context: Current context dictionary
            history: Optional previous answers for context

        Returns:
            Generated answer with code snippets and derived questions
        """
        logger.info(
            "Generating answer",
            question_id=question.id,
            question_type=question.question_type,
        )

        # Build the prompt
        prompt = self._build_answer_prompt(question, context, history)

        # Generate response
        response = await self.ai_client.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(question.question_type),
            max_tokens=self.settings.ai.anthropic_max_tokens,
        )

        # Parse the response
        answer = self._parse_response(question.id, response)

        logger.info(
            "Answer generated",
            question_id=question.id,
            has_code=bool(answer.code_snippets),
            derived_questions=len(answer.derived_questions),
        )

        return answer

    async def generate_preview(
        self,
        predicted_question: str,
        question_type: QuestionType,
        context: dict[str, Any],
    ) -> str:
        """
        Generate a preview answer for a predicted question.

        This is a lighter-weight generation for predictions.
        """
        prompt = f"""Context:
{json.dumps(context, indent=2, default=str)[:2000]}

Predicted Question ({question_type.value}):
{predicted_question}

Provide a concise but helpful answer that addresses this question.
Focus on practical implementation guidance.
Include code examples if relevant.
"""

        response = await self.ai_client.generate(
            prompt=prompt,
            system_prompt="You are an expert software engineer. Provide clear, actionable answers.",
            max_tokens=2000,
        )

        return response.strip()

    def _build_answer_prompt(
        self,
        question: Question,
        context: dict[str, Any],
        history: list[Answer] | None,
    ) -> str:
        """Build the prompt for answer generation."""
        parts = []

        # Add context
        parts.append("## Current Context")
        parts.append(self._format_context(context))
        parts.append("")

        # Add relevant history
        if history:
            parts.append("## Recent Q&A History")
            for answer in history[-3:]:
                parts.append(f"- Q: {answer.content[:200]}...")
            parts.append("")

        # Add the question
        parts.append("## Question")
        parts.append(f"Type: {question.question_type.value}")
        parts.append(f"Content: {question.content}")
        parts.append("")

        # Add any question-specific context
        if question.context:
            parts.append("## Question Context")
            parts.append(json.dumps(question.context, indent=2, default=str)[:1000])
            parts.append("")

        # Add instructions
        parts.append("## Instructions")
        parts.append(self._get_answer_instructions(question.question_type))

        return "\n".join(parts)

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context for the prompt."""
        formatted_parts = []

        project = context.get("project", {})
        if project:
            formatted_parts.append(f"Project: {project.get('name', 'Unknown')}")
            formatted_parts.append(f"Phase: {project.get('current_phase', 'implementation')}")

            if project.get("pending_components"):
                formatted_parts.append(
                    f"Pending Components: {', '.join(project['pending_components'][:5])}"
                )

            if project.get("completed_components"):
                formatted_parts.append(
                    f"Completed: {', '.join(project['completed_components'][:5])}"
                )

            if project.get("architecture_type"):
                formatted_parts.append(f"Architecture: {project['architecture_type']}")

        recent_qa = context.get("recent_qa", [])
        if recent_qa:
            formatted_parts.append("\nRecent Activity:")
            for qa in recent_qa[-2:]:
                formatted_parts.append(f"  - {qa.get('question', '')[:100]}")

        return "\n".join(formatted_parts) if formatted_parts else "No specific context"

    def _get_system_prompt(self, question_type: QuestionType) -> str:
        """Get the system prompt based on question type."""
        base_prompt = """You are an expert software engineer with deep knowledge across
multiple programming languages and frameworks. You provide clear, practical, and
production-ready solutions.

When answering:
1. Be specific and actionable
2. Include working code examples when relevant
3. Consider edge cases and error handling
4. Follow best practices and design patterns
5. Suggest follow-up improvements when appropriate

"""

        type_specific = {
            QuestionType.IMPLEMENTATION: """Focus on clean, maintainable code.
Provide complete implementations with proper error handling.
Include type hints and documentation.""",
            QuestionType.ARCHITECTURE: """Focus on scalability, maintainability, and best practices.
Consider trade-offs between different approaches.
Recommend appropriate design patterns.""",
            QuestionType.TESTING: """Focus on comprehensive test coverage.
Include unit tests, integration tests, and edge cases.
Suggest mocking strategies where appropriate.""",
            QuestionType.DEBUGGING: """Focus on identifying root causes.
Provide step-by-step debugging approaches.
Suggest preventive measures.""",
            QuestionType.OPTIMIZATION: """Focus on measurable improvements.
Consider time and space complexity.
Provide before/after comparisons when possible.""",
            QuestionType.DOCUMENTATION: """Focus on clarity and completeness.
Include examples and usage patterns.
Follow documentation best practices.""",
            QuestionType.CLARIFICATION: """Focus on understanding requirements.
Ask clarifying questions if needed.
Provide options when multiple approaches are valid.""",
        }

        return base_prompt + type_specific.get(question_type, "")

    def _get_answer_instructions(self, question_type: QuestionType) -> str:
        """Get specific instructions based on question type."""
        instructions = {
            QuestionType.IMPLEMENTATION: """Provide:
1. A clear explanation of the approach
2. Complete, working code implementation
3. Any necessary imports or dependencies
4. Usage examples
5. 2-3 follow-up questions that might arise""",
            QuestionType.ARCHITECTURE: """Provide:
1. Recommended architecture/design
2. Rationale for the choice
3. Component diagram or structure description
4. Potential trade-offs
5. 2-3 follow-up questions about implementation""",
            QuestionType.TESTING: """Provide:
1. Testing strategy overview
2. Sample test code
3. Test data suggestions
4. Coverage considerations
5. 2-3 follow-up questions about edge cases""",
            QuestionType.DEBUGGING: """Provide:
1. Likely root causes
2. Step-by-step debugging approach
3. Fix implementation
4. Prevention strategies
5. 2-3 follow-up questions about related issues""",
            QuestionType.OPTIMIZATION: """Provide:
1. Performance analysis
2. Optimization approach
3. Optimized implementation
4. Expected improvements
5. 2-3 follow-up questions about further optimization""",
            QuestionType.DOCUMENTATION: """Provide:
1. Documentation structure
2. Sample documentation content
3. Examples and usage
4. API reference format
5. 2-3 follow-up questions about documentation coverage""",
            QuestionType.CLARIFICATION: """Provide:
1. Interpretation of the requirement
2. Clarifying questions if needed
3. Options if multiple approaches exist
4. Recommended approach
5. 2-3 follow-up questions to refine requirements""",
        }

        base_instruction = """
Format your response as:
### Answer
[Your main answer here]

### Code (if applicable)
```language
[Code here]
```

### Follow-up Questions
1. [Question 1]
2. [Question 2]
3. [Question 3]
"""

        return instructions.get(question_type, instructions[QuestionType.IMPLEMENTATION]) + base_instruction

    def _parse_response(self, question_id: str, response: str) -> Answer:
        """Parse the AI response into an Answer object."""
        # Extract code snippets
        code_snippets = self._extract_code_blocks(response)

        # Extract follow-up questions
        derived_questions = self._extract_follow_up_questions(response)

        # Calculate confidence based on response quality
        confidence = self._calculate_confidence(response, code_snippets)

        return Answer(
            question_id=question_id,
            content=response,
            code_snippets=code_snippets,
            confidence=confidence,
            derived_questions=derived_questions,
        )

    def _extract_code_blocks(self, text: str) -> list[str]:
        """Extract code blocks from markdown text."""
        # Match ```language ... ``` blocks
        pattern = r"```[\w]*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches if m.strip()]

    def _extract_follow_up_questions(self, text: str) -> list[str]:
        """Extract follow-up questions from the response."""
        questions = []

        # Look for numbered questions after "Follow-up Questions" header
        follow_up_section = re.search(
            r"Follow-up Questions[:\s]*\n(.*?)(?=\n###|\n##|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )

        if follow_up_section:
            section_text = follow_up_section.group(1)
            # Extract numbered items
            numbered_items = re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|\n*$)", section_text)
            questions.extend([q.strip() for q in numbered_items if q.strip()])

        # Also look for questions marked with "?"
        if not questions:
            all_questions = re.findall(r"([^.!?\n]+\?)", text)
            # Filter to likely follow-up questions (not rhetorical)
            for q in all_questions[-5:]:
                if len(q) > 20 and "what" in q.lower() or "how" in q.lower():
                    questions.append(q.strip())

        return questions[:5]  # Limit to 5 questions

    def _calculate_confidence(self, response: str, code_snippets: list[str]) -> float:
        """Calculate confidence score for the answer."""
        confidence = 0.5  # Base confidence

        # Increase for length (up to a point)
        word_count = len(response.split())
        if word_count > 100:
            confidence += 0.1
        if word_count > 300:
            confidence += 0.1

        # Increase for code
        if code_snippets:
            confidence += 0.15
            # Bonus for substantial code
            total_code_lines = sum(len(c.split("\n")) for c in code_snippets)
            if total_code_lines > 10:
                confidence += 0.05

        # Increase for structured response
        if "###" in response or "##" in response:
            confidence += 0.05

        # Increase for explanations
        if any(
            marker in response.lower()
            for marker in ["because", "therefore", "this is", "note that"]
        ):
            confidence += 0.05

        return min(0.95, confidence)  # Cap at 0.95
