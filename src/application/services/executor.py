"""
Executor service (Worker) - Executes tasks and generates code.

Handles the actual execution of coding tasks, interacting with
LLM providers to generate code, modify files, and solve problems.
"""

import asyncio
import time
from typing import Any, Optional

from src.core.utils.logging import get_logger
from src.domain.entities.task import Task, TaskResult, TaskStatus, TaskType
from src.domain.entities.question import Question
from src.domain.interfaces.llm_provider import ILLMProvider, LLMMessage
from src.domain.interfaces.event_bus import IEventBus, TaskCompletedEvent


class Executor:
    """
    Executor service for running coding tasks.

    Coordinates with LLM providers to:
    - Generate code based on requirements
    - Modify existing code
    - Fix bugs
    - Run tests
    - Provide explanations
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        event_bus: Optional[IEventBus] = None,
    ):
        """Initialize the executor.

        Args:
            llm_provider: LLM provider for code generation.
            event_bus: Optional event bus for notifications.
        """
        self._llm = llm_provider
        self._event_bus = event_bus
        self._logger = get_logger(
            "executor",
            provider=llm_provider.provider_name,
        )

    async def execute(
        self,
        task: Task,
        context: Optional[str] = None,
    ) -> TaskResult:
        """Execute a task.

        Args:
            task: Task to execute.
            context: Optional additional context.

        Returns:
            Task execution result.
        """
        self._logger.info(
            "Executing task",
            task_id=task.id,
            task_type=task.task_type.value,
        )

        task.start()
        start_time = time.time()

        try:
            # Select execution method based on task type
            if task.task_type == TaskType.CODE_GENERATION:
                result = await self._execute_code_generation(task, context)
            elif task.task_type == TaskType.CODE_MODIFICATION:
                result = await self._execute_code_modification(task, context)
            elif task.task_type == TaskType.BUG_FIX:
                result = await self._execute_bug_fix(task, context)
            elif task.task_type == TaskType.ANALYSIS:
                result = await self._execute_analysis(task, context)
            else:
                result = await self._execute_generic(task, context)

            result.execution_time = time.time() - start_time
            task.complete(result)

            # Publish event
            if self._event_bus:
                await self._event_bus.publish(TaskCompletedEvent(
                    aggregate_id=task.id,
                    payload={
                        "session_id": task.session_id,
                        "success": result.success,
                        "execution_time": result.execution_time,
                    },
                ))

            self._logger.info(
                "Task completed",
                task_id=task.id,
                success=result.success,
                execution_time=result.execution_time,
            )

            return result

        except Exception as e:
            error_msg = str(e)
            task.fail(error_msg)

            self._logger.error(
                "Task execution failed",
                task_id=task.id,
                error=error_msg,
            )

            return TaskResult(
                success=False,
                output="",
                error=error_msg,
                execution_time=time.time() - start_time,
            )

    async def execute_question(
        self,
        question: Question,
        context: Optional[str] = None,
    ) -> str:
        """Execute a question directly without creating a task.

        Args:
            question: Question to answer.
            context: Optional additional context.

        Returns:
            Answer string.
        """
        self._logger.debug(
            "Answering question",
            question_id=question.id,
        )

        messages = self._build_question_messages(question, context)

        response = await self._llm.complete(
            messages=messages,
            max_tokens=4096,
        )

        answer = response.content
        question.mark_processed(answer)

        return answer

    async def _execute_code_generation(
        self,
        task: Task,
        context: Optional[str],
    ) -> TaskResult:
        """Execute a code generation task."""
        prompt = self._build_code_gen_prompt(task, context)

        response = await self._llm.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are an expert software engineer. Generate clean, well-documented code.",
                ),
                LLMMessage(role="user", content=prompt),
            ],
            max_tokens=4096,
        )

        return TaskResult(
            success=True,
            output=response.content,
            tokens_used=response.total_tokens,
            files_created=self._extract_file_paths(response.content, "created"),
        )

    async def _execute_code_modification(
        self,
        task: Task,
        context: Optional[str],
    ) -> TaskResult:
        """Execute a code modification task."""
        prompt = self._build_code_mod_prompt(task, context)

        response = await self._llm.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are an expert software engineer. Modify code carefully, preserving existing functionality unless changes are requested.",
                ),
                LLMMessage(role="user", content=prompt),
            ],
            max_tokens=4096,
        )

        return TaskResult(
            success=True,
            output=response.content,
            tokens_used=response.total_tokens,
            files_modified=self._extract_file_paths(response.content, "modified"),
        )

    async def _execute_bug_fix(
        self,
        task: Task,
        context: Optional[str],
    ) -> TaskResult:
        """Execute a bug fix task."""
        prompt = self._build_bug_fix_prompt(task, context)

        response = await self._llm.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are an expert debugger. Analyze the bug carefully and provide a minimal, targeted fix.",
                ),
                LLMMessage(role="user", content=prompt),
            ],
            max_tokens=4096,
        )

        return TaskResult(
            success=True,
            output=response.content,
            tokens_used=response.total_tokens,
            files_modified=self._extract_file_paths(response.content, "modified"),
        )

    async def _execute_analysis(
        self,
        task: Task,
        context: Optional[str],
    ) -> TaskResult:
        """Execute an analysis task."""
        prompt = self._build_analysis_prompt(task, context)

        response = await self._llm.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are an expert software analyst. Provide clear, actionable insights.",
                ),
                LLMMessage(role="user", content=prompt),
            ],
            max_tokens=4096,
        )

        return TaskResult(
            success=True,
            output=response.content,
            tokens_used=response.total_tokens,
        )

    async def _execute_generic(
        self,
        task: Task,
        context: Optional[str],
    ) -> TaskResult:
        """Execute a generic task."""
        prompt = f"""Task: {task.description}

Instructions:
{task.instructions}

{f"Context: {context}" if context else ""}"""

        response = await self._llm.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are a helpful AI assistant specialized in software development.",
                ),
                LLMMessage(role="user", content=prompt),
            ],
            max_tokens=4096,
        )

        return TaskResult(
            success=True,
            output=response.content,
            tokens_used=response.total_tokens,
        )

    def _build_question_messages(
        self,
        question: Question,
        context: Optional[str],
    ) -> list[LLMMessage]:
        """Build messages for answering a question."""
        messages = [
            LLMMessage(
                role="system",
                content="You are a helpful AI assistant specialized in software development. Provide clear, accurate, and actionable answers.",
            ),
        ]

        if context:
            messages.append(LLMMessage(
                role="user",
                content=f"Context:\n{context}\n\nQuestion: {question.content}",
            ))
        else:
            messages.append(LLMMessage(
                role="user",
                content=question.content,
            ))

        return messages

    def _build_code_gen_prompt(
        self,
        task: Task,
        context: Optional[str],
    ) -> str:
        """Build prompt for code generation."""
        parts = [
            f"## Task: {task.description}",
            f"\n## Requirements:\n{task.instructions}",
        ]

        if context:
            parts.append(f"\n## Context:\n{context}")

        parts.append("\n## Instructions:")
        parts.append("1. Generate clean, well-documented code")
        parts.append("2. Follow best practices for the language")
        parts.append("3. Include necessary imports and dependencies")
        parts.append("4. Add appropriate error handling")

        return "\n".join(parts)

    def _build_code_mod_prompt(
        self,
        task: Task,
        context: Optional[str],
    ) -> str:
        """Build prompt for code modification."""
        parts = [
            f"## Task: {task.description}",
            f"\n## Modification Required:\n{task.instructions}",
        ]

        if context:
            parts.append(f"\n## Current Code:\n{context}")

        parts.append("\n## Instructions:")
        parts.append("1. Make minimal changes to achieve the goal")
        parts.append("2. Preserve existing functionality")
        parts.append("3. Maintain code style consistency")
        parts.append("4. Update tests if applicable")

        return "\n".join(parts)

    def _build_bug_fix_prompt(
        self,
        task: Task,
        context: Optional[str],
    ) -> str:
        """Build prompt for bug fixing."""
        parts = [
            f"## Bug Description: {task.description}",
            f"\n## Expected vs Actual:\n{task.instructions}",
        ]

        if context:
            parts.append(f"\n## Relevant Code:\n{context}")

        parts.append("\n## Instructions:")
        parts.append("1. Identify the root cause")
        parts.append("2. Propose a minimal fix")
        parts.append("3. Explain why the fix works")
        parts.append("4. Note any potential side effects")

        return "\n".join(parts)

    def _build_analysis_prompt(
        self,
        task: Task,
        context: Optional[str],
    ) -> str:
        """Build prompt for analysis."""
        parts = [
            f"## Analysis Request: {task.description}",
            f"\n## Focus Areas:\n{task.instructions}",
        ]

        if context:
            parts.append(f"\n## Code/Data to Analyze:\n{context}")

        parts.append("\n## Provide:")
        parts.append("1. Key observations")
        parts.append("2. Potential issues or improvements")
        parts.append("3. Recommendations with priorities")

        return "\n".join(parts)

    def _extract_file_paths(self, content: str, action: str) -> list[str]:
        """Extract file paths mentioned in response."""
        # Simple extraction - look for common file path patterns
        import re

        paths = []
        # Match patterns like `path/to/file.py` or "path/to/file.py"
        pattern = r'[`"\']([\w./\-]+\.[a-z]+)[`"\']'
        matches = re.findall(pattern, content)

        for match in matches:
            if "/" in match or match.endswith((".py", ".js", ".ts", ".java", ".go")):
                paths.append(match)

        return list(set(paths))
