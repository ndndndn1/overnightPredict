"""Claude/Anthropic LLM provider implementation."""

import asyncio
import subprocess
import time
from typing import Any, AsyncIterator, Optional

from src.core.config.settings import ClaudeConfig
from src.core.exceptions import LLMProviderError, RateLimitError
from src.core.utils.logging import get_logger
from src.domain.interfaces.llm_provider import (
    ICodeExecutionProvider,
    LLMMessage,
    LLMResponse,
)
from src.infrastructure.llm_providers.base_provider import BaseLLMProvider


class ClaudeProvider(BaseLLMProvider, ICodeExecutionProvider):
    """
    Claude/Anthropic API provider implementation.

    Supports both the Anthropic API and Claude Code CLI.
    """

    def __init__(self, config: ClaudeConfig):
        """Initialize the Claude provider.

        Args:
            config: Claude configuration.
        """
        super().__init__(config)
        self._claude_config = config
        self._client: Optional[Any] = None
        self._cli_available: Optional[bool] = None
        self._rate_limit_check_time: float = 0

    async def _get_client(self) -> Any:
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                api_key = self._claude_config.api_key
                if api_key is None:
                    raise LLMProviderError(
                        "Claude/Anthropic API key not configured",
                        provider=self.provider_name,
                    )

                self._client = AsyncAnthropic(
                    api_key=api_key.get_secret_value(),
                )
            except ImportError:
                raise LLMProviderError(
                    "Anthropic package not installed. Run: pip install anthropic",
                    provider=self.provider_name,
                )

        return self._client

    def _convert_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[Optional[str], list[dict[str, Any]]]:
        """Convert messages to Anthropic format.

        Returns:
            Tuple of (system_prompt, messages).
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        return system_prompt, anthropic_messages

    async def _do_complete(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop: Optional[list[str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Perform completion using Anthropic API."""
        client = await self._get_client()
        system_prompt, anthropic_messages = self._convert_messages(messages)

        try:
            response = await client.messages.create(
                model=self._config.model,
                max_tokens=max_tokens or 4096,
                system=system_prompt or "",
                messages=anthropic_messages,
                temperature=temperature,
                stop_sequences=stop,
                **kwargs,
            )

            # Extract content
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason or "end_turn",
                raw_response={
                    "id": response.id,
                    "type": response.type,
                    "role": response.role,
                    "stop_reason": response.stop_reason,
                },
            )

        except Exception as e:
            error_str = str(e)

            # Handle rate limiting
            if "rate_limit" in error_str.lower() or "429" in error_str:
                retry_after = self._parse_rate_limit_retry(e)
                self._handle_rate_limit(retry_after)
                raise RateLimitError(
                    "Claude rate limit exceeded",
                    provider=self.provider_name,
                    retry_after=retry_after,
                    cause=e,
                )

            raise LLMProviderError(
                f"Claude completion failed: {error_str}",
                provider=self.provider_name,
                cause=e,
            )

    async def _do_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop: Optional[list[str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Perform streaming completion using Anthropic API."""
        client = await self._get_client()
        system_prompt, anthropic_messages = self._convert_messages(messages)

        try:
            async with client.messages.stream(
                model=self._config.model,
                max_tokens=max_tokens or 4096,
                system=system_prompt or "",
                messages=anthropic_messages,
                temperature=temperature,
                stop_sequences=stop,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            error_str = str(e)

            if "rate_limit" in error_str.lower() or "429" in error_str:
                retry_after = self._parse_rate_limit_retry(e)
                self._handle_rate_limit(retry_after)
                raise RateLimitError(
                    "Claude rate limit exceeded",
                    provider=self.provider_name,
                    retry_after=retry_after,
                    cause=e,
                )

            raise LLMProviderError(
                f"Claude stream failed: {error_str}",
                provider=self.provider_name,
                cause=e,
            )

    def _parse_rate_limit_retry(self, error: Exception) -> float:
        """Parse retry-after from rate limit error."""
        # Default to 60 seconds
        retry_after = 60.0

        # Try to get from headers
        if hasattr(error, "response") and error.response:
            headers = getattr(error.response, "headers", {})
            if "retry-after" in headers:
                try:
                    retry_after = float(headers["retry-after"])
                except ValueError:
                    pass

        return retry_after

    # Claude Code CLI integration

    async def _check_cli_available(self) -> bool:
        """Check if Claude Code CLI is available."""
        if self._cli_available is not None:
            return self._cli_available

        cli_path = self._claude_config.cli_path or "claude"

        try:
            process = await asyncio.create_subprocess_exec(
                cli_path, "--version",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            await process.wait()
            self._cli_available = process.returncode == 0
        except Exception:
            self._cli_available = False

        return self._cli_available

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        working_directory: Optional[str] = None,
        timeout: float = 60.0,
    ) -> tuple[str, bool]:
        """Execute code using Claude Code CLI or subprocess.

        Args:
            code: Code to execute.
            language: Programming language.
            working_directory: Directory to execute in.
            timeout: Execution timeout.

        Returns:
            Tuple of (output, success).
        """
        if language == "python":
            try:
                process = await asyncio.create_subprocess_exec(
                    "python", "-c", code,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=working_directory,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )

                output = stdout.decode() + stderr.decode()
                success = process.returncode == 0
                return (output, success)

            except asyncio.TimeoutError:
                return ("Execution timed out", False)
            except Exception as e:
                return (str(e), False)

        else:
            return (f"Language {language} not directly supported", False)

    async def read_file(self, path: str) -> str:
        """Read a file."""
        import aiofiles

        try:
            async with aiofiles.open(path, "r") as f:
                return await f.read()
        except Exception as e:
            raise LLMProviderError(
                f"Failed to read file {path}: {str(e)}",
                provider=self.provider_name,
                cause=e,
            )

    async def write_file(self, path: str, content: str) -> bool:
        """Write a file."""
        import aiofiles

        try:
            async with aiofiles.open(path, "w") as f:
                await f.write(content)
            return True
        except Exception as e:
            self._logger.error(f"Failed to write file {path}", error=str(e))
            return False

    async def list_files(
        self,
        path: str,
        pattern: Optional[str] = None,
    ) -> list[str]:
        """List files in directory."""
        import os
        from pathlib import Path
        import fnmatch

        try:
            result = []
            for item in os.listdir(path):
                if pattern is None or fnmatch.fnmatch(item, pattern):
                    result.append(str(Path(path) / item))
            return result
        except Exception as e:
            self._logger.error(f"Failed to list files in {path}", error=str(e))
            return []

    async def check_rate_limit(self) -> tuple[bool, Optional[float]]:
        """Check rate limit status with periodic validation."""
        # First check base rate limit
        is_limited, retry_after = await super().check_rate_limit()
        if is_limited:
            return (is_limited, retry_after)

        # For Claude, periodically check if rate limit has reset
        if self._claude_config.rate_limit_wait:
            now = time.time()
            if now - self._rate_limit_check_time > self._claude_config.rate_limit_check_interval:
                self._rate_limit_check_time = now
                # Rate limit check is implicit in the next API call

        return (False, None)

    async def wait_for_rate_limit(self) -> None:
        """Wait for Claude rate limit with periodic checks."""
        if not self._claude_config.rate_limit_wait:
            return await super().wait_for_rate_limit()

        while True:
            is_limited, retry_after = await self.check_rate_limit()
            if not is_limited:
                return

            wait_time = min(
                retry_after or 60.0,
                self._claude_config.rate_limit_check_interval,
            )

            self._logger.info(
                "Waiting for Claude rate limit",
                wait_time=wait_time,
            )

            await asyncio.sleep(wait_time)
