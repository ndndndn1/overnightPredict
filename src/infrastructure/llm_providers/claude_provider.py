"""Claude/Anthropic LLM provider implementation."""

import asyncio
import json
import subprocess
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, AsyncIterator, Optional
from urllib.parse import parse_qs, urlparse

from src.core.config.settings import ClaudeConfig, ClaudeAuthType
from src.core.exceptions import LLMProviderError, RateLimitError
from src.core.utils.logging import get_logger
from src.domain.interfaces.llm_provider import (
    ICodeExecutionProvider,
    LLMMessage,
    LLMResponse,
)
from src.infrastructure.llm_providers.base_provider import BaseLLMProvider


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handler for OAuth callback."""

    auth_code: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self) -> None:
        """Handle GET request (OAuth callback)."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "code" in params:
            OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html><body>
                <h1>Authentication Successful!</h1>
                <p>You can close this window and return to OvernightPredict.</p>
                <script>window.close();</script>
                </body></html>
            """)
        elif "error" in params:
            OAuthCallbackHandler.error = params.get("error_description", ["Unknown error"])[0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"""
                <html><body>
                <h1>Authentication Failed</h1>
                <p>Error: {OAuthCallbackHandler.error}</p>
                </body></html>
            """.encode())

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress logging."""
        pass


class ClaudeProvider(BaseLLMProvider, ICodeExecutionProvider):
    """
    Claude/Anthropic API provider implementation.

    Supports multiple authentication methods:
    - API Key: Standard Anthropic API key
    - OAuth: Browser-based login for Claude Pro/Team subscribers
    - Session Key: Direct session key from browser cookies

    Also supports Claude Code CLI integration.
    """

    def __init__(self, config: ClaudeConfig):
        """Initialize the Claude provider.

        Args:
            config: Claude configuration.
        """
        super().__init__(config)
        self._claude_config = config
        self._client: Optional[Any] = None
        self._session_client: Optional[Any] = None  # For subscription auth
        self._cli_available: Optional[bool] = None
        self._rate_limit_check_time: float = 0
        self._oauth_token: Optional[str] = None
        self._authenticated = False

    @property
    def auth_type(self) -> ClaudeAuthType:
        """Get current authentication type."""
        return self._claude_config.auth_type

    @property
    def is_subscription_auth(self) -> bool:
        """Check if using subscription-based authentication."""
        return self._claude_config.is_subscription_auth

    async def authenticate(self) -> bool:
        """Authenticate with Claude based on configured auth type.

        Returns:
            True if authentication successful.
        """
        auth_type = self._claude_config.auth_type

        if auth_type == ClaudeAuthType.API_KEY:
            return await self._authenticate_api_key()
        elif auth_type == ClaudeAuthType.OAUTH:
            return await self._authenticate_oauth()
        elif auth_type == ClaudeAuthType.SESSION_KEY:
            return await self._authenticate_session_key()

        return False

    async def _authenticate_api_key(self) -> bool:
        """Authenticate using Anthropic API key."""
        try:
            await self._get_client()
            self._authenticated = True
            self._logger.info("Authenticated with API key")
            return True
        except Exception as e:
            self._logger.error("API key authentication failed", error=str(e))
            return False

    async def _authenticate_oauth(self) -> bool:
        """Authenticate using OAuth browser flow for Claude Pro/Team.

        This opens a browser window for the user to log in to their
        Claude account and authorize the application.
        """
        self._logger.info("Starting OAuth authentication flow")

        # OAuth configuration for Claude
        # Note: These would need to be registered with Anthropic
        client_id = "overnight-predict"
        redirect_uri = f"http://localhost:{self._claude_config.oauth_callback_port}/callback"
        auth_url = f"https://claude.ai/oauth/authorize?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code&scope=chat"

        # Start local server for callback
        server = HTTPServer(
            ("localhost", self._claude_config.oauth_callback_port),
            OAuthCallbackHandler,
        )
        server.timeout = 300  # 5 minute timeout

        # Open browser for authentication
        self._logger.info("Opening browser for authentication...")
        webbrowser.open(auth_url)

        # Wait for callback
        OAuthCallbackHandler.auth_code = None
        OAuthCallbackHandler.error = None

        try:
            while OAuthCallbackHandler.auth_code is None and OAuthCallbackHandler.error is None:
                server.handle_request()
        finally:
            server.server_close()

        if OAuthCallbackHandler.error:
            self._logger.error("OAuth failed", error=OAuthCallbackHandler.error)
            return False

        if OAuthCallbackHandler.auth_code:
            self._oauth_token = OAuthCallbackHandler.auth_code
            self._authenticated = True
            self._logger.info("OAuth authentication successful")
            return True

        return False

    async def _authenticate_session_key(self) -> bool:
        """Authenticate using session key from browser.

        The session key can be extracted from browser cookies after
        logging into claude.ai.
        """
        session_key = self._claude_config.session_key
        if session_key is None:
            self._logger.error("Session key not configured")
            return False

        try:
            # Validate session key by making a test request
            import aiohttp

            async with aiohttp.ClientSession() as session:
                headers = {
                    "Cookie": f"sessionKey={session_key.get_secret_value()}",
                    "Content-Type": "application/json",
                }
                async with session.get(
                    "https://claude.ai/api/auth/session",
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._claude_config.account_email = data.get("email")
                        self._claude_config.subscription_type = data.get("subscription_type")
                        self._authenticated = True
                        self._logger.info(
                            "Session key authentication successful",
                            email=self._claude_config.account_email,
                            subscription=self._claude_config.subscription_type,
                        )
                        return True
                    else:
                        self._logger.error("Session key invalid or expired")
                        return False
        except ImportError:
            self._logger.error("aiohttp package required for session auth. Run: pip install aiohttp")
            return False
        except Exception as e:
            self._logger.error("Session key authentication failed", error=str(e))
            return False

    async def _get_client(self) -> Any:
        """Get or create the Anthropic client based on auth type."""
        if self._client is None:
            auth_type = self._claude_config.auth_type

            if auth_type == ClaudeAuthType.API_KEY:
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

            elif auth_type in (ClaudeAuthType.OAUTH, ClaudeAuthType.SESSION_KEY):
                # For subscription auth, we use a custom client
                if not self._authenticated:
                    success = await self.authenticate()
                    if not success:
                        raise LLMProviderError(
                            "Claude subscription authentication failed",
                            provider=self.provider_name,
                        )
                # Use session-based client
                self._client = self._create_session_client()

        return self._client

    def _create_session_client(self) -> Any:
        """Create a client for session-based authentication."""
        # This creates a wrapper that uses the Claude web API
        return ClaudeSessionClient(
            session_key=self._claude_config.session_key,
            oauth_token=self._oauth_token,
            logger=self._logger,
        )

    async def get_subscription_status(self) -> dict[str, Any]:
        """Get subscription status for authenticated account.

        Returns:
            Dictionary with subscription details.
        """
        if not self.is_subscription_auth:
            return {"type": "api", "unlimited": True}

        return {
            "type": self._claude_config.subscription_type or "unknown",
            "email": self._claude_config.account_email,
            "daily_limit": self._claude_config.daily_message_limit,
            "messages_used": self._claude_config.messages_used_today,
            "messages_remaining": (
                self._claude_config.daily_message_limit - self._claude_config.messages_used_today
                if self._claude_config.daily_message_limit
                else None
            ),
        }

    def increment_message_count(self) -> None:
        """Increment the daily message count for subscription accounts."""
        if self.is_subscription_auth:
            self._claude_config.messages_used_today += 1

    def reset_daily_count(self) -> None:
        """Reset daily message count (call at midnight)."""
        self._claude_config.messages_used_today = 0

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


class ClaudeSessionClient:
    """
    Client for Claude web API using session authentication.

    This client uses the same API that the Claude web interface uses,
    allowing users with Pro/Team subscriptions to use their account.
    """

    def __init__(
        self,
        session_key: Optional[Any] = None,
        oauth_token: Optional[str] = None,
        logger: Optional[Any] = None,
    ):
        """Initialize the session client.

        Args:
            session_key: Session key from browser cookies.
            oauth_token: OAuth token from browser auth.
            logger: Logger instance.
        """
        self._session_key = session_key
        self._oauth_token = oauth_token
        self._logger = logger
        self._organization_id: Optional[str] = None
        self._conversation_id: Optional[str] = None

    async def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "OvernightPredict/1.0",
        }

        if self._session_key:
            headers["Cookie"] = f"sessionKey={self._session_key.get_secret_value()}"
        elif self._oauth_token:
            headers["Authorization"] = f"Bearer {self._oauth_token}"

        return headers

    async def create_message(
        self,
        messages: list[dict[str, Any]],
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a message using Claude web API.

        Args:
            messages: List of message dicts.
            model: Model to use.
            max_tokens: Maximum tokens.
            **kwargs: Additional parameters.

        Returns:
            Response dictionary.
        """
        import aiohttp

        headers = await self._get_headers()

        # Convert messages to Claude web API format
        prompt = self._convert_to_web_format(messages)

        payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens_to_sample": max_tokens,
        }

        if self._conversation_id:
            payload["conversation_id"] = self._conversation_id

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://claude.ai/api/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 429:
                    # Rate limited
                    retry_after = response.headers.get("Retry-After", "60")
                    raise Exception(f"Rate limited. Retry after {retry_after}s")
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")

    def _convert_to_web_format(self, messages: list[dict[str, Any]]) -> str:
        """Convert standard message format to web API format."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"Human: {content}")
        return "\n\n".join(parts)

    @property
    def messages(self) -> "ClaudeSessionClient":
        """Return self for API compatibility."""
        return self

    async def create(self, **kwargs: Any) -> Any:
        """Create method for API compatibility."""
        return await self.create_message(**kwargs)
