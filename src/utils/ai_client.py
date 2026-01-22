"""AI client abstraction for multiple providers."""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import AuthMethod, Settings

logger = structlog.get_logger(__name__)

# Path to store session credentials
CREDENTIALS_PATH = Path.home() / ".overnight" / "credentials.json"


class AIClient(ABC):
    """Abstract base class for AI clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the AI model."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Any:
        """Generate a streaming response from the AI model."""
        pass


class CredentialsManager:
    """Manage stored credentials for session-based authentication."""

    @staticmethod
    def get_credentials_path() -> Path:
        """Get the path to the credentials file."""
        return CREDENTIALS_PATH

    @staticmethod
    def load_credentials() -> dict[str, Any]:
        """Load stored credentials from file."""
        creds_path = CredentialsManager.get_credentials_path()
        if creds_path.exists():
            try:
                with open(creds_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load credentials", error=str(e))
        return {}

    @staticmethod
    def save_credentials(credentials: dict[str, Any]) -> None:
        """Save credentials to file."""
        creds_path = CredentialsManager.get_credentials_path()
        creds_path.parent.mkdir(parents=True, exist_ok=True)

        # Secure file permissions (owner read/write only)
        with open(creds_path, "w") as f:
            json.dump(credentials, f, indent=2)

        try:
            os.chmod(creds_path, 0o600)
        except OSError:
            pass  # Windows doesn't support chmod

    @staticmethod
    def clear_credentials() -> None:
        """Clear stored credentials."""
        creds_path = CredentialsManager.get_credentials_path()
        if creds_path.exists():
            creds_path.unlink()


class AnthropicClient(AIClient):
    """Anthropic Claude API client with multiple auth methods."""

    # Claude.ai API endpoints for session-based auth
    CLAUDE_API_BASE = "https://api.claude.ai"

    def __init__(self, settings: Settings) -> None:
        """Initialize the Anthropic client."""
        self.settings = settings
        self._client: Any = None
        self._http_client: httpx.AsyncClient | None = None
        self._auth_method = self._determine_auth_method()

    def _determine_auth_method(self) -> AuthMethod:
        """Determine the best available authentication method."""
        # Check explicit setting first
        if self.settings.ai.auth_method != AuthMethod.API_KEY:
            return self.settings.ai.auth_method

        # Auto-detect based on available credentials
        if self.settings.anthropic_api_key:
            return AuthMethod.API_KEY

        if self.settings.anthropic_session_token or self.settings.anthropic_session_key:
            return AuthMethod.SESSION_TOKEN

        # Check stored credentials
        creds = CredentialsManager.load_credentials()
        if creds.get("anthropic_session_token"):
            return AuthMethod.SESSION_TOKEN

        return AuthMethod.API_KEY

    def _get_session_credentials(self) -> tuple[str, str]:
        """Get session credentials from settings or stored file."""
        session_token = self.settings.anthropic_session_token
        session_key = self.settings.anthropic_session_key

        if not session_token:
            creds = CredentialsManager.load_credentials()
            session_token = creds.get("anthropic_session_token", "")
            session_key = creds.get("anthropic_session_key", "")

        return session_token, session_key

    @property
    def client(self) -> Any:
        """Get the Anthropic client (lazy init) - for API key auth."""
        if self._client is None:
            if self._auth_method != AuthMethod.API_KEY:
                raise ValueError(
                    f"Cannot use SDK client with auth method: {self._auth_method}. "
                    "Use generate() method directly."
                )
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=self.settings.anthropic_api_key
                )
            except ImportError:
                raise ImportError("anthropic package not installed")
        return self._client

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get HTTP client for session-based auth."""
        if self._http_client is None:
            session_token, session_key = self._get_session_credentials()

            if not session_token:
                raise ValueError(
                    "Session token not configured. Run 'overnight login' to authenticate."
                )

            cookies = {"sessionKey": session_token}
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "OvernightPredict/1.0",
            }

            if session_key:
                headers["Authorization"] = f"Bearer {session_key}"

            self._http_client = httpx.AsyncClient(
                base_url=self.CLAUDE_API_BASE,
                cookies=cookies,
                headers=headers,
                timeout=120.0,
            )
        return self._http_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response using Claude."""
        max_tokens = max_tokens or self.settings.ai.anthropic_max_tokens
        temperature = temperature or self.settings.ai.anthropic_temperature

        if self._auth_method == AuthMethod.API_KEY:
            return await self._generate_with_api_key(
                prompt, system_prompt, max_tokens, temperature
            )
        else:
            return await self._generate_with_session(
                prompt, system_prompt, max_tokens, temperature
            )

    async def _generate_with_api_key(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate response using API key authentication."""
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.client.messages.create(
                model=self.settings.ai.anthropic_model,
                max_tokens=max_tokens,
                system=system_prompt or "",
                messages=messages,
                temperature=temperature,
            )

            return response.content[0].text
        except Exception as e:
            logger.error("Anthropic API error", error=str(e))
            raise

    async def _generate_with_session(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate response using session-based authentication."""
        http_client = await self._get_http_client()

        # Build request payload for Claude.ai API
        payload = {
            "prompt": prompt,
            "model": self.settings.ai.anthropic_model,
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = await http_client.post(
                "/api/append_message",
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("completion", result.get("content", ""))

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error(
                    "Session expired or invalid. Run 'overnight login' to re-authenticate."
                )
            raise
        except Exception as e:
            logger.error("Claude session API error", error=str(e))
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a streaming response using Claude."""
        max_tokens = max_tokens or self.settings.ai.anthropic_max_tokens
        temperature = temperature or self.settings.ai.anthropic_temperature

        if self._auth_method == AuthMethod.API_KEY:
            async for text in self._stream_with_api_key(
                prompt, system_prompt, max_tokens, temperature
            ):
                yield text
        else:
            async for text in self._stream_with_session(
                prompt, system_prompt, max_tokens, temperature
            ):
                yield text

    async def _stream_with_api_key(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> Any:
        """Stream response using API key authentication."""
        messages = [{"role": "user", "content": prompt}]

        async with self.client.messages.stream(
            model=self.settings.ai.anthropic_model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=messages,
            temperature=temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def _stream_with_session(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> Any:
        """Stream response using session-based authentication."""
        http_client = await self._get_http_client()

        payload = {
            "prompt": prompt,
            "model": self.settings.ai.anthropic_model,
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with http_client.stream(
                "POST",
                "/api/append_message",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data and data != "[DONE]":
                            try:
                                chunk = json.loads(data)
                                if "completion" in chunk:
                                    yield chunk["completion"]
                                elif "delta" in chunk:
                                    yield chunk["delta"].get("text", "")
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error("Claude session stream error", error=str(e))
            raise

    async def close(self) -> None:
        """Close HTTP client connection."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


class OpenAIClient(AIClient):
    """OpenAI API client."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the OpenAI client."""
        self.settings = settings
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Get the OpenAI client (lazy init)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            except ImportError:
                raise ImportError("openai package not installed")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response using OpenAI."""
        max_tokens = max_tokens or self.settings.ai.openai_max_tokens
        temperature = temperature or self.settings.ai.openai_temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.ai.openai_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("OpenAI API error", error=str(e))
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a streaming response using OpenAI."""
        max_tokens = max_tokens or self.settings.ai.openai_max_tokens
        temperature = temperature or self.settings.ai.openai_temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.settings.ai.openai_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class MockAIClient(AIClient):
    """Mock AI client for testing."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the mock client."""
        self.settings = settings
        self._responses: list[str] = []
        self._response_index = 0

    def set_responses(self, responses: list[str]) -> None:
        """Set predefined responses for testing."""
        self._responses = responses
        self._response_index = 0

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a mock response."""
        if self._responses:
            response = self._responses[self._response_index % len(self._responses)]
            self._response_index += 1
            return response

        # Generate a basic response based on the prompt
        return f"""### Answer
Based on the prompt, here is a comprehensive response.

### Code
```python
def example_implementation():
    \"\"\"Example implementation based on the prompt.\"\"\"
    pass
```

### Follow-up Questions
1. What additional features should be implemented?
2. How should error handling be improved?
3. What tests are needed?
"""

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Any:
        """Generate a mock streaming response."""
        response = await self.generate(
            prompt, system_prompt, max_tokens, temperature, **kwargs
        )
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.01)


_cached_ai_client: AIClient | None = None


def clear_ai_client_cache() -> None:
    """Clear the cached AI client to pick up new credentials."""
    global _cached_ai_client
    _cached_ai_client = None


def get_ai_client(settings: Settings | None = None) -> AIClient:
    """Get the appropriate AI client based on settings."""
    global _cached_ai_client

    if _cached_ai_client is not None:
        return _cached_ai_client

    if settings is None:
        from src.core.config import get_settings
        settings = get_settings()

    # Check for stored credentials if environment variables are not set
    if not settings.anthropic_api_key and not settings.anthropic_session_token:
        creds = CredentialsManager.load_credentials()
        if creds.get("anthropic_api_key"):
            settings.anthropic_api_key = creds["anthropic_api_key"]
        if creds.get("anthropic_session_token"):
            settings.anthropic_session_token = creds["anthropic_session_token"]

    provider = settings.ai.primary_provider.lower()

    if provider == "anthropic":
        _cached_ai_client = AnthropicClient(settings)
    elif provider == "openai":
        _cached_ai_client = OpenAIClient(settings)
    elif provider == "mock":
        _cached_ai_client = MockAIClient(settings)
    else:
        logger.warning(
            "Unknown AI provider, using mock",
            provider=provider,
        )
        _cached_ai_client = MockAIClient(settings)

    return _cached_ai_client
