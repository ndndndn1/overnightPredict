"""
Application settings and configuration management.

Uses pydantic-settings for type-safe configuration with environment variable support.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    CLAUDE = "claude"
    ANTHROPIC = "anthropic"  # Alias for claude


class ContextSharingType(str, Enum):
    """Types of context sharing mechanisms."""

    FILE = "file"
    CLOUD_BUCKET = "cloud_bucket"
    REMOTE_ENV = "remote_env"
    REDIS = "redis"


class LLMProviderConfig(BaseSettings):
    """Configuration for a single LLM provider."""

    model_config = SettingsConfigDict(extra="allow")

    provider: LLMProvider
    api_key: Optional[SecretStr] = None
    api_base: Optional[str] = None
    model: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7
    enabled: bool = False

    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_multiplier: float = 2.0


class OpenAIConfig(LLMProviderConfig):
    """OpenAI-specific configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        extra="allow",
    )

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4-turbo-preview"
    organization_id: Optional[str] = None


class DeepSeekConfig(LLMProviderConfig):
    """DeepSeek-specific configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DEEPSEEK_",
        extra="allow",
    )

    provider: LLMProvider = LLMProvider.DEEPSEEK
    api_base: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-coder"


class ClaudeConfig(LLMProviderConfig):
    """Claude/Anthropic-specific configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_",
        extra="allow",
    )

    provider: LLMProvider = LLMProvider.CLAUDE
    model: str = "claude-3-opus-20240229"

    # Claude Code specific settings
    cli_path: Optional[str] = None  # Path to claude-code CLI
    rate_limit_wait: bool = True  # Wait when rate limited
    rate_limit_check_interval: float = 60.0  # Seconds between rate limit checks


class ContextSharingConfig(BaseSettings):
    """Configuration for context sharing."""

    model_config = SettingsConfigDict(
        env_prefix="CONTEXT_",
        extra="allow",
    )

    enabled: bool = False
    sharing_type: ContextSharingType = ContextSharingType.FILE

    # File sharing
    shared_path: Optional[Path] = None

    # Cloud bucket (S3/GCS)
    bucket_name: Optional[str] = None
    bucket_prefix: str = "overnight-predict/"
    aws_region: str = "us-east-1"

    # Redis
    redis_url: Optional[str] = None
    redis_prefix: str = "overnight:"

    # Remote environment
    remote_host: Optional[str] = None
    remote_path: Optional[str] = None
    ssh_key_path: Optional[Path] = None


class PredictionConfig(BaseSettings):
    """Configuration for prediction and evaluation."""

    model_config = SettingsConfigDict(
        env_prefix="PREDICTION_",
        extra="allow",
    )

    # Similarity threshold for prediction accuracy
    accuracy_threshold: float = 0.7

    # Number of questions to predict ahead
    lookahead_count: int = 5

    # Minimum accuracy before strategy switch
    min_accuracy_for_keep: float = 0.6

    # History window for evaluation
    evaluation_window: int = 10

    # Semantic similarity model
    similarity_model: str = "all-MiniLM-L6-v2"


class OrchestratorConfig(BaseSettings):
    """Configuration for the orchestrator."""

    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_",
        extra="allow",
    )

    # Maximum concurrent sessions
    max_sessions: int = 10

    # Session timeout in seconds
    session_timeout: float = 3600.0

    # Health check interval
    health_check_interval: float = 30.0

    # Auto-recovery on failure
    auto_recovery: bool = True


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # Application
    app_name: str = "OvernightPredict"
    debug: bool = False
    log_level: str = "INFO"

    # Working directory
    work_dir: Path = Field(default_factory=lambda: Path.cwd() / ".overnight")

    # Database
    db_path: Optional[Path] = None

    # LLM Providers
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)

    # Context sharing
    context_sharing: ContextSharingConfig = Field(default_factory=ContextSharingConfig)

    # Prediction
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)

    # Orchestrator
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)

    @field_validator("work_dir", mode="before")
    @classmethod
    def ensure_work_dir(cls, v: Any) -> Path:
        """Ensure work directory exists."""
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("db_path", mode="before")
    @classmethod
    def set_default_db_path(cls, v: Any, info: Any) -> Path:
        """Set default database path if not provided."""
        if v is None:
            work_dir = info.data.get("work_dir", Path.cwd() / ".overnight")
            return work_dir / "overnight.db"
        return Path(v) if isinstance(v, str) else v

    def get_enabled_providers(self) -> list[LLMProviderConfig]:
        """Get list of enabled LLM providers."""
        providers = []
        if self.openai.enabled:
            providers.append(self.openai)
        if self.deepseek.enabled:
            providers.append(self.deepseek)
        if self.claude.enabled:
            providers.append(self.claude)
        return providers

    def get_provider_config(self, provider: LLMProvider) -> LLMProviderConfig:
        """Get configuration for a specific provider."""
        mapping = {
            LLMProvider.OPENAI: self.openai,
            LLMProvider.DEEPSEEK: self.deepseek,
            LLMProvider.CLAUDE: self.claude,
            LLMProvider.ANTHROPIC: self.claude,
        }
        return mapping[provider]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
