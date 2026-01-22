"""Configuration management for OvernightPredict."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AIConfig(BaseModel):
    """AI provider configuration."""

    primary_provider: str = "anthropic"
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_max_tokens: int = 8192
    anthropic_temperature: float = 0.7
    openai_model: str = "gpt-4-turbo"
    openai_max_tokens: int = 8192
    openai_temperature: float = 0.7
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000


class PredictionConfig(BaseModel):
    """Prediction system configuration."""

    accuracy_threshold: float = 0.7
    lookahead_count: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    initial_strategy: str = "context_based"
    available_strategies: list[str] = Field(
        default_factory=lambda: [
            "context_based",
            "pattern_matching",
            "semantic_similarity",
            "hybrid",
        ]
    )
    adaptation_rate: float = 0.1
    min_samples_for_adjustment: int = 10


class SessionConfig(BaseModel):
    """Session management configuration."""

    max_parallel_sessions: int = 10
    session_timeout_minutes: int = 60
    checkpoint_interval_seconds: int = 30
    auto_scale_enabled: bool = True
    min_sessions: int = 2
    max_sessions: int = 20
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3


class CodeGenerationConfig(BaseModel):
    """Code generation configuration."""

    languages: list[str] = Field(
        default_factory=lambda: ["python", "typescript", "javascript", "go", "rust", "java"]
    )
    default_template: str = "enterprise"
    lint_on_generate: bool = True
    test_on_generate: bool = True
    security_scan: bool = True


class StorageConfig(BaseModel):
    """Storage configuration."""

    database_type: str = "sqlite"
    database_path: str = "./data/overnight.db"
    cache_type: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    file_enabled: bool = True
    file_path: str = "./logs/overnight.log"
    rotation: str = "daily"
    retention_days: int = 30


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_enabled: bool = True
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"]
    )


class Settings(BaseSettings):
    """Main settings class combining all configurations."""

    # Environment variables
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    database_url: str = Field(default="sqlite:///./data/overnight.db", alias="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    secret_key: str = Field(default="dev-secret-key-change-in-production", alias="SECRET_KEY")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Nested configurations
    ai: AIConfig = Field(default_factory=AIConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    sessions: SessionConfig = Field(default_factory=SessionConfig)
    code_generation: CodeGenerationConfig = Field(default_factory=CodeGenerationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Settings":
        """Load settings from YAML file and environment variables."""
        yaml_path = Path(yaml_path)

        if yaml_path.exists():
            with open(yaml_path) as f:
                yaml_config = yaml.safe_load(f) or {}
        else:
            yaml_config = {}

        # Transform YAML structure to flat config
        settings_dict = cls._transform_yaml_config(yaml_config)

        return cls(**settings_dict)

    @staticmethod
    def _transform_yaml_config(yaml_config: dict[str, Any]) -> dict[str, Any]:
        """Transform nested YAML config to settings format."""
        result: dict[str, Any] = {}

        # AI config
        if "ai" in yaml_config:
            ai_cfg = yaml_config["ai"]
            result["ai"] = AIConfig(
                primary_provider=ai_cfg.get("primary_provider", "anthropic"),
                anthropic_model=ai_cfg.get("anthropic", {}).get(
                    "model", "claude-sonnet-4-20250514"
                ),
                anthropic_max_tokens=ai_cfg.get("anthropic", {}).get("max_tokens", 8192),
                anthropic_temperature=ai_cfg.get("anthropic", {}).get("temperature", 0.7),
                openai_model=ai_cfg.get("openai", {}).get("model", "gpt-4-turbo"),
                openai_max_tokens=ai_cfg.get("openai", {}).get("max_tokens", 8192),
                openai_temperature=ai_cfg.get("openai", {}).get("temperature", 0.7),
                requests_per_minute=ai_cfg.get("rate_limit", {}).get("requests_per_minute", 60),
                tokens_per_minute=ai_cfg.get("rate_limit", {}).get("tokens_per_minute", 100000),
            )

        # Prediction config
        if "prediction" in yaml_config:
            pred_cfg = yaml_config["prediction"]
            strategy_cfg = pred_cfg.get("strategy", {})
            result["prediction"] = PredictionConfig(
                accuracy_threshold=pred_cfg.get("accuracy_threshold", 0.7),
                lookahead_count=pred_cfg.get("lookahead_count", 5),
                embedding_model=pred_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
                initial_strategy=strategy_cfg.get("initial_strategy", "context_based"),
                available_strategies=strategy_cfg.get(
                    "available_strategies",
                    ["context_based", "pattern_matching", "semantic_similarity", "hybrid"],
                ),
                adaptation_rate=strategy_cfg.get("adaptation_rate", 0.1),
                min_samples_for_adjustment=strategy_cfg.get("min_samples_for_adjustment", 10),
            )

        # Sessions config
        if "sessions" in yaml_config:
            sess_cfg = yaml_config["sessions"]
            auto_scale = sess_cfg.get("auto_scale", {})
            result["sessions"] = SessionConfig(
                max_parallel_sessions=sess_cfg.get("max_parallel_sessions", 10),
                session_timeout_minutes=sess_cfg.get("session_timeout_minutes", 60),
                checkpoint_interval_seconds=sess_cfg.get("checkpoint_interval_seconds", 30),
                auto_scale_enabled=auto_scale.get("enabled", True),
                min_sessions=auto_scale.get("min_sessions", 2),
                max_sessions=auto_scale.get("max_sessions", 20),
                scale_up_threshold=auto_scale.get("scale_up_threshold", 0.8),
                scale_down_threshold=auto_scale.get("scale_down_threshold", 0.3),
            )

        # Code generation config
        if "code_generation" in yaml_config:
            code_cfg = yaml_config["code_generation"]
            quality_cfg = code_cfg.get("quality", {})
            result["code_generation"] = CodeGenerationConfig(
                languages=code_cfg.get(
                    "languages", ["python", "typescript", "javascript", "go", "rust", "java"]
                ),
                default_template=code_cfg.get("default_template", "enterprise"),
                lint_on_generate=quality_cfg.get("lint_on_generate", True),
                test_on_generate=quality_cfg.get("test_on_generate", True),
                security_scan=quality_cfg.get("security_scan", True),
            )

        # Storage config
        if "storage" in yaml_config:
            storage_cfg = yaml_config["storage"]
            db_cfg = storage_cfg.get("database", {})
            cache_cfg = storage_cfg.get("cache", {})
            result["storage"] = StorageConfig(
                database_type=db_cfg.get("type", "sqlite"),
                database_path=db_cfg.get("path", "./data/overnight.db"),
                cache_type=cache_cfg.get("type", "memory"),
                redis_url=cache_cfg.get("redis_url", "redis://localhost:6379/0"),
                cache_ttl_seconds=cache_cfg.get("ttl_seconds", 3600),
            )

        # Logging config
        if "logging" in yaml_config:
            log_cfg = yaml_config["logging"]
            file_cfg = log_cfg.get("file", {})
            result["logging"] = LoggingConfig(
                level=log_cfg.get("level", "INFO"),
                format=log_cfg.get("format", "json"),
                file_enabled=file_cfg.get("enabled", True),
                file_path=file_cfg.get("path", "./logs/overnight.log"),
                rotation=file_cfg.get("rotation", "daily"),
                retention_days=file_cfg.get("retention_days", 30),
            )

        # API config
        if "api" in yaml_config:
            api_cfg = yaml_config["api"]
            cors_cfg = api_cfg.get("cors", {})
            result["api"] = APIConfig(
                host=api_cfg.get("host", "0.0.0.0"),
                port=api_cfg.get("port", 8000),
                workers=api_cfg.get("workers", 4),
                cors_enabled=cors_cfg.get("enabled", True),
                cors_origins=cors_cfg.get(
                    "origins", ["http://localhost:3000", "http://localhost:8080"]
                ),
            )

        return result


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

    if config_path.exists():
        return Settings.from_yaml(config_path)

    return Settings()
