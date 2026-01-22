"""Factory for creating LLM providers."""

from typing import Optional

from src.core.config.settings import (
    Settings,
    LLMProvider,
    LLMProviderConfig,
    OpenAIConfig,
    DeepSeekConfig,
    ClaudeConfig,
)
from src.core.exceptions import ConfigurationError
from src.domain.interfaces.llm_provider import ILLMProvider
from src.infrastructure.llm_providers.openai_provider import OpenAIProvider
from src.infrastructure.llm_providers.deepseek_provider import DeepSeekProvider
from src.infrastructure.llm_providers.claude_provider import ClaudeProvider


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.

    Supports creating providers from configuration or explicit parameters.
    """

    _instances: dict[str, ILLMProvider] = {}

    @classmethod
    def create(
        cls,
        provider: LLMProvider,
        config: Optional[LLMProviderConfig] = None,
        settings: Optional[Settings] = None,
    ) -> ILLMProvider:
        """Create an LLM provider instance.

        Args:
            provider: Provider type to create.
            config: Optional explicit configuration.
            settings: Optional settings to get configuration from.

        Returns:
            LLM provider instance.

        Raises:
            ConfigurationError: If provider cannot be created.
        """
        # Get configuration
        if config is None:
            if settings is None:
                from src.core.config.settings import get_settings
                settings = get_settings()

            config = settings.get_provider_config(provider)

        # Create provider based on type
        if provider == LLMProvider.OPENAI:
            if not isinstance(config, OpenAIConfig):
                config = OpenAIConfig(**config.__dict__)
            return OpenAIProvider(config)

        elif provider == LLMProvider.DEEPSEEK:
            if not isinstance(config, DeepSeekConfig):
                config = DeepSeekConfig(**config.__dict__)
            return DeepSeekProvider(config)

        elif provider in (LLMProvider.CLAUDE, LLMProvider.ANTHROPIC):
            if not isinstance(config, ClaudeConfig):
                config = ClaudeConfig(**config.__dict__)
            return ClaudeProvider(config)

        else:
            raise ConfigurationError(
                f"Unknown provider: {provider}",
                details={"provider": provider},
            )

    @classmethod
    def get_or_create(
        cls,
        provider: LLMProvider,
        config: Optional[LLMProviderConfig] = None,
        settings: Optional[Settings] = None,
    ) -> ILLMProvider:
        """Get existing or create new provider instance.

        Uses singleton pattern per provider type.

        Args:
            provider: Provider type.
            config: Optional configuration.
            settings: Optional settings.

        Returns:
            LLM provider instance.
        """
        key = f"{provider.value}"

        if key not in cls._instances:
            cls._instances[key] = cls.create(provider, config, settings)

        return cls._instances[key]

    @classmethod
    def create_all_enabled(
        cls,
        settings: Optional[Settings] = None,
    ) -> dict[str, ILLMProvider]:
        """Create instances for all enabled providers.

        Args:
            settings: Optional settings.

        Returns:
            Dictionary of provider name to instance.
        """
        if settings is None:
            from src.core.config.settings import get_settings
            settings = get_settings()

        providers = {}

        for config in settings.get_enabled_providers():
            try:
                provider = cls.create(config.provider, config, settings)
                providers[config.provider.value] = provider
            except Exception as e:
                # Log but continue with other providers
                import logging
                logging.warning(f"Failed to create provider {config.provider}: {e}")

        return providers

    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached instances."""
        cls._instances.clear()


async def setup_providers(settings: Optional[Settings] = None) -> dict[str, ILLMProvider]:
    """Set up and verify all enabled providers.

    Args:
        settings: Optional settings.

    Returns:
        Dictionary of available providers.
    """
    providers = LLMProviderFactory.create_all_enabled(settings)

    # Health check all providers
    available = {}
    for name, provider in providers.items():
        if await provider.health_check():
            available[name] = provider

    return available
