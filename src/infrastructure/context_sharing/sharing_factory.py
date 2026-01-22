"""Factory for creating context sharing instances."""

from pathlib import Path
from typing import Optional

from src.core.config.settings import ContextSharingConfig, ContextSharingType
from src.core.exceptions import ConfigurationError
from src.domain.interfaces.context_store import IContextSynchronizer
from src.infrastructure.context_sharing.file_sharing import FileContextSharing
from src.infrastructure.context_sharing.cloud_sharing import CloudContextSharing


class ContextSharingFactory:
    """
    Factory for creating context sharing instances.

    Supports file-based, cloud-based, and Redis-based sharing.
    """

    @classmethod
    def create(
        cls,
        config: ContextSharingConfig,
    ) -> Optional[IContextSynchronizer]:
        """Create a context sharing instance based on configuration.

        Args:
            config: Context sharing configuration.

        Returns:
            Context synchronizer instance, or None if disabled.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if not config.enabled:
            return None

        if config.sharing_type == ContextSharingType.FILE:
            if not config.shared_path:
                raise ConfigurationError(
                    "shared_path required for file sharing",
                    details={"sharing_type": "file"},
                )
            return FileContextSharing(config.shared_path)

        elif config.sharing_type == ContextSharingType.CLOUD_BUCKET:
            if not config.bucket_name:
                raise ConfigurationError(
                    "bucket_name required for cloud sharing",
                    details={"sharing_type": "cloud_bucket"},
                )
            return CloudContextSharing(
                bucket_name=config.bucket_name,
                prefix=config.bucket_prefix,
                region=config.aws_region,
            )

        elif config.sharing_type == ContextSharingType.REDIS:
            # Redis implementation would go here
            raise ConfigurationError(
                "Redis context sharing not yet implemented",
                details={"sharing_type": "redis"},
            )

        elif config.sharing_type == ContextSharingType.REMOTE_ENV:
            # Remote environment sharing would use SSH
            raise ConfigurationError(
                "Remote environment sharing not yet implemented",
                details={"sharing_type": "remote_env"},
            )

        else:
            raise ConfigurationError(
                f"Unknown sharing type: {config.sharing_type}",
                details={"sharing_type": str(config.sharing_type)},
            )

    @classmethod
    async def create_and_initialize(
        cls,
        config: ContextSharingConfig,
    ) -> Optional[IContextSynchronizer]:
        """Create and initialize a context sharing instance.

        Args:
            config: Context sharing configuration.

        Returns:
            Initialized context synchronizer, or None if disabled.
        """
        sharing = cls.create(config)
        if sharing is None:
            return None

        # Initialize if method exists
        if hasattr(sharing, "initialize"):
            await sharing.initialize()

        return sharing
