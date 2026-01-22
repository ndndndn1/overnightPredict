"""Cloud-based context sharing implementation (S3/GCS compatible)."""

import json
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from src.core.exceptions import ContextSharingError
from src.core.utils.id_generator import generate_id
from src.core.utils.logging import get_logger
from src.domain.interfaces.context_store import IContextSynchronizer
from src.domain.value_objects.context import Context, ContextType, SharedContext


class CloudContextSharing(IContextSynchronizer):
    """
    Cloud-based context sharing using S3-compatible storage.

    Suitable for distributed sessions across different machines.
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "overnight-predict/",
        region: str = "us-east-1",
    ):
        """Initialize cloud-based sharing.

        Args:
            bucket_name: S3 bucket name.
            prefix: Key prefix for stored objects.
            region: AWS region.
        """
        self._bucket_name = bucket_name
        self._prefix = prefix
        self._region = region
        self._logger = get_logger(
            "sharing.cloud",
            bucket=bucket_name,
            prefix=prefix,
        )
        self._client: Optional[Any] = None
        self._subscriptions: dict[str, tuple[str, Callable[[SharedContext], Awaitable[None]]]] = {}

    async def _get_client(self) -> Any:
        """Get or create S3 client."""
        if self._client is None:
            try:
                import aioboto3

                session = aioboto3.Session()
                self._client = await session.client(
                    "s3",
                    region_name=self._region,
                ).__aenter__()
            except ImportError:
                raise ContextSharingError(
                    "aioboto3 package not installed. Run: pip install aioboto3",
                    sharing_type="cloud",
                )
        return self._client

    def _get_key(self, group_id: str, context_id: str) -> str:
        """Get S3 key for context."""
        return f"{self._prefix}groups/{group_id}/{context_id}.json"

    def _get_group_prefix(self, group_id: str) -> str:
        """Get S3 prefix for group."""
        return f"{self._prefix}groups/{group_id}/"

    async def sync(self, group_id: str) -> None:
        """Synchronize context for a group (no-op for cloud)."""
        self._logger.debug("Sync called", group_id=group_id)

    async def push(self, shared_context: SharedContext) -> bool:
        """Push context to cloud storage."""
        try:
            client = await self._get_client()

            context_id = generate_id(prefix="ctx")
            key = self._get_key(shared_context.group_id, context_id)

            data = {
                "id": context_id,
                "content": shared_context.context.content,
                "context_type": shared_context.context.context_type.value,
                "source": shared_context.context.source,
                "shared_by": shared_context.shared_by,
                "shared_at": shared_context.shared_at.isoformat(),
                "group_id": shared_context.group_id,
                "priority": shared_context.priority,
                "expires_at": shared_context.expires_at.isoformat() if shared_context.expires_at else None,
            }

            await client.put_object(
                Bucket=self._bucket_name,
                Key=key,
                Body=json.dumps(data),
                ContentType="application/json",
            )

            self._logger.debug(
                "Pushed context to cloud",
                context_id=context_id,
                group_id=shared_context.group_id,
            )
            return True

        except Exception as e:
            self._logger.error("Failed to push to cloud", error=str(e))
            raise ContextSharingError(
                f"Failed to push context to cloud: {str(e)}",
                sharing_type="cloud",
                cause=e,
            )

    async def pull(
        self,
        group_id: str,
        since: Optional[datetime] = None,
    ) -> list[SharedContext]:
        """Pull contexts from cloud storage."""
        try:
            client = await self._get_client()
            prefix = self._get_group_prefix(group_id)

            response = await client.list_objects_v2(
                Bucket=self._bucket_name,
                Prefix=prefix,
            )

            contexts = []

            for obj in response.get("Contents", []):
                try:
                    # Get object
                    get_response = await client.get_object(
                        Bucket=self._bucket_name,
                        Key=obj["Key"],
                    )

                    body = await get_response["Body"].read()
                    data = json.loads(body.decode("utf-8"))

                    shared_at = datetime.fromisoformat(data["shared_at"])

                    # Filter by time if specified
                    if since and shared_at < since:
                        continue

                    # Check expiration
                    expires_at = None
                    if data.get("expires_at"):
                        expires_at = datetime.fromisoformat(data["expires_at"])
                        if datetime.utcnow() > expires_at:
                            # Delete expired object
                            await client.delete_object(
                                Bucket=self._bucket_name,
                                Key=obj["Key"],
                            )
                            continue

                    context = Context(
                        content=data["content"],
                        context_type=ContextType(data["context_type"]),
                        source=data.get("source", ""),
                        timestamp=shared_at,
                    )

                    shared = SharedContext(
                        context=context,
                        shared_by=data["shared_by"],
                        shared_at=shared_at,
                        group_id=data["group_id"],
                        priority=data.get("priority", 0),
                        expires_at=expires_at,
                    )

                    contexts.append(shared)

                except Exception as e:
                    self._logger.warning(f"Failed to read object {obj['Key']}", error=str(e))

            # Sort by shared_at
            contexts.sort(key=lambda x: x.shared_at)
            return contexts

        except Exception as e:
            self._logger.error("Failed to pull from cloud", error=str(e))
            raise ContextSharingError(
                f"Failed to pull contexts from cloud: {str(e)}",
                sharing_type="cloud",
                cause=e,
            )

    async def subscribe(
        self,
        group_id: str,
        callback: Callable[[SharedContext], Awaitable[None]],
    ) -> str:
        """Subscribe to context updates (polling-based for cloud)."""
        subscription_id = generate_id(prefix="sub")
        self._subscriptions[subscription_id] = (group_id, callback)

        self._logger.info(
            "Subscribed to cloud updates (polling)",
            group_id=group_id,
            subscription_id=subscription_id,
        )

        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from updates."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
        self._subscriptions.clear()
