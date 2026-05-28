"""Async Azure CosmosDB Memory History."""

from __future__ import annotations

import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Type

from azure.core import MatchConditions
from azure.cosmos import PartitionKey
from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    messages_from_dict,
    messages_to_dict,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from azure.cosmos.aio import ContainerProxy

USER_AGENT = "langchain-azure-cosmosdb-chathistory"


class AsyncCosmosDBChatMessageHistory(BaseChatMessageHistory):
    """Async chat message history backed by Azure CosmosDB."""

    def __init__(
        self,
        cosmos_endpoint: str,
        cosmos_database: str,
        cosmos_container: str,
        session_id: str,
        user_id: str,
        credential: Any = None,
        connection_string: Optional[str] = None,
        ttl: Optional[int] = None,
        cosmos_client_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialise a new AsyncCosmosDBChatMessageHistory instance.

        Make sure to call ``prepare_cosmos`` or use the async context
        manager to ensure the database is ready.

        Either a credential or a connection string must be provided.

        Args:
            cosmos_endpoint: The connection endpoint for the account.
            cosmos_database: The name of the database to use.
            cosmos_container: The name of the container to use.
            session_id: The session ID to use.
            user_id: The user ID to use.
            credential: The credential for Azure Cosmos DB.
            connection_string: The connection string to authenticate.
            ttl: Time to live (seconds) for documents in the container.
            cosmos_client_kwargs: Additional kwargs for the CosmosClient.
        """
        self.cosmos_endpoint = cosmos_endpoint
        self.cosmos_database = cosmos_database
        self.cosmos_container = cosmos_container
        self.credential = credential
        self.conn_string = connection_string
        self.session_id = session_id
        self.user_id = user_id
        self.ttl = ttl

        self.messages: List[BaseMessage] = []
        self._loaded_count: int = 0
        self._etag: Optional[str] = None
        if self.credential:
            self._client = AsyncCosmosClient(
                url=self.cosmos_endpoint,
                credential=self.credential,
                user_agent=USER_AGENT,
                **cosmos_client_kwargs or {},
            )
        elif self.conn_string:
            self._client = AsyncCosmosClient.from_connection_string(
                conn_str=self.conn_string,
                user_agent=USER_AGENT,
                **cosmos_client_kwargs or {},
            )
        else:
            raise ValueError("Either a connection string or a credential must be set.")
        self._container: Optional[ContainerProxy] = None

    async def prepare_cosmos(self) -> None:
        """Prepare the CosmosDB client asynchronously.

        Use this method or the async context manager to make sure your
        database is ready.
        """
        database = await self._client.create_database_if_not_exists(
            self.cosmos_database
        )
        self._container = await database.create_container_if_not_exists(
            self.cosmos_container,
            partition_key=PartitionKey("/user_id"),
            default_ttl=self.ttl,
        )
        await self.load_messages()

    async def __aenter__(self) -> AsyncCosmosDBChatMessageHistory:
        """Async context manager entry point."""
        await self._client.__aenter__()
        await self.prepare_cosmos()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Async context manager exit."""
        try:
            await self.upsert_messages()
        finally:
            await self._client.__aexit__(exc_type, exc_val, traceback)

    async def load_messages(self) -> None:
        """Retrieve the messages from Cosmos asynchronously."""
        if not self._container:
            raise ValueError("Container not initialized")
        try:
            item = await self._container.read_item(
                item=self.session_id, partition_key=self.user_id
            )
        except CosmosHttpResponseError:
            logger.info("no session found")
            self._loaded_count = len(self.messages)
            return
        self._etag = item.get("_etag")
        if "messages" in item and len(item["messages"]) > 0:
            self.messages = messages_from_dict(item["messages"])
        self._loaded_count = len(self.messages)

    def add_message(self, message: BaseMessage) -> None:
        """Not implemented. Use ``aadd_messages`` instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use the async method `aadd_messages` instead.")

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the store asynchronously.

        Args:
            messages: List of messages to add.
        """
        self.messages.extend(messages)
        await self.upsert_messages()

    async def upsert_messages(self) -> None:
        """Update the cosmosdb item asynchronously."""
        if not self._container:
            raise ValueError("Container not initialized")

        body = {
            "id": self.session_id,
            "user_id": self.user_id,
            "messages": messages_to_dict(self.messages),
        }
        etag = getattr(self, "_etag", None)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                kwargs: dict = {"body": body}
                if etag:
                    kwargs["etag"] = etag
                    kwargs["match_condition"] = MatchConditions.IfNotModified
                response = await self._container.upsert_item(**kwargs)
                self._etag = (
                    response.get("_etag") if isinstance(response, dict) else None
                )
                self._loaded_count = len(self.messages)
                return
            except CosmosHttpResponseError as e:
                if e.status_code == 412 and attempt < max_retries - 1:
                    pending = self.messages[self._loaded_count :]
                    await self.load_messages()
                    if pending:
                        self.messages.extend(pending)
                    body["messages"] = messages_to_dict(self.messages)
                    etag = self._etag
                else:
                    logger.warning(
                        "Failed to upsert messages for session %s",
                        self.session_id,
                    )
                    raise

    def clear(self) -> None:
        """Not implemented. Use ``aclear`` instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use the async method `aclear` instead.")

    async def aclear(self) -> None:
        """Clear session memory from this memory and cosmos."""
        self.messages = []
        self._loaded_count = 0
        self._etag = None
        if self._container:
            try:
                await self._container.delete_item(
                    item=self.session_id, partition_key=self.user_id
                )
            except Exception:
                logger.warning("Failed to delete session %s", self.session_id)
                raise
