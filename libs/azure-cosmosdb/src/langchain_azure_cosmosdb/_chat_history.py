"""Azure CosmosDB Memory History."""

from __future__ import annotations

import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, List, Optional, Type

from azure.core import MatchConditions
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosHttpResponseError
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    messages_from_dict,
    messages_to_dict,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from azure.cosmos import ContainerProxy

USER_AGENT = "langchain-azure-cosmosdb-chathistory"


class CosmosDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history backed by Azure CosmosDB."""

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
        """Initializes a new instance of the CosmosDBChatMessageHistory class.

        Make sure to call prepare_cosmos or use the context manager to make
        sure your database is ready.

        Either a credential or a connection string must be provided.

        :param cosmos_endpoint: The connection endpoint for the Azure Cosmos DB account.
        :param cosmos_database: The name of the database to use.
        :param cosmos_container: The name of the container to use.
        :param session_id: The session ID to use, can be overwritten while loading.
        :param user_id: The user ID to use, can be overwritten while loading.
        :param credential: The credential to use to authenticate to Azure Cosmos DB.
        :param connection_string: The connection string to use to authenticate.
        :param ttl: The time to live (in seconds) to use for documents in the container.
        :param cosmos_client_kwargs: Additional kwargs to pass to the CosmosClient.
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
            self._client = CosmosClient(
                url=self.cosmos_endpoint,
                credential=self.credential,
                user_agent=USER_AGENT,
                **cosmos_client_kwargs or {},
            )
        elif self.conn_string:
            self._client = CosmosClient.from_connection_string(
                conn_str=self.conn_string,
                user_agent=USER_AGENT,
                **cosmos_client_kwargs or {},
            )
        else:
            raise ValueError("Either a connection string or a credential must be set.")
        self._container: Optional[ContainerProxy] = None

    def prepare_cosmos(self) -> None:
        """Prepare the CosmosDB client.

        Use this function or the context manager to make sure your database is ready.
        """
        database = self._client.create_database_if_not_exists(self.cosmos_database)
        self._container = database.create_container_if_not_exists(
            self.cosmos_container,
            partition_key=PartitionKey("/user_id"),
            default_ttl=self.ttl,
        )
        self.load_messages()

    def __enter__(self) -> "CosmosDBChatMessageHistory":
        """Context manager entry point."""
        self._client.__enter__()
        self.prepare_cosmos()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Context manager exit."""
        try:
            self.upsert_messages()
        finally:
            self._client.__exit__(exc_type, exc_val, traceback)

    def load_messages(self) -> None:
        """Retrieve the messages from Cosmos."""
        if not self._container:
            raise ValueError("Container not initialized")
        try:
            item = self._container.read_item(
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
        """Add a self-created message to the store."""
        self.messages.append(message)
        self.upsert_messages()

    def upsert_messages(self) -> None:
        """Update the cosmosdb item."""
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
                response = self._container.upsert_item(**kwargs)
                self._etag = (
                    response.get("_etag") if isinstance(response, dict) else None
                )
                self._loaded_count = len(self.messages)
                return
            except CosmosHttpResponseError as e:
                if e.status_code == 412 and attempt < max_retries - 1:
                    pending = self.messages[self._loaded_count :]
                    self.load_messages()
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
        """Clear session memory from this memory and cosmos."""
        self.messages = []
        self._loaded_count = 0
        self._etag = None
        if self._container:
            try:
                self._container.delete_item(
                    item=self.session_id, partition_key=self.user_id
                )
            except Exception:
                logger.warning("Failed to delete session %s", self.session_id)
                raise
