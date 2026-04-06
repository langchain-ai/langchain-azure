# type: ignore
"""Integration tests for AsyncCosmosDBChatMessageHistory."""

import os
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

HOST = os.environ.get("COSMOSDB_ENDPOINT", "")
KEY = os.environ.get("COSMOSDB_KEY", "")

pytestmark = pytest.mark.skipif(
    not HOST or not KEY,
    reason="COSMOSDB_ENDPOINT/COSMOSDB_KEY not set",
)


async def test_async_chat_history_add_and_retrieve() -> None:
    from langchain_azure_cosmosdb import AsyncCosmosDBChatMessageHistory

    session_id = f"test_async_session_{uuid.uuid4().hex[:8]}"
    async with AsyncCosmosDBChatMessageHistory(
        cosmos_endpoint=HOST,
        cosmos_database="test_async_chat_history_db",
        cosmos_container="test_async_chat_history",
        session_id=session_id,
        user_id="test_user",
        connection_string=f"AccountEndpoint={HOST};AccountKey={KEY}",
    ) as chat_history:
        await chat_history.aadd_messages(
            [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ]
        )

        assert len(chat_history.messages) == 2
        assert chat_history.messages[0].content == "Hello"
        assert chat_history.messages[1].content == "Hi there!"


async def test_async_chat_history_clear() -> None:
    from langchain_azure_cosmosdb import AsyncCosmosDBChatMessageHistory

    session_id = f"test_async_clear_{uuid.uuid4().hex[:8]}"
    async with AsyncCosmosDBChatMessageHistory(
        cosmos_endpoint=HOST,
        cosmos_database="test_async_chat_history_db",
        cosmos_container="test_async_chat_history",
        session_id=session_id,
        user_id="test_user",
        connection_string=f"AccountEndpoint={HOST};AccountKey={KEY}",
    ) as chat_history:
        await chat_history.aadd_messages([HumanMessage(content="Hello")])
        assert len(chat_history.messages) == 1

        await chat_history.aclear()
        assert len(chat_history.messages) == 0
