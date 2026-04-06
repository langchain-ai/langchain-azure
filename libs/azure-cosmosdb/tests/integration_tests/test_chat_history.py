# type: ignore
import os
import uuid

import pytest
from langchain_azure_cosmosdb import CosmosDBChatMessageHistory

HOST = os.environ.get("COSMOSDB_ENDPOINT", "")
KEY = os.environ.get("COSMOSDB_KEY", "")

pytestmark = pytest.mark.skipif(
    not HOST or not KEY,
    reason="COSMOSDB_ENDPOINT/COSMOSDB_KEY not set",
)


def test_chat_history_add_and_retrieve() -> None:
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    with CosmosDBChatMessageHistory(
        cosmos_endpoint=HOST,
        cosmos_database="test_chat_history_db",
        cosmos_container="test_chat_history",
        session_id=session_id,
        user_id="test_user",
        connection_string=f"AccountEndpoint={HOST};AccountKey={KEY}",
    ) as chat_history:
        from langchain_core.messages import AIMessage, HumanMessage

        chat_history.add_message(HumanMessage(content="Hello"))
        chat_history.add_message(AIMessage(content="Hi there!"))

        assert len(chat_history.messages) == 2
        assert chat_history.messages[0].content == "Hello"
        assert chat_history.messages[1].content == "Hi there!"


def test_chat_history_clear() -> None:
    session_id = f"test_clear_{uuid.uuid4().hex[:8]}"
    with CosmosDBChatMessageHistory(
        cosmos_endpoint=HOST,
        cosmos_database="test_chat_history_db",
        cosmos_container="test_chat_history",
        session_id=session_id,
        user_id="test_user",
        connection_string=f"AccountEndpoint={HOST};AccountKey={KEY}",
    ) as chat_history:
        from langchain_core.messages import HumanMessage

        chat_history.add_message(HumanMessage(content="Hello"))
        assert len(chat_history.messages) == 1

        chat_history.clear()
        assert len(chat_history.messages) == 0
