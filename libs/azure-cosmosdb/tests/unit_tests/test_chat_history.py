"""Unit tests for CosmosDBChatMessageHistory."""

from unittest.mock import MagicMock, patch

import pytest
from azure.core import MatchConditions


def test_missing_credential_and_connection_string_raises() -> None:
    with patch("azure.cosmos.CosmosClient"):
        from langchain_azure_cosmosdb import CosmosDBChatMessageHistory

        with pytest.raises(
            ValueError,
            match="Either a connection string or a credential must be set",
        ):
            CosmosDBChatMessageHistory(
                cosmos_endpoint="https://fake.documents.azure.com:443/",
                cosmos_database="testdb",
                cosmos_container="testcontainer",
                session_id="s1",
                user_id="u1",
                credential=None,
                connection_string=None,
            )


def test_init_with_connection_string() -> None:
    mock_client = MagicMock()
    with patch(
        "azure.cosmos.CosmosClient.from_connection_string",
        return_value=mock_client,
    ):
        from langchain_azure_cosmosdb import CosmosDBChatMessageHistory

        history = CosmosDBChatMessageHistory(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_database="testdb",
            cosmos_container="testcontainer",
            session_id="s1",
            user_id="u1",
            connection_string="AccountEndpoint=https://fake;AccountKey=fakekey==;",
        )
        assert history.session_id == "s1"
        assert history.user_id == "u1"


def test_init_with_credential() -> None:
    mock_client = MagicMock()
    with patch(
        "azure.cosmos.CosmosClient",
        return_value=mock_client,
    ):
        from langchain_azure_cosmosdb import CosmosDBChatMessageHistory

        history = CosmosDBChatMessageHistory(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_database="testdb",
            cosmos_container="testcontainer",
            session_id="s1",
            user_id="u1",
            credential="fake-credential",
        )
        assert history.session_id == "s1"
        assert history.messages == []


# ---------------------------------------------------------------------------
# ETag-based concurrency on upsert_messages
# ---------------------------------------------------------------------------


def _make_history():
    mock_client = MagicMock()
    with patch("azure.cosmos.CosmosClient", return_value=mock_client):
        from langchain_azure_cosmosdb import CosmosDBChatMessageHistory

        history = CosmosDBChatMessageHistory(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_database="testdb",
            cosmos_container="testcontainer",
            session_id="s1",
            user_id="u1",
            credential="fake-credential",
        )
        history._container = MagicMock()
        return history


def test_load_messages_captures_etag() -> None:
    history = _make_history()
    history._container.read_item.return_value = {
        "id": "s1",
        "user_id": "u1",
        "messages": [],
        "_etag": '"etag-123"',
    }
    history.load_messages()
    assert history._etag == '"etag-123"'


def test_load_messages_no_session_no_etag() -> None:
    from azure.cosmos.exceptions import CosmosHttpResponseError

    history = _make_history()
    history._container.read_item.side_effect = CosmosHttpResponseError(
        status_code=404, message="not found"
    )
    history.load_messages()
    assert not hasattr(history, "_etag") or history._etag is None


def test_upsert_passes_etag_when_available() -> None:
    from langchain_core.messages import HumanMessage

    history = _make_history()
    history._etag = '"etag-abc"'
    history.messages = [HumanMessage(content="hi")]
    history._container.read_item.return_value = {"_etag": '"etag-new"'}

    history.upsert_messages()

    call_kwargs = history._container.upsert_item.call_args[1]
    assert call_kwargs["etag"] == '"etag-abc"'
    assert call_kwargs["match_condition"] == MatchConditions.IfNotModified


def test_upsert_retries_on_412_conflict() -> None:
    from azure.cosmos.exceptions import CosmosHttpResponseError
    from langchain_core.messages import HumanMessage

    history = _make_history()
    history._etag = '"stale"'
    history.messages = [HumanMessage(content="m1")]

    history._container.upsert_item.side_effect = [
        CosmosHttpResponseError(status_code=412, message="Precondition Failed"),
        None,
    ]
    history._container.read_item.return_value = {
        "id": "s1",
        "user_id": "u1",
        "messages": [],
        "_etag": '"fresh"',
    }
    history.upsert_messages()
    assert history._container.upsert_item.call_count == 2


def test_upsert_raises_on_non_412_error() -> None:
    from azure.cosmos.exceptions import CosmosHttpResponseError
    from langchain_core.messages import HumanMessage

    history = _make_history()
    history._etag = '"e"'
    history.messages = [HumanMessage(content="x")]
    history._container.upsert_item.side_effect = CosmosHttpResponseError(
        status_code=500, message="Server Error"
    )
    with pytest.raises(CosmosHttpResponseError):
        history.upsert_messages()
