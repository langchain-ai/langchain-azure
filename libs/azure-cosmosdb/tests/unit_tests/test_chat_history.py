"""Unit tests for CosmosDBChatMessageHistory."""

from unittest.mock import MagicMock, patch

import pytest


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
