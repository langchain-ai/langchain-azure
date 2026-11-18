from __future__ import annotations

from unittest import mock

import pytest
from langchain_core.documents import Document

from langchain_azure_kusto import InvalidKustoInputError, KustoDatabase, KustoLoader
from tests.unit_tests.conftest import response, table


@pytest.fixture
def rows() -> object:
    return response(
        table(["Id", "Text", "Tags"], [[1, "hello", ["a"]], [2, None, None]]),
        table(["Id", "Text", "Tags"], [[3, "bye", []]]),
    )


def test_load_uses_all_columns_and_results(
    db: KustoDatabase, sync_client: mock.MagicMock, rows: object
) -> None:
    sync_client.return_value.execute_query.return_value = rows

    docs = KustoLoader("T", db).load()

    assert docs == [
        Document(page_content='Id: 1\nText: hello\nTags: ["a"]'),
        Document(page_content="Id: 2\nText: \nTags: "),
        Document(page_content="Id: 3\nText: bye\nTags: []"),
    ]
    options = sync_client.return_value.execute_query.call_args.args[2]._options
    assert options["request_readonly"] is True
    assert "query_take_max_records" not in options


def test_load_splits_content_and_metadata(
    db: KustoDatabase, sync_client: mock.MagicMock, rows: object
) -> None:
    sync_client.return_value.execute_query.return_value = rows

    docs = KustoLoader("T", db, metadata_columns=["Id", "Tags"]).load()
    assert docs[0] == Document(
        page_content="Text: hello", metadata={"Id": 1, "Tags": ["a"]}
    )

    docs = KustoLoader(
        "T", db, page_content_columns=["Text"], metadata_columns=["Id"]
    ).load()
    assert docs[0] == Document(page_content="Text: hello", metadata={"Id": 1})


def test_unknown_columns_are_reported(
    db: KustoDatabase, sync_client: mock.MagicMock, rows: object
) -> None:
    sync_client.return_value.execute_query.return_value = rows
    loader = KustoLoader("T", db, page_content_columns=["Body"])
    with pytest.raises(ValueError, match="Body.*Available columns: Id, Text, Tags"):
        loader.load()


def test_management_commands_are_rejected(
    db: KustoDatabase, sync_client: mock.MagicMock
) -> None:
    with pytest.raises(InvalidKustoInputError):
        KustoLoader(".show tables", db).load()
    sync_client.assert_not_called()


async def test_async_load(
    db: KustoDatabase, async_client: mock.MagicMock, rows: object
) -> None:
    async_client.return_value.execute_query.return_value = rows

    docs = await KustoLoader("T", db, metadata_columns=["Id"]).aload()

    assert [doc.metadata["Id"] for doc in docs] == [1, 2, 3]
    assert docs[2].page_content == "Text: bye\nTags: []"
