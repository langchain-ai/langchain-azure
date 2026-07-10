"""Integration tests for the async surface on SQLServerVectorStore."""

import os
import uuid
from typing import AsyncGenerator, List

import pytest
import pytest_asyncio
from langchain_core.documents import Document

from langchain_sqlserver.vectorstores import SQLServerVectorStore
from tests.utils.fake_embeddings import DeterministicFakeEmbedding

_CONNECTION_STRING = str(os.environ.get("TEST_AZURESQLSERVER_TRUSTED_CONNECTION"))
EMBEDDING_LENGTH = 1536


def _unique_table() -> str:
    """Build a per-test table name so concurrent runs do not collide."""
    return f"lc_test_async_{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def store() -> AsyncGenerator[SQLServerVectorStore, None]:
    """Provide a per-test vector store and drop the underlying table on
    teardown."""
    store = SQLServerVectorStore(
        connection_string=_CONNECTION_STRING,
        embedding_length=EMBEDDING_LENGTH,
        embedding_function=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
        table_name=_unique_table(),
        batch_size=200,
    )
    yield store
    store.drop()


@pytest.mark.asyncio
async def test_aadd_texts_returns_ids_for_each_input(
    store: SQLServerVectorStore,
) -> None:
    """``aadd_texts`` persists every input and returns matching ids, so
    callers can immediately reference rows they just inserted."""
    texts = ["alpha", "beta", "gamma"]
    metadatas = [{"i": i} for i in range(len(texts))]
    ids = await store.aadd_texts(texts, metadatas)
    assert len(ids) == len(texts)


@pytest.mark.asyncio
async def test_aadd_documents_round_trips_through_aget_by_ids(
    store: SQLServerVectorStore,
) -> None:
    """Documents inserted via ``aadd_documents`` round-trip through
    ``aget_by_ids``, including their metadata."""
    docs: List[Document] = [
        Document(page_content="rabbit", metadata={"color": "black"}),
        Document(page_content="cherry", metadata={"color": "red"}),
    ]
    ids = await store.aadd_documents(docs, ids=["a", "b"])
    assert ids == ["a", "b"]
    fetched = {d.id: d for d in await store.aget_by_ids(["a", "b"])}
    assert fetched["a"].page_content == "rabbit"
    assert fetched["a"].metadata == {"color": "black"}
    assert fetched["b"].page_content == "cherry"


@pytest.mark.asyncio
async def test_asimilarity_search_returns_expected_count(
    store: SQLServerVectorStore,
) -> None:
    """``asimilarity_search`` honors ``k`` and returns ``Document`` rows."""
    await store.aadd_texts(["red", "blue", "green", "yellow", "purple"])
    result = await store.asimilarity_search("red", k=3)
    assert len(result) == 3
    assert all(isinstance(d, Document) for d in result)


@pytest.mark.asyncio
async def test_asimilarity_search_with_score_orders_by_distance(
    store: SQLServerVectorStore,
) -> None:
    """``asimilarity_search_with_score`` returns ``(Document, distance)``
    pairs ordered by ascending distance (lower = more similar)."""
    await store.aadd_texts(["red", "blue", "green"])
    pairs = await store.asimilarity_search_with_score("red", k=3)
    scores = [score for _doc, score in pairs]
    assert scores == sorted(scores)


@pytest.mark.asyncio
async def test_adelete_by_ids_removes_only_matching_rows(
    store: SQLServerVectorStore,
) -> None:
    """``adelete(ids=[...])`` removes only the rows whose ``custom_id``
    matches and leaves the others intact."""
    await store.aadd_texts(["a", "b", "c"], ids=["1", "2", "3"])
    assert await store.adelete(["1", "3"]) is True
    remaining = await store.aget_by_ids(["1", "2", "3"])
    assert [d.id for d in remaining] == ["2"]


@pytest.mark.asyncio
async def test_adelete_none_clears_table(
    store: SQLServerVectorStore,
) -> None:
    """``adelete(None)`` removes every row in the table, matching the sync
    `delete(None)` semantics."""
    await store.aadd_texts(["a", "b"], ids=["1", "2"])
    assert await store.adelete(None) is True
    assert await store.aget_by_ids(["1", "2"]) == []


@pytest.mark.asyncio
async def test_afrom_documents_creates_and_populates_store() -> None:
    """``afrom_documents`` builds a new store and inserts the documents
    in one call — same contract as the sync ``from_documents``."""
    docs = [Document(page_content=f"item-{i}") for i in range(3)]
    table = _unique_table()
    vs = await SQLServerVectorStore.afrom_documents(
        connection_string=_CONNECTION_STRING,
        embedding=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
        embedding_length=EMBEDDING_LENGTH,
        table_name=table,
        documents=docs,
    )
    try:
        result = await vs.asimilarity_search("item", k=3)
        assert len(result) == 3
    finally:
        vs.drop()
