"""Unit tests for AzureAISearchRetriever."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from langchain_azure_ai.retrievers.azure_ai_search import AzureAISearchRetriever


def make_retriever(filter: str | None = None) -> AzureAISearchRetriever:
    return AzureAISearchRetriever(
        service_name="my-svc", index_name="my-index", api_key="k", filter=filter
    )


def test_build_search_url_preserves_query_with_ampersand() -> None:
    """A raw '&' in the query used to truncate everything after it."""
    retriever = make_retriever()

    url = retriever._build_search_url("cats & dogs")

    query_params = parse_qs(urlparse(url).query)
    assert query_params["search"] == ["cats & dogs"]


def test_build_search_url_preserves_query_with_percent_and_hash() -> None:
    retriever = make_retriever()

    url = retriever._build_search_url("100% #trending")

    query_params = parse_qs(urlparse(url).query)
    assert query_params["search"] == ["100% #trending"]


def test_build_search_url_does_not_allow_query_to_inject_params() -> None:
    """A query shaped like '&$filter=...' must stay part of the search text."""
    retriever = make_retriever(filter="category eq 'clothing'")

    url = retriever._build_search_url("foo&$filter=1 eq 1")

    query_params = parse_qs(urlparse(url).query)
    assert query_params["search"] == ["foo&$filter=1 eq 1"]
    assert query_params["$filter"] == ["category eq 'clothing'"]


def test_build_search_url_encodes_filter() -> None:
    retriever = make_retriever(filter="category eq 'clothing & shoes'")

    url = retriever._build_search_url("shirts")

    query_params = parse_qs(urlparse(url).query)
    assert query_params["$filter"] == ["category eq 'clothing & shoes'"]


def test_get_relevant_documents_sends_full_query() -> None:
    response = MagicMock(status_code=200)
    response.text = json.dumps(
        {"value": [{"id": "1", "content": "hello world", "score": 0.9}]}
    )

    with patch("requests.get", return_value=response) as mock_get:
        retriever = make_retriever()
        docs = retriever._get_relevant_documents("cats & dogs", run_manager=MagicMock())

    sent_url = mock_get.call_args.args[0]
    query_params = parse_qs(urlparse(sent_url).query)
    assert query_params["search"] == ["cats & dogs"]
    assert len(docs) == 1
    assert docs[0].page_content == "hello world"
    assert docs[0].metadata == {"id": "1", "score": 0.9}


def make_async_session(status: int, body: dict) -> tuple[MagicMock, MagicMock]:
    response = AsyncMock()
    response.status = status
    response.json = AsyncMock(return_value=body)
    response_cm = MagicMock()
    response_cm.__aenter__ = AsyncMock(return_value=response)
    response_cm.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(return_value=response_cm)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)
    return session_cm, session


async def test_aget_relevant_documents_sends_full_query() -> None:
    session_cm, session = make_async_session(
        200, {"value": [{"id": "1", "content": "hello world"}]}
    )

    retriever = make_retriever()
    with patch("aiohttp.ClientSession", return_value=session_cm):
        docs = await retriever._aget_relevant_documents(
            "cats & dogs", run_manager=MagicMock()
        )

    sent_url = session.get.call_args.args[0]
    query_params = parse_qs(urlparse(sent_url).query)
    assert query_params["search"] == ["cats & dogs"]
    assert len(docs) == 1
    assert docs[0].page_content == "hello world"


def test_get_relevant_documents_raises_on_error_status() -> None:
    response = MagicMock(status_code=403)
    response.text = json.dumps({"error": {"message": "Access denied"}})

    with patch("requests.get", return_value=response):
        with pytest.raises(Exception, match="Error in search request"):
            make_retriever()._get_relevant_documents("cats", run_manager=MagicMock())


async def test_aget_relevant_documents_raises_on_error_status() -> None:
    """The sync path raises on a non-200 status; the async path must too."""
    session_cm, _ = make_async_session(403, {"error": {"message": "Access denied"}})

    with patch("aiohttp.ClientSession", return_value=session_cm):
        with pytest.raises(Exception, match="Error in search request"):
            await make_retriever()._aget_relevant_documents(
                "cats", run_manager=MagicMock()
            )
