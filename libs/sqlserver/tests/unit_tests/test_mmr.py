"""Unit test for maximal marginal relevance result ordering (no live DB).

``max_marginal_relevance_search_by_vector`` must return documents in the order
chosen by the MMR algorithm, not in the store's original fetch order.
"""

import json
from types import SimpleNamespace
from typing import Any

from langchain_sqlserver.vectorstores import SQLServerVectorStore


class _FakeResult:
    """Stands in for a SQLAlchemy Row from ``_search_store``.

    Supports ``result[0]`` (the JSON-encoded embedding) and the
    ``EmbeddingStore``/``distance`` attributes used downstream.
    """

    def __init__(self, embedding: list[float], content: str) -> None:
        self._embedding_json = json.dumps(embedding)
        self.EmbeddingStore = SimpleNamespace(content=content, content_metadata={})
        self.distance = 0.0

    def __getitem__(self, index: int) -> Any:
        return self._embedding_json


def test_mmr_preserves_selection_order() -> None:
    """The returned docs must follow the MMR ranking. With query [1, 0] the
    closest doc is 'C' ([1, 0]); MMR then picks 'A' over 'B'. The old code
    filtered by original index and returned ['A', 'C'] instead of ['C', 'A'].

    ``lambda_mult`` is deliberately not 0.5: at 0.5 the relevance and
    redundancy terms cancel for both remaining candidates (each scores 0), so
    the second pick would depend on tie-breaking inside
    ``maximal_marginal_relevance`` rather than on the ranking itself.
    """
    store = SQLServerVectorStore.__new__(SQLServerVectorStore)
    results = [
        _FakeResult([0.9, 0.1], "A"),
        _FakeResult([0.0, 1.0], "B"),
        _FakeResult([1.0, 0.0], "C"),
    ]
    store._search_store = lambda *args, **kwargs: results  # type: ignore[method-assign]

    docs = store.max_marginal_relevance_search_by_vector(
        [1.0, 0.0], k=2, fetch_k=3, lambda_mult=0.7
    )

    assert [d.page_content for d in docs] == ["C", "A"]
