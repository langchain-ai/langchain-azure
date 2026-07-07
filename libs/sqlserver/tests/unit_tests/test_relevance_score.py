"""Unit tests for relevance-score-function selection (no live DB).

``_select_relevance_score_fn`` must honour every distance strategy that the
``distance_strategy`` property accepts. That property normalizes string inputs
case-insensitively, so a store built with ``distance_strategy="COSINE"`` is
valid; the selector must map it to the cosine scoring function rather than
raising.
"""

import pytest

from langchain_sqlserver.vectorstores import DistanceStrategy, SQLServerVectorStore


def _make_store(strategy: object) -> SQLServerVectorStore:
    store = SQLServerVectorStore.__new__(SQLServerVectorStore)
    store.override_relevance_score_fn = None
    store._distance_strategy = strategy  # type: ignore[assignment]
    return store


@pytest.mark.parametrize(
    "strategy, expected",
    [
        (DistanceStrategy.COSINE, "_cosine_relevance_score_fn"),
        (DistanceStrategy.DOT, "_max_inner_product_relevance_score_fn"),
        (DistanceStrategy.EUCLIDEAN, "_euclidean_relevance_score_fn"),
        ("cosine", "_cosine_relevance_score_fn"),
        ("dot", "_max_inner_product_relevance_score_fn"),
        ("euclidean", "_euclidean_relevance_score_fn"),
        # Mixed / upper case strings are accepted by the `distance_strategy`
        # property, so the selector must accept them too.
        ("COSINE", "_cosine_relevance_score_fn"),
        ("Dot", "_max_inner_product_relevance_score_fn"),
        ("Euclidean", "_euclidean_relevance_score_fn"),
    ],
)
def test_select_relevance_score_fn(strategy: object, expected: str) -> None:
    store = _make_store(strategy)
    assert store._select_relevance_score_fn().__name__ == expected


def test_select_relevance_score_fn_prefers_override() -> None:
    store = SQLServerVectorStore.__new__(SQLServerVectorStore)
    override = lambda distance: distance  # noqa: E731
    store.override_relevance_score_fn = override
    store._distance_strategy = "unsupported"  # type: ignore[assignment]
    assert store._select_relevance_score_fn() is override


def test_select_relevance_score_fn_rejects_unknown_strategy() -> None:
    store = _make_store("manhattan")
    with pytest.raises(ValueError):
        store._select_relevance_score_fn()
