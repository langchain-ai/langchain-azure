"""Unit tests for `langchain_azure_ai.utils.math`."""

import numpy as np
import pytest

from langchain_azure_ai.utils.math import cosine_similarity_top_k


def test_top_k_returns_nothing_when_no_candidate_clears_the_threshold() -> None:
    """No row should clear a threshold above every achievable similarity.

    `X` and `Y` only contain orthogonal/opposite vectors, so every cosine
    similarity is <= 0. Asking for results above 0.5 must come back empty,
    not fall back to returning the whole candidate set.
    """
    x = [[1.0, 0.0]]
    y = [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]

    indices, scores = cosine_similarity_top_k(x, y, top_k=5, score_threshold=0.5)

    assert indices == []
    assert scores == []


def test_top_k_zero_returns_nothing() -> None:
    """An explicit `top_k=0` must return zero results, not every candidate.

    `0` is falsy, so a naive `top_k or default` would mistake it for "not
    provided" and fall back to ranking every candidate instead.
    """
    x = [[1.0, 0.0]]
    y = [[1.0, 0.0], [0.0, 1.0]]

    indices, scores = cosine_similarity_top_k(x, y, top_k=0)

    assert indices == []
    assert scores == []


def test_score_threshold_zero_excludes_negative_similarities() -> None:
    """An explicit `score_threshold=0.0` must still filter out negatives.

    `0.0` is falsy, so a naive `score_threshold or default` would mistake it
    for "not provided" and fall back to accepting everything above -1.0.
    """
    x = [[1.0, 0.0]]
    y = [
        [-0.5, np.sqrt(3) / 2],  # similarity -0.5, must be excluded
        [0.3, np.sqrt(1 - 0.3**2)],  # similarity 0.3, must be kept
    ]

    indices, scores = cosine_similarity_top_k(x, y, top_k=5, score_threshold=0.0)

    assert indices == [(0, 1)]
    assert scores == pytest.approx([0.3])


def test_top_k_still_returns_matches_that_clear_the_threshold() -> None:
    """The ordinary path -- some rows pass the threshold -- is unaffected."""
    x = [[1.0, 0.0]]
    y = [[1.0, 0.0], [0.0, 1.0]]

    indices, scores = cosine_similarity_top_k(x, y, top_k=5, score_threshold=0.5)

    assert indices == [(0, 0)]
    assert scores == pytest.approx([1.0])
