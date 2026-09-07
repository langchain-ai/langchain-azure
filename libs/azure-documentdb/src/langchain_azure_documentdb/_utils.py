"""Vector math utilities for Azure DocumentDB integrations."""

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def cosine_similarity(left: Matrix, right: Matrix) -> np.ndarray:
    """Return row-wise cosine similarity for two equal-width matrices."""
    if len(left) == 0 or len(right) == 0:
        return np.array([])

    left_array = np.array(left)
    right_array = np.array(right)
    if left_array.shape[1] != right_array.shape[1]:
        raise ValueError(
            "Number of columns in left and right must be the same. "
            f"left has shape {left_array.shape} and right has shape "
            f"{right_array.shape}."
        )
    try:
        import simsimd as simd

        left_array = np.array(left_array, dtype=np.float32)
        right_array = np.array(right_array, dtype=np.float32)
        return 1 - np.array(simd.cdist(left_array, right_array, metric="cosine"))
    except ImportError:
        logger.debug(
            "Unable to import simsimd; using the NumPy cosine implementation."
        )
        left_norm = np.linalg.norm(left_array, axis=1)
        right_norm = np.linalg.norm(right_array, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(left_array, right_array.T) / np.outer(
                left_norm, right_norm
            )
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    indexes = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(indexes) < min(k, len(embedding_list)):
        best_score = -np.inf
        index_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for index, query_score in enumerate(similarity_to_query):
            if index in indexes:
                continue
            redundant_score = max(similarity_to_selected[index])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                index_to_add = index
        indexes.append(index_to_add)
        selected = np.append(selected, [embedding_list[index_to_add]], axis=0)
    return indexes