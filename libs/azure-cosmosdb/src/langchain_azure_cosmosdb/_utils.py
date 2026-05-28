"""Utility functions for working with vectors and vectorstores."""

import logging
from enum import Enum
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from azure.cosmos import PartitionKey
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def extract_partition_key_paths(pk_def: Any) -> List[str]:
    """Extract partition key paths from a partition key definition.

    Supports:

    * ``azure.cosmos.PartitionKey`` instances with a single-path ``Hash`` /
      ``MultiHash``-of-one configuration. Hierarchical partition keys (HPK,
      multi-path) are explicitly rejected because the batch insertion code
      groups items by a single scalar partition-key value and would silently
      produce wrong groups for HPK.
    * Plain dict definitions of the form ``{"paths": [...], "kind": ...}``
      following the same constraints.

    Args:
        pk_def: The partition key definition stored in
            ``container_properties["partition_key"]``.

    Returns:
        A list with exactly one path string (e.g. ``["/id"]``).

    Raises:
        TypeError: ``pk_def`` is neither a ``PartitionKey`` nor a dict.
        ValueError: The definition declares a hierarchical / multi-path
            partition key, which is not currently supported by the
            transactional batch insertion path.
    """
    path: Any
    if isinstance(pk_def, PartitionKey):
        path = pk_def.path
    elif isinstance(pk_def, dict):
        path = pk_def.get("paths") or pk_def.get("path")
        if path is None:
            raise ValueError("Partition key dict must contain a 'paths' or 'path' key.")
    elif isinstance(pk_def, str):
        path = pk_def
    elif hasattr(pk_def, "path"):
        # Duck-type for ``PartitionKey``-like objects (e.g. test mocks).
        path = pk_def.path
    else:
        raise TypeError(
            "partition_key must be an azure.cosmos.PartitionKey instance, a "
            f"dict, or a path string; got {type(pk_def).__name__}."
        )

    if isinstance(path, str):
        paths = [path]
    elif isinstance(path, (list, tuple)):
        paths = list(path)
    else:
        raise TypeError(f"Unsupported partition key path type: {type(path).__name__}.")

    if len(paths) != 1:
        raise ValueError(
            "Hierarchical partition keys (multiple paths) are not supported by "
            "the transactional batch insertion path; got paths="
            f"{paths!r}."
        )
    return paths


def extract_partition_key_value(item: Dict[str, Any], pk_paths: List[str]) -> Any:
    """Extract the partition-key value for ``item`` given its key paths.

    Args:
        item: The document to extract the value from.
        pk_paths: Key paths from :func:`extract_partition_key_paths` (e.g.
            ``["/tenant/id"]``).

    Returns:
        The scalar partition-key value.
    """
    parts = [p for p in pk_paths[0].split("/") if p]
    value: Any = item
    for part in parts:
        try:
            value = value[part]
        except KeyError as exc:
            raise ValueError(
                f"Partition key path '/{'/'.join(parts)}' not found in document. "
                f"Missing key '{part}'. Document keys: {list(item.keys())}"
            ) from exc
        except TypeError as exc:
            raise ValueError(
                f"Partition key path '/{'/'.join(parts)}' encountered a "
                f"non-object value at '{part}'. "
                f"Document keys: {list(item.keys())}"
            ) from exc
    return value


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - np.array(simd.cdist(X, Y, metric="cosine"))
        return Z
    except ImportError:
        logger.debug(
            "Unable to import simsimd, defaulting to NumPy implementation. If you want "
            "to use simsimd please install with `pip install simsimd`."
        )
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"


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
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


def filter_complex_metadata(
    documents: List[Document],
    *,
    allowed_types: Tuple[Type, ...] = (str, bool, int, float),
) -> List[Document]:
    """Filter out metadata types that are not supported for a vector store."""
    updated_documents = []
    for document in documents:
        filtered_metadata = {
            key: value
            for key, value in document.metadata.items()
            if isinstance(value, allowed_types)
        }

        updated_documents.append(
            document.model_copy(update={"metadata": filtered_metadata})
        )

    return updated_documents
