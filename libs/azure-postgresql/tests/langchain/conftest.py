"""Pytest fixtures for LangChain PostgreSQL vectorstore tests.

This module provides async and sync table fixtures plus helpers used by
tests under the LangChain integration folder.
"""

import uuid
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any

import numpy as np
import pytest
import scipy.optimize  # type: ignore[import-untyped]
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from numpy.random import PCG64, SeedSequence
from numpy.random import Generator as RNG
from psycopg import sql
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from pydantic import BaseModel, PositiveInt

from langchain_azure_postgresql.common import Algorithm, VectorType
from langchain_azure_postgresql.langchain import (
    AsyncAzurePGVectorStore,
    AzurePGVectorStore,
)

_FIXTURE_PARAMS_TABLE: dict[str, Any] = {
    "scope": "class",
    "params": [
        {
            "existing": False,
            "table_name": "non_existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": VectorType.vector,
            "embedding_dimension": 1_536,
            "embedding_index": None,
            "metadata_columns": "metadata_column",
        },
        {
            "existing": False,
            "table_name": "non_existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": VectorType.vector,
            "embedding_dimension": 1_536,
            "embedding_index": None,
            "metadata_columns": ["metadata_column1", "metadata_column2"],
        },
        {
            "existing": False,
            "table_name": "non_existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": VectorType.vector,
            "embedding_dimension": 1_536,
            "embedding_index": None,
            "metadata_columns": [
                ("metadata_column1", "text"),
                ("metadata_column2", "double precision"),
            ],
        },
        {
            "existing": True,
            "table_name": "existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": VectorType.vector,
            "embedding_dimension": 1_536,
            "embedding_index": None,
            "metadata_columns": "metadata_column",
        },
        {
            "existing": True,
            "table_name": "existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": VectorType.vector,
            "embedding_dimension": 1_536,
            "embedding_index": None,
            "metadata_columns": ["metadata_column1", "metadata_column2"],
        },
        {
            "existing": True,
            "table_name": "existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": VectorType.vector,
            "embedding_dimension": 1_536,
            "embedding_index": None,
            "metadata_columns": [
                ("metadata_column1", "text"),
                ("metadata_column2", "double precision"),
            ],
        },
    ],
    "ids": [
        "non-existing-table-metadata-str",
        "non-existing-table-metadata-list",
        "non-existing-table-metadata-list-tuple",
        "existing-table-metadata-str",
        "existing-table-metadata-list",
        "existing-table-metadata-list-tuple",
    ],
}


class MockEmbedding(Embeddings):
    def __init__(
        self,
        dimension: int,
        target_similarity: np.ndarray | None = None,
        *,
        seed: int = 42,
    ):
        """A mock implementation of the Embeddings class to be used for similarity tests.

        :param dimension: The number of features/dimensions in an embedding vector
        :type dimension: int
        :param target_similarity: The target cosine similarity matrix
        :type target_similarity: np.ndarray | None
        :param seed: The random seed for reproducibility
        :type seed: int
        """
        # The user can provide the target cosine similarity matrix, or else
        # we default to the below matrix for our documents and texts fixtures.
        # There, we have the following:
        #   - kittens are _very_ close to cats,
        #   - tigers are closer to cats than to dogs,
        #   - animals is central among animals, and,
        #   - plants are far from all others.
        self._target_similarity = target_similarity or np.array(
            [
                # kittens, cats, tigers, dogs, animals, plants
                [1.0, 0.84, 0.377, 0.310, 0.510, -0.191],  # kittens
                [0.84, 1.0, 0.59, 0.35, 0.56, -0.159],  # cats
                [0.377, 0.590, 1.000, 0.259, 0.454, -0.240],  # tigers
                [0.310, 0.350, 0.259, 1.000, 0.391, -0.191],  # dogs
                [0.510, 0.560, 0.454, 0.391, 1.000, -0.107],  # animals
                [-0.191, -0.159, -0.240, -0.191, -0.107, 1.000],  # plants
            ],
            dtype=np.float32,
        )
        assert self._target_similarity.shape[0] == self._target_similarity.shape[1], (
            "Target cosine similarity matrix must be square"
        )

        self._n = dimension  # number of features/dimenstions in an embedding vector
        self._m = self._target_similarity.shape[0]  # number of categories
        self._generator = RNG(PCG64(SeedSequence(seed)))

        self._embedding_vectors = self._optimize_similarity_matrix(
            self._n, self._m, self._target_similarity
        )[1]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if text.find("kittens") >= 0:  # return the embedding vector for kittens
            return self._embedding_vectors[:, 0].tolist()
        elif text.find("cats") >= 0:  # return the embedding vector for cats
            return self._embedding_vectors[:, 1].tolist()
        elif text.find("tigers") >= 0:  # return the embedding vector for tigers
            return self._embedding_vectors[:, 2].tolist()
        elif text.find("dogs") >= 0:  # return the embedding vector for dogs
            return self._embedding_vectors[:, 3].tolist()
        elif text.find("animals") >= 0:  # return the embedding vector for animals
            return self._embedding_vectors[:, 4].tolist()
        else:  # return the embedding vector for plants
            return self._embedding_vectors[:, 5].tolist()

    def _normalize_columns(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize columns of matrix to unit norm."""
        norms = np.linalg.norm(matrix, axis=0, keepdims=True)
        return matrix / norms

    def _compute_cosine_similarity(self, matrix: np.ndarray) -> np.ndarray:
        """Compute MxM cosine similarity matrix from NxM matrix with unit-norm columns."""
        return matrix.T @ matrix

    def _objective_and_gradient(
        self, x_flat: np.ndarray, target: np.ndarray, N: int, M: int
    ) -> tuple[float, np.ndarray]:
        """Compute objective (Frobenius norm squared) and gradient.

        :param x_flat: Flattened N*M parameter vector
        :type x_flat: np.ndarray
        :param target: Target MxM similarity matrix
        :type target: np.ndarray
        :param N: Feature dimension
        :type N: int
        :param M: Number of categories
        :type M: int
        :return: Objective value and gradient vector
        :rtype: tuple[float, np.ndarray]
        """
        # Reshape and normalize columns
        X = x_flat.reshape(N, M)
        X = self._normalize_columns(X)

        # Compute current similarity matrix
        S = X.T @ X

        # Compute objective (Frobenius norm squared)
        diff = S - target
        objective = np.sum(diff**2)

        # Compute gradient with tangent space projection
        grad_X = 2 * X @ diff
        grad_X_proj = np.zeros_like(grad_X)
        for j in range(M):
            col = X[:, j]
            grad_col = grad_X[:, j]
            # Project out component parallel to column (maintain unit norm)
            grad_X_proj[:, j] = grad_col - np.dot(grad_col, col) * col

        return objective, grad_X_proj.flatten()

    def _optimize_similarity_matrix(
        self,
        N: int,
        M: int,
        target_similarity: np.ndarray,
        max_iter: int = 1000,
    ) -> tuple[scipy.optimize.OptimizeResult, np.ndarray]:
        """Find NxM matrix that produces target MxM cosine similarity matrix.

        :param N: Feature dimension
        :type N: int
        :param M: Number of categories
        :type M: int
        :param target_similarity: MxM target cosine similarity matrix
        :type target_similarity: np.ndarray
        :param max_iter: Maximum optimization iterations
        :type max_iter: int
        :return: Optimization result object and optimized NxM matrix
        :rtype: tuple
        """
        # Initialize with random unit-norm columns
        X_init = self._normalize_columns(self._generator.standard_normal((N, M)))
        x_flat = X_init.flatten()

        # Optimize
        result = scipy.optimize.minimize(
            fun=lambda x: self._objective_and_gradient(x, target_similarity, N, M),
            x0=x_flat,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-8},
        )

        # Extract optimized matrix
        X_optimized = result.x.reshape(N, M)
        X_optimized = self._normalize_columns(X_optimized)

        return result, X_optimized


class MockUUID:
    def __init__(self, uuid4: Callable[[], uuid.UUID] = uuid.uuid4):
        self._generated_uuids: list[uuid.UUID] = []
        self._original_uuid4 = uuid4

    def __call__(self) -> uuid.UUID:
        generated_uuid = self._original_uuid4()
        self._generated_uuids.append(generated_uuid)
        return generated_uuid

    @property
    def generated_uuids(self) -> list[uuid.UUID]:
        return self._generated_uuids


def transform_metadata_columns(
    columns: list[str] | list[tuple[str, str]] | str,
) -> list[tuple[str, str]]:
    """Normalize metadata column definitions to a list of (name, type) tuples.

    :param columns: A single column name (string), a list of column names
                    (strings), or a list of (name, type) tuples.
    :type columns: list[str] | list[tuple[str, str]] | str
    :return: A list of (column_name, column_type) tuples. Strings are mapped to
             "text", except a single-string input which maps to type "jsonb".
    :rtype: list[tuple[str, str]]
    """
    if isinstance(columns, str):
        return [(columns, "jsonb")]
    else:
        return [(col, "text") if isinstance(col, str) else col for col in columns]


class Table(BaseModel):
    """Table configuration for test parameterization.

    :param existing: Whether the table should be created before running a test.
    :param schema_name: Schema where the table resides.
    :param table_name: Name of the table.
    :param id_column: Primary key column name (uuid).
    :param content_column: Text content column name.
    :param embedding_column: Vector/embedding column name.
    :param embedding_type: Embedding type (e.g., "vector").
    :param embedding_dimension: Embedding dimension length.
    :param metadata_columns: List of metadata column names or (name, type) tuples.
    """

    existing: bool
    schema_name: str
    table_name: str
    id_column: str
    content_column: str
    embedding_column: str
    embedding_type: VectorType
    embedding_dimension: PositiveInt
    embedding_index: Algorithm | None
    metadata_columns: list[str] | list[tuple[str, str]] | str


@pytest.fixture(**_FIXTURE_PARAMS_TABLE)
async def async_table(
    async_connection_pool: AsyncConnectionPool,
    async_schema: str,
    request: pytest.FixtureRequest,
) -> AsyncGenerator[Table, Any]:
    """Fixture to provide a parametrized table configuration for asynchronous tests.

    This fixture yields a `Table` model with normalized metadata columns. When
    the parameter `existing` is `True`, it creates the table in the provided
    schema before yielding and drops it after the test class completes.

    :param async_connection_pool: The asynchronous connection pool to use for DDL.
    :type async_connection_pool: AsyncConnectionPool
    :param async_schema: The schema name where the table should be created.
    :type async_schema: str
    :param request: The pytest request object providing parametrization.
    :type request: pytest.FixtureRequest
    :return: An asynchronous generator yielding a `Table` configuration.
    :rtype: AsyncGenerator[Table, Any]
    """
    assert isinstance(request.param, dict), "Request param must be a dictionary"

    table = Table(
        existing=request.param.get("existing", None),
        schema_name=async_schema,
        table_name=request.param.get("table_name", "langchain"),
        id_column=request.param.get("id_column", "id_column"),
        content_column=request.param.get("content_column", "content_column"),
        embedding_column=request.param.get("embedding_column", "embedding_column"),
        embedding_type=request.param.get("embedding_type", "vector"),
        embedding_dimension=request.param.get("embedding_dimension", 1_536),
        embedding_index=request.param.get("embedding_index", None),
        metadata_columns=request.param.get("metadata_columns", "metadata_column"),
    )

    # Needed to make mypy happy during type checking
    table.metadata_columns = transform_metadata_columns(table.metadata_columns)

    if table.existing:
        async with async_connection_pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                sql.SQL(
                    """
                    create table {table_name} (
                        {id_column} uuid primary key,
                        {content_column} text,
                        {embedding_column} {embedding_type}({embedding_dimension}),
                        {metadata_columns}
                    )
                    """
                ).format(
                    table_name=sql.Identifier(async_schema, table.table_name),
                    id_column=sql.Identifier(table.id_column),
                    content_column=sql.Identifier(table.content_column),
                    embedding_column=sql.Identifier(table.embedding_column),
                    embedding_type=sql.Identifier(table.embedding_type),
                    embedding_dimension=sql.Literal(table.embedding_dimension),
                    metadata_columns=sql.SQL(", ").join(
                        sql.SQL("{col} {type}").format(
                            col=sql.Identifier(col), type=sql.SQL(type)
                        )
                        for col, type in table.metadata_columns
                    ),
                )
            )

    yield table

    async with async_connection_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            sql.SQL("drop table {table} cascade").format(
                table=sql.Identifier(async_schema, table.table_name)
            )
        )


@pytest.fixture(scope="class")
async def async_vectorstore(
    async_connection_pool: AsyncConnectionPool, async_table: Table
) -> AsyncAzurePGVectorStore:
    return AsyncAzurePGVectorStore(
        embedding=MockEmbedding(dimension=async_table.embedding_dimension),
        connection_pool=async_connection_pool,
        schema_name=async_table.schema_name,
        table_name=async_table.table_name,
        id_column=async_table.id_column,
        content_column=async_table.content_column,
        embedding_column=async_table.embedding_column,
        embedding_type=async_table.embedding_type,
        embedding_dimension=async_table.embedding_dimension,
        embedding_index=async_table.embedding_index,
        metadata_columns=async_table.metadata_columns,
    )


@pytest.fixture(
    params=[
        "documents-ids-success",
        "documents-no-ids-success",
        "documents-ids-overridden-success",
        "documents-ids-overridden-failure",
    ]
)
def documents_ids(
    request: pytest.FixtureRequest,
) -> tuple[list[Document], list[str] | None]:
    assert isinstance(request.param, str), "Expected request.param to be a string"
    assert request.param in [
        "documents-ids-success",
        "documents-no-ids-success",
        "documents-ids-overridden-success",
        "documents-ids-overridden-failure",
    ], "Expected request.param to be one of the predefined document scenarios."

    if request.param == "documents-ids-success":
        return (
            [
                Document(
                    id="00000000-0000-0000-0000-000000000001",
                    page_content="Document 1 about kittens",
                    metadata={"metadata_column1": "document1", "metadata_column2": 1.0},
                ),
                Document(
                    id="00000000-0000-0000-0000-000000000002",
                    page_content="Document 2 about cats",
                    metadata={"metadata_column1": "document2", "metadata_column2": 2.0},
                ),
                Document(
                    id="00000000-0000-0000-0000-000000000003",
                    page_content="Document 3 about dogs",
                    metadata={"metadata_column1": "document3", "metadata_column2": 3.0},
                ),
                Document(
                    id="00000000-0000-0000-0000-000000000004",
                    page_content="Document 4 about plants",
                    metadata={"metadata_column1": "document4", "metadata_column2": 4.0},
                ),
            ],
            None,
        )
    elif request.param == "documents-no-ids-success":
        return (
            [
                Document(
                    page_content="Document 1 about kittens",
                    metadata={"metadata_column1": "document1", "metadata_column2": 1.0},
                ),
                Document(
                    page_content="Document 2 about cats",
                    metadata={"metadata_column1": "document2", "metadata_column2": 2.0},
                ),
                Document(
                    page_content="Document 3 about dogs",
                    metadata={"metadata_column1": "document3", "metadata_column2": 3.0},
                ),
                Document(
                    page_content="Document 4 about plants",
                    metadata={"metadata_column1": "document4", "metadata_column2": 4.0},
                ),
            ],
            None,
        )
    elif request.param == "documents-ids-overridden-success":
        return (
            [
                Document(
                    id="00000000-0000-0000-0000-000000000001",
                    page_content="Document 1 about kittens",
                    metadata={"metadata_column1": "document1", "metadata_column2": 1.0},
                ),
                Document(
                    id="00000000-0000-0000-0000-000000000002",
                    page_content="Document 2 about cats",
                    metadata={"metadata_column1": "document2", "metadata_column2": 2.0},
                ),
                Document(
                    id="00000000-0000-0000-0000-000000000003",
                    page_content="Document 3 about dogs",
                    metadata={"metadata_column1": "document3", "metadata_column2": 3.0},
                ),
                Document(
                    id="00000000-0000-0000-0000-000000000004",
                    page_content="Document 4 about plants",
                    metadata={"metadata_column1": "document4", "metadata_column2": 4.0},
                ),
            ],
            [
                "00000000-0000-0000-0000-000000000005",
                "00000000-0000-0000-0000-000000000006",
                "00000000-0000-0000-0000-000000000007",
                "00000000-0000-0000-0000-000000000008",
            ],
        )
    else:  # documents-ids-overridden-failure
        return (
            [
                Document(
                    id="00000000-0000-0000-0000-000000000001",
                    page_content="Document 1 about kittens",
                    metadata={"metadata_column1": "document1", "metadata_column2": 1.0},
                ),
                Document(
                    id="00000000-0000-0000-0000-000000000002",
                    page_content="Document 2 about cats",
                    metadata={"metadata_column1": "document2", "metadata_column2": 2.0},
                ),
                Document(
                    id="00000000-0000-0000-0000-000000000003",
                    page_content="Document 3 about dogs",
                    metadata={"metadata_column1": "document3", "metadata_column2": 3.0},
                ),
                Document(
                    id="00000000-0000-0000-0000-000000000004",
                    page_content="Document 4 about plants",
                    metadata={"metadata_column1": "document4", "metadata_column2": 4.0},
                ),
            ],
            [
                "00000000-0000-0000-0000-000000000005",
                "00000000-0000-0000-0000-000000000006",
                "00000000-0000-0000-0000-000000000007",
            ],
        )


@pytest.fixture(
    params=[
        "texts-success",
        "texts-ids-success",
        "texts-metadatas-success",
        "texts-ids-metadatas-success",
        "texts-ids-failure",
        "texts-metadatas-failure",
    ],
)
def texts_ids_metadatas(
    request: pytest.FixtureRequest,
) -> tuple[list[str], list[str] | None, list[dict[str, Any]] | None]:
    assert isinstance(request.param, str), "Expected request.param to be a string"
    assert request.param in [
        "texts-success",
        "texts-ids-success",
        "texts-metadatas-success",
        "texts-ids-metadatas-success",
        "texts-ids-failure",
        "texts-metadatas-failure",
    ], "Expected request.param to be one of the predefined text scenarios."

    if request.param == "texts-success":
        return (
            [
                "Text 1 about cats",
                "Text 2 about tigers",
                "Text 3 about dogs",
                "Text 4 about plants",
            ],
            None,
            None,
        )
    elif request.param == "texts-ids-success":
        return (
            [
                "Text 1 about cats",
                "Text 2 about tigers",
                "Text 3 about dogs",
                "Text 4 about plants",
            ],
            [
                "00000000-0000-0000-0000-100000000001",
                "00000000-0000-0000-0000-100000000002",
                "00000000-0000-0000-0000-100000000003",
                "00000000-0000-0000-0000-100000000004",
            ],
            None,
        )
    elif request.param == "texts-metadatas-success":
        return (
            [
                "Text 1 about cats",
                "Text 2 about tigers",
                "Text 3 about dogs",
                "Text 4 about plants",
            ],
            None,
            [
                {"metadata_column1": "text1", "metadata_column2": 1.0},
                {"metadata_column1": "text2", "metadata_column2": 2.0},
                {"metadata_column1": "text3", "metadata_column2": 3.0},
                {"metadata_column1": "text4", "metadata_column2": 4.0},
            ],
        )
    elif request.param == "texts-ids-metadatas-success":
        return (
            [
                "Text 1 about cats",
                "Text 2 about tigers",
                "Text 3 about dogs",
                "Text 4 about plants",
            ],
            [
                "00000000-0000-0000-0000-100000000005",
                "00000000-0000-0000-0000-100000000006",
                "00000000-0000-0000-0000-100000000007",
                "00000000-0000-0000-0000-100000000008",
            ],
            [
                {"metadata_column1": "text1", "metadata_column2": 1.0},
                {"metadata_column1": "text2", "metadata_column2": 2.0},
                {"metadata_column1": "text3", "metadata_column2": 3.0},
                {"metadata_column1": "text4", "metadata_column2": 4.0},
            ],
        )
    elif request.param == "texts-ids-failure":
        return (
            [
                "Text 1 about cats",
                "Text 2 about tigers",
                "Text 3 about dogs",
                "Text 4 about plants",
            ],
            [
                "00000000-0000-0000-0000-100000000001",
                "00000000-0000-0000-0000-100000000002",
                "00000000-0000-0000-0000-100000000003",
            ],
            None,
        )
    else:  # texts-metadatas-failure
        return (
            [
                "Text 1 about cats",
                "Text 2 about tigers",
                "Text 3 about dogs",
                "Text 4 about plants",
            ],
            None,
            [
                {"metadata_column1": "text1", "metadata_column2": 1.0},
                {"metadata_column1": "text2", "metadata_column2": 2.0},
                {"metadata_column1": "text3", "metadata_column2": 3.0},
            ],
        )


@pytest.fixture
def mock_uuid(monkeypatch: pytest.MonkeyPatch) -> MockUUID:
    mock = MockUUID()
    monkeypatch.setattr(uuid, "uuid4", mock)
    return mock


@pytest.fixture(**_FIXTURE_PARAMS_TABLE)
def table(
    connection_pool: ConnectionPool,
    schema: str,
    request: pytest.FixtureRequest,
) -> Generator[Table, Any, None]:
    """Fixture to provide a parametrized table configuration for synchronous tests.

    This fixture yields a `Table` model with normalized metadata columns. When
    the parameter `existing` is `True`, it creates the table in the provided
    schema before yielding and drops it after the test class completes.

    :param connection_pool: The synchronous connection pool to use for DDL.
    :type connection_pool: ConnectionPool
    :param schema: The schema name where the table should be created.
    :type schema: str
    :param request: The pytest request object providing parametrization.
    :type request: pytest.FixtureRequest
    :return: A generator yielding a `Table` configuration.
    :rtype: Generator[Table, Any, None]
    """
    assert isinstance(request.param, dict), "Request param must be a dictionary"

    table = Table(
        existing=request.param.get("existing", None),
        schema_name=schema,
        table_name=request.param.get("table_name", "langchain"),
        id_column=request.param.get("id_column", "id_column"),
        content_column=request.param.get("content_column", "content_column"),
        embedding_column=request.param.get("embedding_column", "embedding_column"),
        embedding_type=request.param.get("embedding_type", "vector"),
        embedding_dimension=request.param.get("embedding_dimension", 1_536),
        embedding_index=request.param.get("embedding_index", None),
        metadata_columns=request.param.get("metadata_columns", "metadata_column"),
    )

    # Needed to make mypy happy during type checking
    table.metadata_columns = transform_metadata_columns(table.metadata_columns)

    if table.existing:
        with connection_pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    create table {table_name} (
                        {id_column} uuid primary key,
                        {content_column} text,
                        {embedding_column} {embedding_type}({embedding_dimension}),
                        {metadata_columns}
                    )
                    """
                ).format(
                    table_name=sql.Identifier(schema, table.table_name),
                    id_column=sql.Identifier(table.id_column),
                    content_column=sql.Identifier(table.content_column),
                    embedding_column=sql.Identifier(table.embedding_column),
                    embedding_type=sql.Identifier(table.embedding_type),
                    embedding_dimension=sql.Literal(table.embedding_dimension),
                    metadata_columns=sql.SQL(", ").join(
                        sql.SQL("{col} {type}").format(
                            col=sql.Identifier(col), type=sql.SQL(type)
                        )
                        for col, type in table.metadata_columns
                    ),
                )
            )

    yield table

    with connection_pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("drop table {table} cascade").format(
                table=sql.Identifier(schema, table.table_name)
            )
        )


@pytest.fixture(scope="class")
def vectorstore(connection_pool: ConnectionPool, table: Table) -> AzurePGVectorStore:
    return AzurePGVectorStore(
        embedding=MockEmbedding(dimension=table.embedding_dimension),
        connection_pool=connection_pool,
        schema_name=table.schema_name,
        table_name=table.table_name,
        id_column=table.id_column,
        content_column=table.content_column,
        embedding_column=table.embedding_column,
        embedding_type=table.embedding_type,
        embedding_dimension=table.embedding_dimension,
        embedding_index=table.embedding_index,
        metadata_columns=table.metadata_columns,
    )
