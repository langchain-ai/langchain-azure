"""Shared utilities and models for asynchronous and synchronous operations."""

import asyncio
import base64
import json
import sys
import threading
from abc import abstractmethod
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Annotated, Any, Generic, TypeVar

# typing.Self is introduced in Python 3.11
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# typing.override is introduced in Python 3.12
if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from azure.core.credentials import AccessToken
from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

R = TypeVar("R")
SP = TypeVar("SP", bound="SearchParams")
TOKEN_CREDENTIAL_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"


class SSLMode(str, Enum):
    """SSL mode for Azure Database for PostgreSQL connections."""

    disable = "disable"
    allow = "allow"
    prefer = "prefer"
    require = "require"
    verify_ca = "verify-ca"
    verify_full = "verify-full"


class BasicAuth(BaseModel):
    """Basic username/password authentication for Azure Database for PostgreSQL connections.

    :param username: Username for the connection.
    :type username: str
    :param password: Password for the connection.
    :type password: str
    """

    username: str = "postgres"
    password: str = ""


class BaseConnectionInfo(BaseModel):
    """Base connection information for Azure Database for PostgreSQL connections.

    :param application_name: Name of the application connecting to the database.
    :type application_name: str
    :param host: Hostname of the Azure Database for PostgreSQL server.
    :type host: str | None
    :param dbname: Name of the database to connect to.
    :type dbname: str
    :param port: Port number for the connection.
    :type port: int
    :param sslmode: SSL mode for the connection.
    :type sslmode: SSLMode
    """

    application_name: str = "azure-postgresql"
    host: str | None = None
    dbname: str = "postgres"
    port: Annotated[NonNegativeInt, Field(le=65535)] = 5432
    sslmode: SSLMode = SSLMode.require


def _run_coroutine_in_sync(coroutine: Coroutine[Any, Any, R]) -> R:
    """Run an async coroutine from synchronous code and return its result.

    This helper safely executes a coroutine in the appropriate event loop
    context:

    - If no event loop is running, it uses ``asyncio.run``.
    - If running on the main thread with an active loop, it offloads execution
      to a new loop in a thread to avoid nested loop errors.
    - If running on a non-main thread, it schedules the coroutine on the
      current loop using ``run_coroutine_threadsafe``.

    :param coroutine: The coroutine to execute.
    :type coroutine: Coroutine[Any, Any, R]
    :return: The coroutine's result.
    :rtype: R
    """

    def run_in_new_loop() -> R:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        result = asyncio.run(coroutine)
    else:
        if threading.current_thread() is threading.main_thread():
            if not loop.is_running():
                result = loop.run_until_complete(coroutine)
            else:
                with ThreadPoolExecutor() as pool:
                    future = pool.submit(run_in_new_loop)
                    result = future.result()
        else:
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            result = future.result()

    return result


def get_username_password(
    credentials: BasicAuth | AccessToken,
) -> tuple[str, str]:
    """Get username and password from credentials.

    :param credentials: BasicAuth for username/password or AccessToken for JWT token.
    :type credentials: BasicAuth | AccessToken
    :raises ValueError: User name not found in JWT token header
    :raises TypeError: Invalid credentials type
    :return: Tuple of username and password strings (plaintext).
    :rtype: tuple[str, str]
    """
    if isinstance(credentials, BasicAuth):
        return credentials.username, credentials.password
    elif isinstance(credentials, AccessToken):
        token = credentials.token
        _header, body_, _signature = token.split(".")
        body = json.loads(
            base64.b64decode(body_ + "=" * (4 - len(body_) % 4)).decode("utf-8")
        )
        username: str | None = body.get("upn", body.get("unique_name"))
        if username is None:
            raise ValueError("User name not found in JWT token header")
        return username, token
    else:
        raise TypeError(
            f"Invalid credentials type: {type(credentials)}. "
            "Expected BasicAuth or TokenCredential."
        )


class VectorOpClass(str, Enum):
    """Enumeration for operator classes used in vector indexes."""

    # Full-precision dense vector operator classes
    vector_cosine_ops = "vector_cosine_ops"
    vector_ip_ops = "vector_ip_ops"
    vector_l1_ops = "vector_l1_ops"
    vector_l2_ops = "vector_l2_ops"
    # Half-precision dense vector operator classes
    halfvec_cosine_ops = "halfvec_cosine_ops"
    halfvec_ip_ops = "halfvec_ip_ops"
    halfvec_l1_ops = "halfvec_l1_ops"
    halfvec_l2_ops = "halfvec_l2_ops"
    # Sparse vector operator classes
    sparsevec_cosine_ops = "sparsevec_cosine_ops"
    sparsevec_ip_ops = "sparsevec_ip_ops"
    sparsevec_l1_ops = "sparsevec_l1_ops"
    sparsevec_l2_ops = "sparsevec_l2_ops"
    # Bit vector operator classes
    bit_hamming_ops = "bit_hamming_ops"
    bit_jaccard_ops = "bit_jaccard_ops"

    def to_operator(self) -> str:
        """Return the distance operator as a string.

        :return: The distance operator string.
        :rtype: str
        :raises ValueError: If the vector operator class is unsupported.
        """
        match self:
            case (
                VectorOpClass.vector_cosine_ops
                | VectorOpClass.halfvec_cosine_ops
                | VectorOpClass.sparsevec_cosine_ops
            ):
                return "<=>"
            case (
                VectorOpClass.vector_ip_ops
                | VectorOpClass.halfvec_ip_ops
                | VectorOpClass.sparsevec_ip_ops
            ):
                return "<#>"
            case (
                VectorOpClass.vector_l1_ops
                | VectorOpClass.halfvec_l1_ops
                | VectorOpClass.sparsevec_l1_ops
            ):
                return "<+>"
            case (
                VectorOpClass.vector_l2_ops
                | VectorOpClass.halfvec_l2_ops
                | VectorOpClass.sparsevec_l2_ops
            ):
                return "<->"
            case VectorOpClass.bit_hamming_ops:
                return "<~>"
            case VectorOpClass.bit_jaccard_ops:
                return "<%>"
            case _:
                raise ValueError(f"Unsupported vector operator class: {self}")


class VectorType(str, Enum):
    """Enumeration for vector types used in vector similarity search."""

    bit = "bit"
    halfvec = "halfvec"
    sparsevec = "sparsevec"
    vector = "vector"


class Algorithm(BaseModel, Generic[SP]):
    """Base class for vector index algorithms and their settings.

    Subclasses provide index build-time settings via :meth:`build_settings` and
    the default search-time settings via :meth:`default_search_params`.

    The generic type parameter ``SP`` is a :class:`SearchParams` subtype that
    models the search-time parameters for the algorithm.

    :param op_class: The operator class to use for the vector index.
    :type op_class: VectorOpClass
    :param maintenance_work_mem: The amount of memory to use for maintenance operations.
    :type maintenance_work_mem: str | None
    :param max_parallel_maintenance_workers: The maximum number of parallel workers
                                             for maintenance operations.
    :type max_parallel_maintenance_workers: NonNegativeInt | None
    :param max_parallel_workers: The maximum number of parallel workers for query
                                 execution.
    :type max_parallel_workers: NonNegativeInt | None
    """

    op_class: VectorOpClass = VectorOpClass.vector_cosine_ops
    maintenance_work_mem: str | None = None
    max_parallel_maintenance_workers: Annotated[
        NonNegativeInt | None, Field(le=1_024)
    ] = None
    max_parallel_workers: Annotated[NonNegativeInt | None, Field(le=1_024)] = None

    @abstractmethod
    def default_search_params(self) -> SP:
        """Return the default search parameters for the algorithm.

        :return: An instance of the search parameters model.
        :rtype: SP
        """
        ...

    @abstractmethod
    def build_settings(self, exclude_none: bool = True) -> dict[str, Any]:
        """Return the specific index build settings for the algorithm.

        :param exclude_none: Whether to exclude keys with None values in the dictionary.
        :type exclude_none: bool
        :return: A dictionary containing the settings.
        :rtype: dict[str, Any]
        """
        ...

    def index_settings(self, exclude_none: bool = True) -> dict[str, Any]:
        """Return the general index settings for the algorithm.

        :param exclude_none: Whether to exclude keys with None values in the dictionary.
        :type exclude_none: bool
        :return: A dictionary containing the index settings.
        :rtype: dict[str, Any]
        """
        return {
            key: value
            for key, value in self.model_dump(
                mode="json", exclude_none=exclude_none
            ).items()
            if key
            in [
                "maintenance_work_mem",
                "max_parallel_maintenance_workers",
                "max_parallel_workers",
            ]
        }


class SearchParams(BaseModel):
    """Base model for vector index search parameters.

    Subclasses must implement :meth:`search_settings` to expose the parameters
    with the proper prefix expected by the underlying index implementation.
    """

    @abstractmethod
    def search_settings(self, exclude_none: bool = True) -> dict[str, Any]:
        """Return the specific index search settings for the algorithm.

        :param exclude_none: Whether to exclude keys with None values in the dictionary.
        :type exclude_none: bool
        :return: A dictionary containing the search settings.
        :rtype: dict[str, Any]
        """
        ...


class DiskANNIterativeScanMode(str, Enum):
    """Enumeration for DiskANN iterative scan modes."""

    off = "off"
    relaxed = "relaxed_order"
    strict = "strict_order"


class DiskANNSearchParams(SearchParams):
    """Search-time parameters for DiskANN indexes.

    All settings are exported with the ``diskann.`` prefix when used in SQL.

    :param l_value_is: The value of the L parameter for DiskANN index searching.
    :type l_value_is: PositiveInt | None
    :param iterative_search: The iterative search mode for DiskANN index searching.
    :type iterative_search: DiskANNIterativeScanMode | None
    """

    l_value_is: Annotated[PositiveInt | None, Field(ge=10, le=10_000)] = None
    iterative_search: DiskANNIterativeScanMode | None = None

    @override
    def search_settings(self, exclude_none=True):
        return {
            f"diskann.{key}": value
            for key, value in self.model_dump(
                mode="json", exclude_none=exclude_none
            ).items()
        }


class DiskANN(Algorithm[DiskANNSearchParams]):
    """DiskANN algorithm settings.

    Provides build-time and (via :class:`DiskANNSearchParams`) search-time
    parameters for DiskANN vector indexes.

    :param max_neighbors: The maximum number of edges per node in the graph.
    :type max_neighbors: PositiveInt | None
    :param l_value_ib: The value of the L parameter for DiskANN index building.
    :type l_value_ib: PositiveInt | None
    :param product_quantized: Whether to use product quantization (PQ) for the index.
    :type product_quantized: bool | None
    :param pq_param_num_chunks: Number of chunks for product quantization (PQ).
    :type pq_param_num_chunks: NonNegativeInt | None
    :param pq_param_training_samples: Number of training samples for product quantization (PQ).
    :type pq_param_training_samples: NonNegativeInt | None

    Notes:
    -----
    If ``product_quantized`` is ``True``, ``pq_param_num_chunks`` and
    ``pq_param_training_samples`` can be provided. Otherwise, these parameters
    are invalid and raise a ``ValueError`` during validation.
    """

    max_neighbors: Annotated[PositiveInt | None, Field(ge=20, le=1_538)] = None
    l_value_ib: Annotated[PositiveInt | None, Field(ge=10, le=500)] = None
    product_quantized: bool | None = None
    pq_param_num_chunks: Annotated[NonNegativeInt | None, Field(le=8_000)] = None
    pq_param_training_samples: Annotated[NonNegativeInt | None, Field(le=1_000_000)] = (
        None
    )

    @model_validator(mode="after")
    def sanity_check(self) -> Self:
        if not self.product_quantized and self.pq_param_num_chunks is not None:
            raise ValueError(
                "Parameter 'product_quantized' must be True when 'pq_param_num_chunks' is set."
            )
        if not self.product_quantized and self.pq_param_training_samples is not None:
            raise ValueError(
                "Parameter 'product_quantized' must be True when 'pq_param_training_samples' is set."
            )
        return self

    @override
    def build_settings(self, exclude_none: bool = True) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.model_dump(
                mode="json", exclude_none=exclude_none
            ).items()
            if key
            in [
                "max_neighbors",
                "l_value_ib",
                "product_quantized",
                "pq_param_num_chunks",
                "pq_param_training_samples",
            ]
        }

    @override
    def default_search_params(self) -> DiskANNSearchParams:
        return DiskANNSearchParams()


class HNSWIterativeScanMode(str, Enum):
    """Enumeration for HNSW iterative scan modes."""

    off = "off"
    relaxed = "relaxed_order"
    strict = "strict_order"


class HNSWSearchParams(SearchParams):
    """Search-time parameters for HNSW indexes.

    All settings are exported with the ``hnsw.`` prefix when used in SQL.

    :param ef_search: Size of the dynamic candidate list for HNSW index searching.
    :type ef_search: PositiveInt | None
    :param iterative_scan: The iterative search mode for HNSW index searching.
    :type iterative_scan: HNSWIterativeScanMode | None
    :param max_scan_tuples: The maximum number of tuples to visit during HNSW index
                            searching.
    :type max_scan_tuples: PositiveInt | None
    :param scan_mem_multiplier: The maximum amount of memory to use, as a multiple
                                of ``work_mem``, during HNSW index searching.
    :type scan_mem_multiplier: PositiveFloat | None
    """

    ef_search: Annotated[PositiveInt | None, Field(le=1_000)] = None
    iterative_scan: HNSWIterativeScanMode | None = None
    max_scan_tuples: PositiveInt | None = None
    scan_mem_multiplier: Annotated[PositiveFloat | None, Field(le=1_000)] = None

    @override
    def search_settings(self, exclude_none=True):
        return {
            f"hnsw.{key}": value
            for key, value in self.model_dump(
                mode="json", exclude_none=exclude_none
            ).items()
        }


class HNSW(Algorithm[HNSWSearchParams]):
    """HNSW algorithm settings.

    Provides build-time and (via :class:`HNSWSearchParams`) search-time
    parameters for HNSW vector indexes.

    :param m: The maximum number of connections per layer for HNSW index building.
    :type m: PositiveInt | None
    :param ef_construction: The size of the dynamic candidate list for constructing
                            the HNSW graph.
    :type ef_construction: PositiveInt | None

    Notes:
    -----
    If ``ef_construction`` is not at least twice the value of ``m``, a
    ``ValueError`` will be raised during validation.
    """

    m: Annotated[PositiveInt | None, Field(ge=2, le=100)] = None
    ef_construction: Annotated[PositiveInt | None, Field(ge=4, le=1_000)] = None

    @model_validator(mode="after")
    def sanity_check(self) -> Self:
        if (
            self.m is not None
            and self.ef_construction is not None
            and self.ef_construction < 2 * self.m
        ):
            raise ValueError(
                "Parameter 'ef_construction' must be at least twice the value of 'm'."
            )
        return self

    @override
    def build_settings(self, exclude_none=True):
        return {
            key: value
            for key, value in self.model_dump(
                mode="json", exclude_none=exclude_none
            ).items()
            if key in ["m", "ef_construction"]
        }

    @override
    def default_search_params(self) -> HNSWSearchParams:
        return HNSWSearchParams()


class IVFFlatIterativeScanMode(str, Enum):
    """Enumeration for IVFFlat iterative scan modes."""

    off = "off"
    relaxed = "relaxed_order"


class IVFFlatSearchParams(SearchParams):
    """Search-time parameters for IVF-Flat indexes.

    All settings are exported with the ``ivfflat.`` prefix when used in SQL.

    :param probes: The number of probes to use during IVF-Flat index searching.
    :type probes: PositiveInt | None
    :param iterative_scan: The iterative search mode for IVF-Flat index searching.
    :type iterative_scan: IVFFlatIterativeScanMode | None
    :param max_probes: The maximum number of probes to use during IVF-Flat index
                       searching.
    :type max_probes: PositiveInt | None
    """

    probes: Annotated[PositiveInt | None, Field(le=32_768)] = None
    iterative_scan: IVFFlatIterativeScanMode | None = None
    max_probes: Annotated[PositiveInt | None, Field(le=32_768)] = None

    @override
    def search_settings(self, exclude_none=True):
        return {
            f"ivfflat.{key}": value
            for key, value in self.model_dump(
                mode="json", exclude_none=exclude_none
            ).items()
        }


class IVFFlat(Algorithm[IVFFlatSearchParams]):
    """IVF-Flat algorithm settings.

    Provides build-time and (via :class:`IVFFlatSearchParams`) search-time
    parameters for IVF-Flat vector indexes.

    :param lists: The number of inverted lists to use for IVF-Flat indexing.
    :type lists: PositiveInt | None
    """

    lists: Annotated[PositiveInt | None, Field(le=32_768)] = None

    @override
    def build_settings(self, exclude_none: bool = True) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.model_dump(
                mode="json", exclude_none=exclude_none
            ).items()
            if key in ["lists"]
        }

    @override
    def default_search_params(self) -> IVFFlatSearchParams:
        return IVFFlatSearchParams()


class Extension(BaseModel):
    """Model representing a PostgreSQL extension.

    :param ext_name: Name of the extension to be created, checked or dropped.
    :type ext_name: str
    :param ext_version: Optional version of the extension to be created or checked.
    :type ext_version: str | None
    :param schema_name: Optional schema name where the extension should be created
                        or checked.
    :type schema_name: str | None
    :param cascade: Whether to automatically install the extension dependencies or
                    drop the objects that depend on the extension.
    :type cascade: bool
    """

    ext_name: str
    ext_version: str | None = None
    schema_name: str | None = None
    cascade: bool = False
