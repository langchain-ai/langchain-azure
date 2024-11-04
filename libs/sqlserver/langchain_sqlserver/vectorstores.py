"""This is the SQL Server module.

This module provides the SQLServer_VectorStore class for managing
vectorstores in SQL Server.
"""

from __future__ import annotations

import json
import logging
import re
import struct
import uuid
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
from urllib.parse import urlparse

import numpy as np
import sqlalchemy
from azure.identity import DefaultAzureCredential
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from sqlalchemy import (
    Column,
    ColumnElement,
    Dialect,
    Index,
    Numeric,
    PrimaryKeyConstraint,
    SQLColumnExpression,
    Uuid,
    asc,
    bindparam,
    cast,
    create_engine,
    event,
    func,
    insert,
    label,
    select,
    text,
)
from sqlalchemy.dialects.mssql import JSON, NVARCHAR, VARCHAR
from sqlalchemy.dialects.mssql.base import MSTypeCompiler
from sqlalchemy.engine import URL, Connection, Engine
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy.pool import ConnectionPoolEntry
from sqlalchemy.sql import operators
from sqlalchemy.types import UserDefinedType

COMPARISONS_TO_NATIVE: Dict[str, Callable[[ColumnElement, object], ColumnElement]] = {
    "$eq": operators.eq,
    "$ne": operators.ne,
}

NUMERIC_OPERATORS: Dict[str, Callable[[ColumnElement, object], ColumnElement]] = {
    "$lt": operators.lt,
    "$lte": operators.le,
    "$gt": operators.gt,
    "$gte": operators.ge,
}

SPECIAL_CASED_OPERATORS = {
    "$in",
    "$nin",
    "$like",
}

BETWEEN_OPERATOR = {"$between"}

LOGICAL_OPERATORS = {"$and", "$or"}

SUPPORTED_OPERATORS = (
    set(COMPARISONS_TO_NATIVE)
    .union(NUMERIC_OPERATORS)
    .union(SPECIAL_CASED_OPERATORS)
    .union(BETWEEN_OPERATOR)
    .union(LOGICAL_OPERATORS)
)


class DistanceStrategy(str, Enum):
    """Distance Strategy class for SQLServer_VectorStore.

    Enumerator of the distance strategies for calculating distances
    between vectors.
    """

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT = "dot"


class VectorType(UserDefinedType):
    """VectorType - A custom type definition."""

    cache_ok = True

    def __init__(self, length: int) -> None:
        """__init__ for VectorType class."""
        self.length = length

    def get_col_spec(self, **kw: Any) -> str:
        """get_col_spec function for VectorType class."""
        return "vector(%s)" % self.length

    def bind_processor(self, dialect: Any) -> Any:
        """bind_processor function for VectorType class."""

        def process(value: Any) -> Any:
            return value

        return process

    def result_processor(self, dialect: Any, coltype: Any) -> Any:
        """result_processor function for VectorType class."""

        def process(value: Any) -> Any:
            return value

        return process


# String Constants
#
AZURE_TOKEN_URL = "https://database.windows.net/.default"  # Token URL for Azure DBs.
DISTANCE = "distance"
DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE
DEFAULT_TABLE_NAME = "sqlserver_vectorstore"
DISTANCE_STRATEGY = "distancestrategy"
EMBEDDING = "embedding"
EMBEDDING_LENGTH = "embedding_length"
EMBEDDING_VALUES = "embeddingvalues"
EMPTY_IDS_ERROR_MESSAGE = "Empty list of ids provided"
EXTRA_PARAMS = ";Trusted_Connection=Yes"
INVALID_IDS_ERROR_MESSAGE = "Invalid list of ids provided"
INVALID_INPUT_ERROR_MESSAGE = "Input is not valid."
INVALID_FILTER_INPUT_EXPECTED_DICT = """Invalid filter condition. Expected a dictionary
but got an empty dictionary"""
INVALID_FILTER_INPUT_EXPECTED_AND_OR = """Invalid filter condition.
Expected $and or $or but got: {}"""

SQL_COPT_SS_ACCESS_TOKEN = 1256  # Connection option defined by microsoft in msodbcsql.h

# Query Constants
#
JSON_TO_VECTOR_QUERY = f"cast (:{EMBEDDING_VALUES} as vector(:{EMBEDDING_LENGTH}))"
SERVER_JSON_CHECK_QUERY = "select name from sys.types where system_type_id = 244"
VECTOR_DISTANCE_QUERY = f"""
VECTOR_DISTANCE(:{DISTANCE_STRATEGY},
cast (:{EMBEDDING} as vector(:{EMBEDDING_LENGTH})), embeddings)"""


class SQLServer_VectorStore(VectorStore):
    """SQL Server Vector Store.

    This class provides a vector store interface for adding texts and performing
        similarity searches on the texts in SQL Server.
    """

    def __init__(
        self,
        *,
        connection: Optional[Connection] = None,
        connection_string: str,
        db_schema: Optional[str] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        embedding_function: Embeddings,
        embedding_length: int,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> None:
        """Initialize the SQL Server vector store.

        Args:
            connection: Optional SQLServer connection.
            connection_string: SQLServer connection string.
                If the connection string does not contain a username & password
                or `TrustedConnection=yes`, Entra ID authentication is used.
                SQL Server ODBC connection string can be retrieved from the
                `Connection strings` pane of the database in Azure portal.
                Sample connection string format:
                - "Driver=<drivername>;Server=<servername>;Database=<dbname>;
                Uid=<username>;Pwd=<password>;TrustServerCertificate=no;"
                - "mssql+pyodbc://username:password@servername/dbname?other_params"
            db_schema: The schema in which the vector store will be created.
                This schema must exist and the user must have permissions to the schema.
            distance_strategy: The distance strategy to use for comparing embeddings.
                Default value is COSINE. Available options are:
                - COSINE
                - DOT
                - EUCLIDEAN
            embedding_function: Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface.
            embedding_length: The length (dimension) of the vectors to be stored in the
                table.
                Note that only vectors of same size can be added to the vector store.
            relevance_score_fn: Relevance score funtion to be used.
                Optional param, defaults to None.
            table_name: The name of the table to use for storing embeddings.
                Default value is `sqlserver_vectorstore`.
        """
        self.connection_string = self._get_connection_url(connection_string)
        self._distance_strategy = distance_strategy
        self.embedding_function = embedding_function
        self._embedding_length = embedding_length
        self.schema = db_schema
        self.override_relevance_score_fn = relevance_score_fn
        self.table_name = table_name
        self._bind: Union[Connection, Engine] = (
            connection if connection else self._create_engine()
        )
        self._prepare_json_data_type()
        self._embedding_store = self._get_embedding_store(self.table_name, self.schema)
        self._create_table_if_not_exists()

    def _get_connection_url(self, conn_string: str) -> str:
        if conn_string is None or len(conn_string) == 0:
            logging.error("Connection string value is None or empty.")
            raise ValueError("Connection string value cannot be None.")

        if conn_string.startswith("mssql+pyodbc"):
            # Connection string is in a format that we can parse.
            #
            return conn_string

        try:
            args = conn_string.split(";")
            arg_dict = {}
            for arg in args:
                if "=" in arg:
                    # Split into key value pairs by the first positioned `=` found.
                    # Key-Value pairs are inserted into the dictionary.
                    #
                    key, value = arg.split("=", 1)
                    arg_dict[key.lower().strip()] = value.strip()

            # This will throw a key error if server or database keyword
            # is not present in arg_dict from the connection string.
            #
            database = arg_dict.pop("database")

            # If `server` is present in the dictionary, we split by
            # `,` to obtain host and port details.
            #
            server = arg_dict.pop("server").split(",", 1)
            server_host = server[0]
            server_port = None

            # Server details in SQLServer connection string from Azure portal
            # might be of the form `Server=tcp:servername`. In scenarios like this,
            # we remove the first part (tcp:) because `urlparse` function invoked in
            # `_can_connect_with_entra_id` expects an IP address when it sees `tcp:`
            # We can remove this without fear of a failure because it is omittable in
            # the connection string value.
            #
            if ":" in server_host:
                server_host = server_host.split(":", 1)[1]

            # Check if port is provided in server details,if true,
            # cast value to int if possible.
            #
            if len(server) > 1 and server[1].isdigit():
                server_port = int(server[1])

            # Args needed to be checked
            #
            username = arg_dict.pop("uid", None)
            password = arg_dict.pop("pwd", None)

            if "driver" in arg_dict.keys():
                # Extract driver value from curly braces if present.
                driver = re.search(r"\{([^}]*)\}", arg_dict["driver"])
                if driver is not None:
                    arg_dict["driver"] = driver.group(1)

            # Create connection URL for SQLAlchemy
            #
            url = URL.create(
                "mssql+pyodbc",
                username=username,
                password=password,
                database=database,
                host=server_host,
                port=server_port,
                query=arg_dict,
            )
        except KeyError as k:
            logging.error(
                f"Server, DB details were not provided in the connection string.\n{k}"
            )
            raise Exception(
                "Server, DB details should be provided in connection string."
            )
        except Exception as e:
            logging.error(f"An error has occurred.\n{e.__cause__}")
            raise

        # Return string version of the URL and ensure password
        # passed in is not obfuscated.
        #
        return url.render_as_string(hide_password=False)

    def _can_connect_with_entra_id(self) -> bool:
        """Determine if Entra ID authentication can be used.

        Check the components of the connection string to determine
        if connection via Entra ID authentication is possible or not.

        The connection string is of expected to be of the form:
            "mssql+pyodbc://username:password@servername/dbname?other_params"
        which gets parsed into -> <scheme>://<netloc>/<path>?<query>
        """
        parsed_url = urlparse(self.connection_string)

        if parsed_url is None:
            logging.error("Unable to parse connection string.")
            return False

        invalid_keywords = [
            "trusted_connection=yes",
            "trustedconnection=yes",
            "authentication",
            "integrated security",
        ]
        if (
            parsed_url.username
            or parsed_url.password
            or any(keyword in parsed_url.query.lower() for keyword in invalid_keywords)
        ):
            return False

        return True

    def _create_engine(self) -> Engine:
        if self._can_connect_with_entra_id():
            # Use Entra ID auth. Listen for a connection event
            # when `_create_engine` function from this class is called.
            #
            event.listen(Engine, "do_connect", self._provide_token, once=True)
            logging.info("Using Entra ID Authentication.")

        return create_engine(url=self.connection_string)

    def _create_table_if_not_exists(self) -> None:
        logging.info(f"Creating table {self.table_name}.")
        try:
            with Session(self._bind) as session:
                self._embedding_store.__table__.create(
                    session.get_bind(), checkfirst=True
                )
                session.commit()
        except ProgrammingError as e:
            logging.error(f"Create table {self.table_name} failed.")
            raise Exception(e.__cause__) from None

    def _get_embedding_store(self, name: str, schema: Optional[str]) -> Any:
        DynamicBase = declarative_base(class_registry=dict())  # type: Any
        if self._embedding_length is None or self._embedding_length < 1:
            raise ValueError("`embedding_length` value is not valid.")

        class EmbeddingStore(DynamicBase):
            """This is the base model for SQL vector store."""

            __tablename__ = name
            __table_args__ = (
                PrimaryKeyConstraint("id", mssql_clustered=False),
                Index("idx_custom_id", "custom_id", mssql_clustered=False, unique=True),
                {"schema": schema},
            )
            id = Column(Uuid, primary_key=True, default=uuid.uuid4)
            custom_id = Column(
                VARCHAR(1000), nullable=True
            )  # column for user defined ids.
            content_metadata = Column(JSON, nullable=True)
            content = Column(NVARCHAR, nullable=False)  # defaults to NVARCHAR(MAX)
            embeddings = Column(VectorType(self._embedding_length), nullable=False)

        return EmbeddingStore

    def _prepare_json_data_type(self) -> None:
        """Prepare for JSON data type usage.

        Check if the server has the JSON data type available. If it does,
        we compile JSON data type as JSON instead of NVARCHAR(max) used by
        sqlalchemy. If it doesn't, this defaults to NVARCHAR(max) as specified
        by sqlalchemy.
        """
        try:
            with Session(self._bind) as session:
                result = session.scalar(text(SERVER_JSON_CHECK_QUERY))
                session.close()

                if result is not None:

                    @compiles(JSON, "mssql")
                    def compile_json(
                        element: JSON, compiler: MSTypeCompiler, **kw: Any
                    ) -> str:
                        # return JSON when JSON data type is specified in this class.
                        return result  # json data type name in sql server

        except ProgrammingError as e:
            logging.error(f"Unable to get data types.\n {e.__cause__}\n")

    @property
    def embeddings(self) -> Embeddings:
        """`embeddings` property for SQLServer_VectorStore class."""
        return self.embedding_function

    @property
    def distance_strategy(self) -> str:
        """distance_strategy property for SQLServer_VectorStore class."""
        # Value of distance strategy passed in should be one of the supported values.
        if isinstance(self._distance_strategy, DistanceStrategy):
            return self._distance_strategy.value

        # Match string value with appropriate enum value, if supported.
        distance_strategy_lower = str.lower(self._distance_strategy)

        if distance_strategy_lower == DistanceStrategy.EUCLIDEAN.value:
            return DistanceStrategy.EUCLIDEAN.value
        elif distance_strategy_lower == DistanceStrategy.COSINE.value:
            return DistanceStrategy.COSINE.value
        elif distance_strategy_lower == DistanceStrategy.DOT.value:
            return DistanceStrategy.DOT.value
        else:
            raise ValueError(f"{self._distance_strategy} is not supported.")

    @distance_strategy.setter
    def distance_strategy(self, value: DistanceStrategy) -> None:
        self._distance_strategy = value

    @classmethod
    def from_texts(
        cls: Type[SQLServer_VectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        connection_string: str = str(),
        embedding_length: int = 0,
        table_name: str = DEFAULT_TABLE_NAME,
        db_schema: Optional[str] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> SQLServer_VectorStore:
        """Create a SQL Server vectorStore initialized from texts and embeddings.

        Args:
            texts: Iterable of strings to add into the vectorstore.
            embedding: Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface.
            metadatas: Optional list of metadatas (python dicts) associated
                with the input texts.
            connection_string: SQLServer connection string.
                If the connection string does not contain a username & password
                or `TrustedConnection=yes`, Entra ID authentication is used.
                SQL Server ODBC connection string can be retrieved from the
                `Connection strings` pane of the database in Azure portal.
                Sample connection string format:
                - "Driver=<drivername>;Server=<servername>;Database=<dbname>;
                Uid=<username>;Pwd=<password>;TrustServerCertificate=no;"
                - "mssql+pyodbc://username:password@servername/dbname?other_params"
            embedding_length: The length (dimension) of the vectors to be stored in the
                table.
                Note that only vectors of same size can be added to the vector store.
            table_name: The name of the table to use for storing embeddings.
            db_schema: The schema in which the vector store will be created.
                This schema must exist and the user must have permissions to the schema.
            distance_strategy: The distance strategy to use for comparing embeddings.
                Default value is COSINE. Available options are:
                - COSINE
                - DOT
                - EUCLIDEAN
            ids: Optional list of IDs for the input texts.
            **kwargs: vectorstore specific parameters.

        Returns:
            SQLServer_VectorStore: A SQL Server vectorstore.
        """
        store = cls(
            connection_string=connection_string,
            db_schema=db_schema,
            distance_strategy=distance_strategy,
            embedding_function=embedding,
            embedding_length=embedding_length,
            table_name=table_name,
            **kwargs,
        )

        store.add_texts(texts, metadatas, ids, **kwargs)
        return store

    @classmethod
    def from_documents(
        cls: Type[SQLServer_VectorStore],
        documents: List[Document],
        embedding: Embeddings,
        connection_string: str = str(),
        embedding_length: int = 0,
        table_name: str = DEFAULT_TABLE_NAME,
        db_schema: Optional[str] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> SQLServer_VectorStore:
        """Create a SQL Server vectorStore initialized from texts and embeddings.

        Args:
            documents: Documents to add to the vectorstore.
            embedding: Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface.
            connection_string: SQLServer connection string.
                If the connection string does not contain a username & password
                or `TrustedConnection=yes`, Entra ID authentication is used.
                SQL Server ODBC connection string can be retrieved from the
                `Connection strings` pane of the database in Azure portal.
                Sample connection string format:
                - "Driver=<drivername>;Server=<servername>;Database=<dbname>;
                Uid=<username>;Pwd=<password>;TrustServerCertificate=no;"
                - "mssql+pyodbc://username:password@servername/dbname?other_params"
            embedding_length: The length (dimension) of the vectors to be stored in the
                table.
                Note that only vectors of same size can be added to the vector store.
            table_name: The name of the table to use for storing embeddings.
                Default value is `sqlserver_vectorstore`.
            db_schema: The schema in which the vector store will be created.
                This schema must exist and the user must have permissions to the schema.
            distance_strategy: The distance strategy to use for comparing embeddings.
                Default value is COSINE. Available options are:
                - COSINE
                - DOT
                - EUCLIDEAN
            ids: Optional list of IDs for the input texts.
            **kwargs: vectorstore specific parameters.

        Returns:
            SQLServer_VectorStore: A SQL Server vectorstore.
        """
        texts, metadatas = [], []

        for doc in documents:
            if not isinstance(doc, Document):
                raise ValueError(
                    f"Expected an entry of type Document, but got {type(doc)}"
                )

            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        store = cls(
            connection_string=connection_string,
            db_schema=db_schema,
            distance_strategy=distance_strategy,
            embedding_function=embedding,
            embedding_length=embedding_length,
            table_name=table_name,
            **kwargs,
        )

        store.add_texts(texts, metadatas, ids, **kwargs)
        return store

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Get documents by their IDs from the vectorstore.

        Args:
            ids: List of IDs to retrieve.

        Returns:
            List of Documents
        """
        documents = []

        if ids is None or len(ids) == 0:
            logging.info(EMPTY_IDS_ERROR_MESSAGE)
        else:
            result = self._get_documents_by_ids(ids)
            for item in result:
                if item is not None:
                    documents.append(
                        Document(
                            id=item.custom_id,
                            page_content=item.content,
                            metadata=item.content_metadata,
                        )
                    )

        return documents

    def _get_documents_by_ids(self, ids: Sequence[str], /) -> Sequence[Any]:
        result: Sequence[Any] = []
        try:
            with Session(bind=self._bind) as session:
                statement = select(
                    self._embedding_store.custom_id,
                    self._embedding_store.content,
                    self._embedding_store.content_metadata,
                ).where(self._embedding_store.custom_id.in_(ids))
                result = session.execute(statement).fetchall()
        except DBAPIError as e:
            logging.error(e.__cause__)
        return result

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Determine relevance score function.

        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        If no relevance function is provided in the class constructor,
        selection is based on the distance strategy provided.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # If the relevance score function is not provided, we default to using
        # the distance strategy specified by the user.
        if self._distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.DOT:
            return self._max_inner_product_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        else:
            raise ValueError(
                "There is no supported normalization function for"
                f" {self._distance_strategy} distance strategy."
                "Consider providing relevance_score_fn to "
                "SQLServer_VectorStore construction."
            )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                Default is 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedded_query = self.embedding_function.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedded_query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                Default is 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        results = self._search_store(
            embedding, k=fetch_k, marginal_relevance=True, **kwargs
        )
        embedding_list = [json.loads(result[0]) for result in results]

        mmr_selects = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            lambda_mult=lambda_mult,
            k=k,
        )

        results_as_docs = self._docs_from_result(
            self._docs_and_scores_from_result(results)
        )

        # Return list of Documents from results_as_docs whose position
        # corresponds to the indices in mmr_selects.
        return [
            value for idx, value in enumerate(results_as_docs) if idx in mmr_selects
        ]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to given query.

        Args:
            query: Text to look up the most similar embedding to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Values for filtering on metadata during similarity search.

        Returns:
            List of Documents most similar to the query provided.
        """
        embedded_query = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(embedded_query, k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to the embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Values for filtering on metadata during similarity search.

        Returns:
            List of Documents most similar to the embedding provided.
        """
        similar_docs_with_scores = self.similarity_search_by_vector_with_score(
            embedding, k, **kwargs
        )
        return self._docs_from_result(similar_docs_with_scores)

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Similarity search with score.

        Run similarity search with distance and
            return docs most similar to the embedding vector.

        Args:
            query: Text to look up the most similar embedding to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Values for filtering on metadata during similarity search.

        Returns:
            List of tuple of Document and an accompanying score in order of
            similarity to the query provided.
            Note that, a smaller score implies greater similarity.
        """
        embedded_query = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector_with_score(embedded_query, k, **kwargs)

    def similarity_search_by_vector_with_score(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Similarity search by vector with score.

        Run similarity search with distance, given an embedding
            and return docs most similar to the embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Values for filtering on metadata during similarity search.

        Returns:
            List of tuple of Document and an accompanying score in order of
            similarity to the embedding provided.
            Note that, a smaller score implies greater similarity.
        """
        similar_docs = self._search_store(embedding, k, **kwargs)
        docs_and_scores = self._docs_and_scores_from_result(similar_docs)
        return docs_and_scores

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """`add_texts` function for SQLServer_VectorStore class.

        Compute the embeddings for the input texts and store embeddings
            in the vectorstore.

        Args:
            texts: Iterable of strings to add into the vectorstore.
            metadatas: List of metadatas (python dicts) associated with the input texts.
            ids: List of IDs for the input texts.
            **kwargs: vectorstore specific parameters.

        Returns:
            List of IDs generated from adding the texts into the vectorstore.
        """
        # Embed the texts passed in.
        embedded_texts = self.embedding_function.embed_documents(list(texts))

        # Insert the embedded texts in the vector store table.
        return self._insert_embeddings(texts, embedded_texts, metadatas, ids)

    def drop(self) -> None:
        """Drops every table created during initialization of vector store."""
        logging.info(f"Dropping vector store: {self.table_name}")
        try:
            with Session(bind=self._bind) as session:
                # Drop the table associated with the session bind.
                self._embedding_store.__table__.drop(session.get_bind())
                session.commit()

            logging.info(f"Vector store `{self.table_name}` dropped successfully.")

        except ProgrammingError as e:
            logging.error(f"Unable to drop vector store.\n {e.__cause__}.")

    def _search_store(
        self,
        embedding: List[float],
        k: int,
        filter: Optional[dict] = None,
        marginal_relevance: Optional[bool] = False,
    ) -> List[Any]:
        try:
            with Session(self._bind) as session:
                filter_by = []
                filter_clauses = self._create_filter_clause(filter)
                if filter_clauses is not None:
                    filter_by.append(filter_clauses)

                subquery = label(
                    DISTANCE,
                    text(VECTOR_DISTANCE_QUERY).bindparams(
                        bindparam(
                            DISTANCE_STRATEGY,
                            self.distance_strategy,
                            literal_execute=True,
                        ),
                        bindparam(
                            EMBEDDING,
                            json.dumps(embedding),
                            literal_execute=True,
                        ),
                        bindparam(
                            EMBEDDING_LENGTH,
                            self._embedding_length,
                            literal_execute=True,
                        ),
                    ),
                )

                # Results for marginal relevance includes additional
                # column for embeddings.
                if marginal_relevance:
                    query = (
                        select(
                            text("cast (embeddings as NVARCHAR(MAX))"),
                            subquery,
                            self._embedding_store,
                        )
                        .filter(*filter_by)
                        .order_by(asc(text(DISTANCE)))
                        .limit(k)
                    )
                    results = list(session.execute(query).fetchall())
                else:
                    results = (
                        session.query(
                            self._embedding_store,
                            subquery,
                        )
                        .filter(*filter_by)
                        .order_by(asc(text(DISTANCE)))
                        .limit(k)
                        .all()
                    )
        except ProgrammingError as e:
            logging.error(f"An error has occurred during the search.\n {e.__cause__}")
            raise Exception(e.__cause__) from None

        return results

    def _create_filter_clause(self, filters: Any) -> Any:
        """Create a filter clause.

        Convert LangChain Information Retrieval filter representation to matching
        SQLAlchemy clauses.

        At the top level, we still don't know if we're working with a field
        or an operator for the keys. After we've determined that we can
        call the appropriate logic to handle filter creation.

        Args:
            filters: Dictionary of filters to apply to the query.

        Returns:
            SQLAlchemy clause to apply to the query.

        Ex: For a filter,  {"$or": [{"id": 1}, {"name": "bob"}]}, the result is
            JSON_VALUE(langchain_vector_store_tests.content_metadata, :JSON_VALUE_1) =
              :JSON_VALUE_2 OR JSON_VALUE(langchain_vector_store_tests.content_metadata,
                :JSON_VALUE_3) = :JSON_VALUE_4
        """
        if filters is not None:
            if not isinstance(filters, dict):
                raise ValueError(
                    f"Expected a dict, but got {type(filters)} for value: {filter}"
                )
            if len(filters) == 1:
                # The only operators allowed at the top level are $AND and $OR
                # First check if an operator or a field
                key, value = list(filters.items())[0]
                if key.startswith("$"):
                    # Then it's an operator
                    if key.lower() not in LOGICAL_OPERATORS:
                        raise ValueError(
                            INVALID_FILTER_INPUT_EXPECTED_AND_OR.format(key)
                        )
                else:
                    # Then it's a field
                    return self._handle_field_filter(key, filters[key])

                # Here we handle the $and and $or operators
                if not isinstance(value, list):
                    raise ValueError(
                        f"Expected a list, but got {type(value)} for value: {value}"
                    )
                if key.lower() == "$and":
                    and_ = [self._create_filter_clause(el) for el in value]
                    if len(and_) > 1:
                        return sqlalchemy.and_(*and_)
                    elif len(and_) == 1:
                        return and_[0]
                    else:
                        raise ValueError(INVALID_FILTER_INPUT_EXPECTED_DICT)
                elif key.lower() == "$or":
                    or_ = [self._create_filter_clause(el) for el in value]
                    if len(or_) > 1:
                        return sqlalchemy.or_(*or_)
                    elif len(or_) == 1:
                        return or_[0]
                    else:
                        raise ValueError(INVALID_FILTER_INPUT_EXPECTED_DICT)

            elif len(filters) > 1:
                # Then all keys have to be fields (they cannot be operators)
                for key in filters.keys():
                    if key.startswith("$"):
                        raise ValueError(
                            f"Invalid filter condition. Expected a field but got: {key}"
                        )
                # These should all be fields and combined using an $and operator
                and_ = [self._handle_field_filter(k, v) for k, v in filters.items()]
                if len(and_) > 1:
                    return sqlalchemy.and_(*and_)
                elif len(and_) == 1:
                    return and_[0]
                else:
                    raise ValueError(INVALID_FILTER_INPUT_EXPECTED_DICT)
            else:
                raise ValueError("Got an empty dictionary for filters.")
        else:
            logging.info("No filters are passed, returning")
            return None

    def _handle_field_filter(
        self,
        field: str,
        value: Any,
    ) -> SQLColumnExpression:
        """Create a filter for a specific field.

        Args:
            field: name of field
            value: value to filter
                If provided as is then this will be an equality filter
                If provided as a dictionary then this will be a filter, the key
                will be the operator and the value will be the value to filter by

        Returns:
            sqlalchemy expression

        Ex: For a filter,  {"id": 1}, the result is

            JSON_VALUE(langchain_vector_store_tests.content_metadata, :JSON_VALUE_1) =
              :JSON_VALUE_2
        """
        if field.startswith("$"):
            raise ValueError(
                f"Invalid filter condition. Expected a field but got an operator: "
                f"{field}"
            )

        # Allow [a-zA-Z0-9_], disallow $ for now until we support escape characters
        if not field.isidentifier():
            raise ValueError(
                f"Invalid field name: {field}. Expected a valid identifier."
            )

        if isinstance(value, dict):
            # This is a filter specification that only 1 filter will be for a given
            # field, if multiple filters they are mentioned separately and used with
            # an AND on the top if nothing is specified
            if len(value) != 1:
                raise ValueError(
                    "Invalid filter condition. Expected a value which "
                    "is a dictionary with a single key that corresponds to an operator "
                    f"but got a dictionary with {len(value)} keys. The first few "
                    f"keys are: {list(value.keys())[:3]}"
                )
            operator, filter_value = list(value.items())[0]
            # Verify that operator is an operator
            if operator not in SUPPORTED_OPERATORS:
                raise ValueError(
                    f"Invalid operator: {operator}. "
                    f"Expected one of {SUPPORTED_OPERATORS}"
                )
        else:  # Then we assume an equality operator
            operator = "$eq"
            filter_value = value

        if operator in COMPARISONS_TO_NATIVE:
            operation = COMPARISONS_TO_NATIVE[operator]
            native_result = func.JSON_VALUE(
                self._embedding_store.content_metadata, f"$.{field}"
            )
            native_operation_result = operation(native_result, str(filter_value))
            return native_operation_result

        elif operator in NUMERIC_OPERATORS:
            operation = NUMERIC_OPERATORS[str(operator)]
            numeric_result = func.JSON_VALUE(
                self._embedding_store.content_metadata, f"$.{field}"
            )
            numeric_operation_result = operation(numeric_result, filter_value)

            if not isinstance(filter_value, str):
                numeric_operation_result = operation(
                    cast(numeric_result, Numeric(10, 2)), filter_value
                )

            return numeric_operation_result

        elif operator in BETWEEN_OPERATOR:
            # Use AND with two comparisons
            low, high = filter_value

            # Assuming lower_bound_value is a ColumnElement
            column_value = func.JSON_VALUE(
                self._embedding_store.content_metadata, f"$.{field}"
            )

            greater_operation = NUMERIC_OPERATORS["$gte"]
            lesser_operation = NUMERIC_OPERATORS["$lte"]

            lower_bound = greater_operation(column_value, low)
            upper_bound = lesser_operation(column_value, high)

            # Conditionally cast if filter_value is not a string
            if not isinstance(filter_value, str):
                lower_bound = greater_operation(cast(column_value, Numeric(10, 2)), low)
                upper_bound = lesser_operation(cast(column_value, Numeric(10, 2)), high)

            return sqlalchemy.and_(lower_bound, upper_bound)

        elif operator in SPECIAL_CASED_OPERATORS:
            # We'll do force coercion to text
            if operator in {"$in", "$nin"}:
                for val in filter_value:
                    if not isinstance(val, (str, int, float)):
                        raise NotImplementedError(
                            f"Unsupported type: {type(val)} for value: {val}"
                        )

            queried_field = func.JSON_VALUE(
                self._embedding_store.content_metadata, f"$.{field}"
            )

            if operator in {"$in"}:
                return queried_field.in_([str(val) for val in filter_value])
            elif operator in {"$nin"}:
                return queried_field.nin_([str(val) for val in filter_value])
            elif operator in {"$like"}:
                return queried_field.like(str(filter_value))
            else:
                raise NotImplementedError(f"Operator is not implemented: {operator}. ")
        else:
            raise NotImplementedError()

    def _docs_from_result(self, results: Any) -> List[Document]:
        """Formats the input into a result of type List[Document]."""
        docs = [doc for doc, _ in results if doc is not None]
        return docs

    def _docs_and_scores_from_result(
        self, results: List[Any]
    ) -> List[Tuple[Document, float]]:
        """Formats the input into a result of type Tuple[Document, float].

        If an invalid input is given, it does not attempt to format the value
        and instead logs an error.
        """
        docs_and_scores = []

        for result in results:
            if (
                result is not None
                and result.EmbeddingStore is not None
                and result.distance is not None
            ):
                docs_and_scores.append(
                    (
                        Document(
                            page_content=result.EmbeddingStore.content,
                            metadata=result.EmbeddingStore.content_metadata,
                        ),
                        result.distance,
                    )
                )
            else:
                logging.error(INVALID_INPUT_ERROR_MESSAGE)

        return docs_and_scores

    def _insert_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert the embeddings and the texts in the vectorstore.

        Args:
            texts: Iterable of strings to add into the vectorstore.
            embeddings: List of list of embeddings.
            metadatas: List of metadatas (python dicts) associated with the input texts.
            ids: List of IDs for the input texts.
            **kwargs: vectorstore specific parameters.

        Returns:
            List of IDs generated from adding the texts into the vectorstore.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        try:
            if ids is None:
                # Get IDs from metadata if available.
                ids = [metadata.get("id", uuid.uuid4()) for metadata in metadatas]

            with Session(self._bind) as session:
                documents = []
                for idx, query in enumerate(texts):
                    # For a query, if there is no corresponding ID,
                    # we generate a uuid and add it to the list of IDs to be returned.
                    if idx < len(ids):
                        custom_id = ids[idx]
                    else:
                        ids.append(str(uuid.uuid4()))
                        custom_id = ids[-1]
                    embedding = embeddings[idx]
                    metadata = metadatas[idx] if idx < len(metadatas) else {}

                    # Construct text, embedding, metadata as EmbeddingStore model
                    # to be inserted into the table.
                    sqlquery = select(
                        text(JSON_TO_VECTOR_QUERY).bindparams(
                            bindparam(
                                EMBEDDING_VALUES,
                                json.dumps(embedding),
                                literal_execute=True,
                                # when unique is set to true, the name of the key
                                # for each bindparameter is made unique, to avoid
                                # using the wrong bound parameter during compile.
                                # This is especially needed since we're creating
                                # and storing multiple queries to be bulk inserted
                                # later on.
                                unique=True,
                            ),
                            bindparam(
                                EMBEDDING_LENGTH,
                                self._embedding_length,
                                literal_execute=True,
                            ),
                        )
                    )
                    # `embedding_store` is created in a dictionary format instead
                    # of using the embedding_store object from this class.
                    # This enables the use of `insert().values()` which can only
                    # take a dict and not a custom object.
                    embedding_store = {
                        "custom_id": custom_id,
                        "content_metadata": metadata,
                        "content": query,
                        "embeddings": sqlquery,
                    }
                    documents.append(embedding_store)
                session.execute(insert(self._embedding_store).values(documents))
                session.commit()
        except DBAPIError as e:
            logging.error(f"Add text failed:\n {e.__cause__}\n")
            raise Exception(e.__cause__) from None
        except AttributeError:
            logging.error("Metadata must be a list of dictionaries.")
            raise
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete embeddings in the vectorstore by the ids.

        Args:
            ids: List of IDs to delete. If None, delete all. Default is None.
                No data is deleted if empty list is provided.
            kwargs: vectorstore specific parameters.

        Returns:
            Optional[bool]
        """
        if ids is not None and len(ids) == 0:
            logging.info(EMPTY_IDS_ERROR_MESSAGE)
            return False

        result = self._delete_texts_by_ids(ids)
        if result == 0:
            logging.info(INVALID_IDS_ERROR_MESSAGE)
            return False

        logging.info(f"{result} rows affected.")
        return True

    def _delete_texts_by_ids(self, ids: Optional[List[str]] = None) -> int:
        try:
            with Session(bind=self._bind) as session:
                if ids is None:
                    logging.info("Deleting all data in the vectorstore.")
                    result = session.query(self._embedding_store).delete()
                else:
                    result = (
                        session.query(self._embedding_store)
                        .filter(self._embedding_store.custom_id.in_(ids))
                        .delete()
                    )
                session.commit()
        except DBAPIError as e:
            logging.error(e.__cause__)
        return result

    def _provide_token(
        self,
        dialect: Dialect,
        conn_rec: Optional[ConnectionPoolEntry],
        cargs: List[str],
        cparams: MutableMapping[str, Any],
    ) -> None:
        """Function to retreive access token for connection.

        Get token for SQLServer connection from token URL,
        and use the token to connect to the database.
        """
        credential = DefaultAzureCredential()

        # Remove Trusted_Connection param that SQLAlchemy adds to
        # the connection string by default.
        cargs[0] = cargs[0].replace(EXTRA_PARAMS, str())

        # Create credential token
        token_bytes = credential.get_token(AZURE_TOKEN_URL).token.encode("utf-16-le")
        token_struct = struct.pack(
            f"<I{len(token_bytes)}s", len(token_bytes), token_bytes
        )

        # Apply credential token to keyword argument
        cparams["attrs_before"] = {SQL_COPT_SS_ACCESS_TOKEN: token_struct}
