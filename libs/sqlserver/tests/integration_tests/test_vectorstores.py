"""Test SQLServer_VectorStore functionality."""

import os
from typing import Any, Dict, Generator, List
from unittest import mock
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from sqlalchemy import create_engine, text

from langchain_sqlserver.vectorstores import DistanceStrategy, SQLServer_VectorStore
from tests.utils.fake_embeddings import DeterministicFakeEmbedding
from tests.utils.filtering_test_cases import (
    IDS as filter_ids,
)
from tests.utils.filtering_test_cases import (
    TYPE_1_FILTERING_TEST_CASES,
    TYPE_2_FILTERING_TEST_CASES,
    TYPE_3_FILTERING_TEST_CASES,
    TYPE_4_FILTERING_TEST_CASES,
    TYPE_5_FILTERING_TEST_CASES,
)
from tests.utils.filtering_test_cases import (
    metadatas as filter_metadatas,
)
from tests.utils.filtering_test_cases import (
    texts as filter_texts,
)

pytest.skip(
    "Skipping these tests pending resource availability", allow_module_level=True
)

# Connection String values should be provided in the
# environment running this test suite.
#
_CONNECTION_STRING = str(os.environ.get("TEST_AZURESQLSERVER_TRUSTED_CONNECTION"))
_CONNECTION_STRING_WITH_UID_AND_PWD = str(
    os.environ.get("TEST_AZURESQLSERVER_CONNECTION_STRING_WITH_UID")
)
_ENTRA_ID_CONNECTION_STRING_NO_PARAMS = str(
    os.environ.get("TEST_ENTRA_ID_CONNECTION_STRING_NO_PARAMS")
)
_ENTRA_ID_CONNECTION_STRING_TRUSTED_CONNECTION_NO = str(
    os.environ.get("TEST_ENTRA_ID_CONNECTION_STRING_TRUSTED_CONNECTION_NO")
)
_MASTER_DATABASE_CONNECTION_STRING = str(
    os.environ.get("TEST_AZURESQLSERVER_MASTER_CONNECTION_STRING")
)
_COLLATION_DB_CONNECTION_STRING = str(
    os.environ.get("TEST_AZURESQLSERVER_COLLATION_DB_CONN_STRING")
)
_PYODBC_CONNECTION_STRING = str(os.environ.get("TEST_PYODBC_CONNECTION_STRING"))
_SCHEMA = "lc_test"
_COLLATION_DB_NAME = "LangChainCollationTest"
_TABLE_NAME = "langchain_vector_store_tests"
_TABLE_DOES_NOT_EXIST = "Table %s.%s does not exist."
EMBEDDING_LENGTH = 1536

# Query Strings
#
_CREATE_COLLATION_DB_QUERY = (
    f"create database {_COLLATION_DB_NAME} collate SQL_Latin1_General_CP1_CS_AS;"
)
_COLLATION_QUERY = "select name, collation_name from sys.databases where name = N'%s';"
_DROP_COLLATION_DB_QUERY = f"drop database {_COLLATION_DB_NAME}"
_SYS_TABLE_QUERY = """
select object_id from sys.tables where name = '%s'
and schema_name(schema_id) = '%s'"""


# Combine all test cases into one list with additional debugging
FILTERING_TEST_CASES: List[Any] = []
for filterList in [
    TYPE_1_FILTERING_TEST_CASES,
    TYPE_2_FILTERING_TEST_CASES,
    TYPE_3_FILTERING_TEST_CASES,
    TYPE_4_FILTERING_TEST_CASES,
    TYPE_5_FILTERING_TEST_CASES,
]:
    if isinstance(filterList, list):
        for filters in filterList:
            if isinstance(filters, tuple):
                FILTERING_TEST_CASES.append(filters)


@pytest.fixture
def store() -> Generator[SQLServer_VectorStore, None, None]:
    """Setup resources that are needed for the duration of the test."""
    store = SQLServer_VectorStore(
        connection_string=_CONNECTION_STRING,
        embedding_length=EMBEDDING_LENGTH,
        # DeterministicFakeEmbedding returns embeddings of the same
        # size as `embedding_length`.
        embedding_function=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
        table_name=_TABLE_NAME,
    )
    yield store  # provide this data to the test

    # Drop store after it's done being used in the test case.
    store.drop()


@pytest.fixture
def texts() -> List[str]:
    """Definition of texts used in the tests."""
    query = [
        """I have bought several of the Vitality canned dog food products and have 
        found them all to be of good quality. The product looks more like a stew 
        than a processed meat and it smells better. My Labrador is finicky and she 
        appreciates this product better than  most.""",
        """The candy is just red , No flavor . Just  plan and chewy .
        I would never buy them again""",
        "Arrived in 6 days and were so stale i could not eat any of the 6 bags!!",
        """Got these on sale for roughly 25 cents per cup, which is half the price 
        of my local grocery stores, plus they rarely stock the spicy flavors. These 
        things are a GREAT snack for my office where time is constantly crunched and 
        sometimes you can't escape for a real meal. This is one of my favorite flavors 
        of Instant Lunch and will be back to buy every time it goes on sale.""",
        """If you are looking for a less messy version of licorice for the children, 
        then be sure to try these!  They're soft, easy to chew, and they don't get your 
        hands all sticky and gross in the car, in the summer, at the beach, etc. 
        We love all the flavos and sometimes mix these in with the chocolate to have a 
        very nice snack! Great item, great price too, highly recommend!""",
    ]
    return query  # provide this data to the test.


@pytest.fixture
def metadatas() -> List[dict]:
    """Definition of metadatas used in the tests."""
    query_metadata = [
        {"id": 1, "summary": "Good Quality Dog Food"},
        {"id": 2, "summary": "Nasty No flavor"},
        {"id": 3, "summary": "stale product"},
        {"id": 4, "summary": "Great value and convenient ramen"},
        {"id": 5, "summary": "Great for the kids!"},
    ]
    return query_metadata  # provide this data to the test.


@pytest.fixture
def docs() -> List[Document]:
    """Definition of doc variable used in the tests."""
    docs = [
        Document(
            page_content="rabbit",
            metadata={"color": "black", "type": "pet", "length": 6},
        ),
        Document(
            page_content="cherry",
            metadata={"color": "red", "type": "fruit", "length": 6},
        ),
        Document(
            page_content="hamster",
            metadata={"color": "brown", "type": "pet", "length": 7},
        ),
        Document(
            page_content="cat", metadata={"color": "black", "type": "pet", "length": 3}
        ),
        Document(
            page_content="elderberry",
            metadata={"color": "blue", "type": "fruit", "length": 10},
        ),
    ]
    return docs  # provide this data to the test


def test_sqlserver_add_texts(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that `add_texts` returns equivalent number of ids of input texts."""
    result = store.add_texts(texts, metadatas)
    assert len(result) == len(texts)


def test_sqlserver_add_texts_when_no_metadata_is_provided(
    store: SQLServer_VectorStore,
    texts: List[str],
) -> None:
    """Test that when user calls the add_texts function without providing metadata,
    the embedded text still get added to the vector store."""
    result = store.add_texts(texts)
    assert len(result) == len(texts)


def test_sqlserver_add_texts_when_text_length_and_metadata_length_vary(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that all texts provided are added into the vector store
    even when metadata is not available for all the texts."""
    # We get all metadatas except the last one from our metadatas fixture.
    # The text without a corresponding metadata should be added to the vector store.
    metadatas = metadatas[:-1]
    result = store.add_texts(texts, metadatas)
    assert len(result) == len(texts)


def test_sqlserver_add_texts_when_list_of_given_id_is_less_than_list_of_texts(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that when length of given id is less than length of texts,
    random ids are created."""
    # List of ids is one less than len(texts) which is 5.
    metadatas = metadatas[:-1]
    metadatas.append({"summary": "Great for the kids!"})
    result = store.add_texts(texts, metadatas)
    # Length of ids returned by add_texts function should be equal to length of texts.
    assert len(result) == len(texts)


def test_add_document_with_sqlserver(
    store: SQLServer_VectorStore,
    docs: List[Document],
) -> None:
    """Test that when add_document function is used, it integrates well
    with the add_text function in SQLServer Vector Store."""
    result = store.add_documents(docs)
    assert len(result) == len(docs)


def test_that_a_document_entry_without_metadata_will_be_added_to_vectorstore(
    store: SQLServer_VectorStore,
    docs: List[Document],
) -> None:
    """Test that you can add a document that has no metadata into the vectorstore."""
    documents = docs[:-1]
    documents.append(Document(page_content="elderberry"))
    result = store.add_documents(documents)
    assert len(result) == len(documents)


def test_that_drop_deletes_vector_store(
    store: SQLServer_VectorStore,
    texts: List[str],
) -> None:
    """Test that when drop is called, vector store is deleted
    and a call to add_text raises an exception.
    """
    store.drop()
    with pytest.raises(Exception):
        store.add_texts(texts)


def test_that_add_text_fails_if_text_embedding_length_is_not_equal_to_embedding_length(
    store: SQLServer_VectorStore,
    texts: List[str],
) -> None:
    """Test that a call to add_texts will raise an exception if the embedding_length of
    the embedding function in use is not the same as the embedding_length used in
    creating the vector store."""
    store.add_texts(texts)

    # Assign a new embedding function with a different length to the store.
    #
    store.embedding_function = DeterministicFakeEmbedding(
        size=384
    )  # a different size is used.

    with pytest.raises(Exception):
        # add_texts should fail and raise an exception since embedding length of
        # the newly assigned embedding_function is different from the initial
        # embedding length.
        store.add_texts(texts)


def test_sqlserver_delete_text_by_id_valid_ids_provided(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that delete API deletes texts by id."""

    store.add_texts(texts, metadatas)

    result = store.delete(["1", "2", "5"])
    # Should return true since valid ids are given
    assert result


def test_sqlserver_delete_text_by_id_valid_id_and_invalid_ids_provided(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that delete API deletes texts by id."""

    store.add_texts(texts, metadatas)

    result = store.delete(["1", "2", "6", "9"])
    # Should return true since valid ids are given
    assert result


def test_sqlserver_delete_text_by_id_invalid_ids_provided(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that delete API deletes texts by id."""

    store.add_texts(texts, metadatas)

    result = store.delete(["100000"])
    # Should return False since given id is not in DB
    assert not result


def test_sqlserver_delete_text_by_id_no_ids_provided(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that delete API deletes all data in vectorstore
    when `None` is provided as the parameter."""

    store.add_texts(texts, metadatas)
    result = store.delete(None)

    # Should return True, since None is provided,
    # all data in vectorstore is deleted.
    assert result

    # Check that length of data in vectorstore after
    # delete has been invoked is zero.
    conn = create_engine(_PYODBC_CONNECTION_STRING).connect()
    data = conn.execute(text(f"select * from {_TABLE_NAME}")).fetchall()
    conn.close()

    assert len(data) == 0, f"VectorStore {_TABLE_NAME} is not empty."


def test_sqlserver_delete_text_by_id_empty_list_provided(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that delete API does not delete data
    if empty list of ID is provided."""

    store.add_texts(texts, metadatas)
    result = store.delete([])

    # Should return False, since empty list of ids given
    assert not result


def test_that_multiple_vector_stores_can_be_created(
    store: SQLServer_VectorStore,
) -> None:
    """Tests that when multiple SQLServer_VectorStore objects are
    created, the first created vector store is not reused, but
    multiple vector stores are created."""

    # Create another vector store with a different table name.
    new_store = SQLServer_VectorStore(
        connection_string=_CONNECTION_STRING,
        embedding_function=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
        embedding_length=EMBEDDING_LENGTH,
        table_name="langchain_vector_store_tests_2",
    )

    # Check that the name of the table being created for the embeddingstore
    # is what is expected.
    assert new_store._embedding_store.__table__.name == "langchain_vector_store_tests_2"

    # Drop the new_store table to clean up this test run.
    new_store.drop()


def test_sqlserver_from_texts(
    texts: List[str],
) -> None:
    """Test that a call to `from_texts` initializes a
    SQLServer vectorstore from texts."""
    vectorstore = SQLServer_VectorStore.from_texts(
        connection_string=_CONNECTION_STRING,
        embedding=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
        embedding_length=EMBEDDING_LENGTH,
        table_name=_TABLE_NAME,
        texts=texts,
    )
    assert vectorstore is not None

    # Check that vectorstore contains the texts passed in as parameters.
    connection = create_engine(_PYODBC_CONNECTION_STRING).connect()
    result = connection.execute(text(f"select * from {_TABLE_NAME}")).fetchall()
    connection.close()

    vectorstore.drop()
    assert len(result) == len(texts)


def test_sqlserver_from_documents(
    docs: List[Document],
) -> None:
    """Test that a call to `from_documents` initializes a
    SQLServer vectorstore from documents."""
    vectorstore = SQLServer_VectorStore.from_documents(
        connection_string=_CONNECTION_STRING,
        embedding=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
        embedding_length=EMBEDDING_LENGTH,
        table_name=_TABLE_NAME,
        documents=docs,
    )
    assert vectorstore is not None

    # Check that vectorstore contains the texts passed in as parameters.
    connection = create_engine(_PYODBC_CONNECTION_STRING).connect()
    result = connection.execute(text(f"select * from {_TABLE_NAME}")).fetchall()
    connection.close()

    vectorstore.drop()
    assert len(result) == len(docs)


def test_get_by_ids(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that `get_by_ids` returns documents."""
    ids = store.add_texts(texts, metadatas)

    # Append non-existent id to list of IDs to get.
    ids.append("200")
    documents = store.get_by_ids(ids)

    # Assert that the length of documents returned
    # is the same as length of inserted texts.
    assert len(documents) == len(metadatas)

    # Assert that the length of documents returned is not equal to
    # length of ids since there is a non-existent ID in the list of IDs.
    assert len(documents) != len(ids)


def test_that_schema_input_is_used() -> None:
    """Tests that when a schema is given as input to the SQLServer_VectorStore object,
    a vector store is created within the schema."""
    connection = create_engine(_PYODBC_CONNECTION_STRING).connect()
    # Create a schema in the DB
    connection.execute(text(f"create schema {_SCHEMA}"))

    # Create a vector store in the DB with the schema just created
    sqlserver_vectorstore = SQLServer_VectorStore(
        connection=connection,
        connection_string=_CONNECTION_STRING,
        db_schema=_SCHEMA,
        embedding_function=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
        embedding_length=EMBEDDING_LENGTH,
        table_name=_TABLE_NAME,
    )
    sqlserver_vectorstore.add_texts(["cats"])

    # Confirm table in that schema exists.
    result = connection.execute(text(_SYS_TABLE_QUERY % (_TABLE_NAME, _SCHEMA)))
    assert result.fetchone() is not None, _TABLE_DOES_NOT_EXIST % (_SCHEMA, _TABLE_NAME)
    connection.close()


def test_that_same_name_vector_store_can_be_created_in_different_schemas() -> None:
    """Tests that vector stores can be created with same name in different
    schemas even with the same connection."""
    connection = create_engine(_PYODBC_CONNECTION_STRING).connect()
    # Create a schema in the DB
    connection.execute(text(f"create schema {_SCHEMA}"))

    # Create a vector store in the DB with the schema just created
    sqlserver_vectorstore = SQLServer_VectorStore(
        connection=connection,
        connection_string=_CONNECTION_STRING,
        db_schema=_SCHEMA,
        embedding_function=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
        embedding_length=EMBEDDING_LENGTH,
        table_name=_TABLE_NAME,
    )

    # Create a vector store in the DB with the default schema
    sqlserver_vectorstore_default_schema = SQLServer_VectorStore(
        connection=connection,
        connection_string=_CONNECTION_STRING,
        embedding_function=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
        embedding_length=EMBEDDING_LENGTH,
        table_name=_TABLE_NAME,
    )

    sqlserver_vectorstore.add_texts(["cats"])
    result_with_schema = connection.execute(
        text(_SYS_TABLE_QUERY % (_TABLE_NAME, _SCHEMA))
    )
    assert result_with_schema.fetchone() is not None, _TABLE_DOES_NOT_EXIST % (
        _SCHEMA,
        _TABLE_NAME,
    )

    sqlserver_vectorstore_default_schema.add_texts(["cats"])
    result_with_default = connection.execute(
        text(_SYS_TABLE_QUERY % (_TABLE_NAME, "dbo"))
    )
    assert result_with_default.fetchone() is not None, _TABLE_DOES_NOT_EXIST % (
        "dbo",
        _TABLE_NAME,
    )
    connection.close()


def test_that_only_same_size_embeddings_can_be_added_to_store(
    store: SQLServer_VectorStore,
    texts: List[str],
) -> None:
    """Tests that the vector store can
    take only vectors of same dimensions."""
    # Create a SQLServer_VectorStore without `embedding_length` defined.
    store.add_texts(texts)

    # Add texts using an embedding function with a different length.
    # This should raise an exception.
    #
    store.embedding_function = DeterministicFakeEmbedding(size=420)
    with pytest.raises(Exception):
        store.add_texts(texts)


def test_that_similarity_search_returns_expected_no_of_documents(
    store: SQLServer_VectorStore,
    texts: List[str],
) -> None:
    """Test that the amount of documents returned when similarity search
    is called is the same as the number of documents requested."""
    store.add_texts(texts)
    number_of_docs_to_return = 3
    result = store.similarity_search(query="Good review", k=number_of_docs_to_return)
    assert len(result) == number_of_docs_to_return


def test_that_similarity_search_returns_results_with_scores_sorted_in_ascending_order(
    store: SQLServer_VectorStore,
    texts: List[str],
) -> None:
    """Assert that the list returned by a similarity search
    is sorted in an ascending order. The implication is that
    we have the smallest score (most similar doc.) returned first.
    """
    store.add_texts(texts)
    number_of_docs_to_return = 4
    doc_with_score = store.similarity_search_with_score(
        "Good review", k=number_of_docs_to_return
    )
    assert doc_with_score == sorted(doc_with_score, key=lambda x: x[1])


def test_that_case_sensitivity_does_not_affect_distance_strategy(
    texts: List[str],
) -> None:
    """Test that when distance strategy is set on a case sensitive DB,
    a call to similarity search does not fail."""
    connection_string_to_master = _MASTER_DATABASE_CONNECTION_STRING

    conn = create_engine(connection_string_to_master).connect()
    conn.rollback()

    if conn.connection.dbapi_connection is not None:
        conn.connection.dbapi_connection.autocommit = True

    conn.execute(text(_CREATE_COLLATION_DB_QUERY))
    conn.execute(text(f"use {_COLLATION_DB_NAME}"))

    store = SQLServer_VectorStore(
        connection=conn,
        connection_string=_COLLATION_DB_CONNECTION_STRING,
        # DeterministicFakeEmbedding returns embeddings of the same
        # size as `embedding_length`.
        embedding_length=EMBEDDING_LENGTH,
        embedding_function=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
        table_name=_TABLE_NAME,
    )
    collation_query_result = (
        conn.execute(text(_COLLATION_QUERY % (_COLLATION_DB_NAME))).fetchone()
    )  # Sample return value: ('LangChainVectors', 'SQL_Latin1_General_CP1_CS_AS')

    assert (
        collation_query_result is not None
    ), "No collation data returned from the database."
    # Confirm DB is case sensitive
    assert "_CS" in collation_query_result.collation_name

    store.add_texts(texts)
    store.distance_strategy = DistanceStrategy.DOT

    # Call to similarity_search function should not error out.
    number_of_docs_to_return = 2
    result = store.similarity_search(query="Good review", k=number_of_docs_to_return)
    assert result is not None and len(result) == number_of_docs_to_return
    store.drop()

    # Drop DB with case sensitive collation for test.
    conn.execute(text("use master"))
    conn.execute(text(_DROP_COLLATION_DB_QUERY))
    conn.close()


def test_sqlserver_with_no_metadata_filters(store: SQLServer_VectorStore) -> None:
    store.add_texts(filter_texts, None, filter_ids)
    try:
        test_filter: Dict[str, Any] = {"id": 1}
        expected_ids: List[int] = []
        docs = store.similarity_search("meow", k=5, filter=test_filter)
        returned_ids = [doc.metadata["id"] for doc in docs]
        assert sorted(returned_ids) == sorted(expected_ids), test_filter

    finally:
        store.delete(["1", "2", "3"])


@pytest.mark.parametrize("test_filter, expected_ids", FILTERING_TEST_CASES)
def test_sqlserver_with_metadata_filters(
    store: SQLServer_VectorStore,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    store.add_texts(filter_texts, filter_metadatas, filter_ids)
    try:
        docs = store.similarity_search("meow", k=5, filter=test_filter)
        returned_ids = [doc.metadata["id"] for doc in docs]
        assert sorted(returned_ids) == sorted(expected_ids), test_filter

    finally:
        store.delete(["1", "2", "3"])


@pytest.mark.parametrize(
    "invalid_filter",
    [
        ["hello"],
        {
            "id": 2,
            "$name": "foo",
        },
        {"$or": {}},
        {"$and": {}},
        {"$between": {}},
        {"$eq": {}},
        {"$or": None},
    ],
)
def test_invalid_filters(
    store: SQLServer_VectorStore, invalid_filter: Dict[str, Any]
) -> None:
    """Verify that invalid filters raise an error."""
    store.add_texts(filter_texts, filter_metadatas, filter_ids)
    store.delete(["1", "2", "3"])
    with pytest.raises(ValueError):
        store.similarity_search("meow", k=5, filter=invalid_filter)


def test_that_rows_with_duplicate_custom_id_cannot_be_entered(
    store: SQLServer_VectorStore,
    texts: List[str],
) -> None:
    """Test that if a row is specified with existing ID in the table,
    `add_texts` fails."""
    metadatas = [
        {"id": 1, "summary": "Good Quality Dog Food"},
        {"id": 2, "summary": "Nasty No flavor"},
        {"id": 2, "summary": "stale product"},
        {"id": 4, "summary": "Great value and convenient ramen"},
        {"id": 5, "summary": "Great for the kids!"},
    ]
    with pytest.raises(Exception):
        store.add_texts(texts, metadatas)


def test_that_entra_id_authentication_connection_is_successful(
    texts: List[str],
) -> None:
    """Test that given a valid entra id auth string, connection to DB is successful."""
    vector_store = connect_to_vector_store(_ENTRA_ID_CONNECTION_STRING_NO_PARAMS)
    vector_store.add_texts(texts)

    # drop vector_store
    vector_store.drop()


def test_that_max_marginal_relevance_search_returns_expected_no_of_documents(
    store: SQLServer_VectorStore,
    texts: List[str],
) -> None:
    """Test that the size of documents returned when `max_marginal_relevance_search`
    is called is the expected number of documents requested."""
    store.add_texts(texts)
    number_of_docs_to_return = 3
    result = store.max_marginal_relevance_search(
        query="Good review", k=number_of_docs_to_return
    )
    assert len(result) == number_of_docs_to_return


def test_similarity_search_with_relevance_score(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that the size of documents returned when
    `similarity_search_with_relevance_scores` is called
    is the expected number of documents requested."""
    number_of_docs_to_return = 3

    store.add_texts(texts, metadatas)
    result = store.similarity_search_with_relevance_scores(
        "Good review", k=number_of_docs_to_return
    )
    assert len(result) == number_of_docs_to_return


def test_similarity_search_with_relevance_score_result_constraint(
    store: SQLServer_VectorStore,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    """Test that the result from similarity_search_with_relevance_score
    is capped between 0 and 1."""
    store.add_texts(texts, metadatas)
    result = store.similarity_search_with_relevance_scores("Good review")

    for value in result:
        assert (
            value[1] >= 0 and value[1] <= 1
        ), f"Relevance score {value[1]} not in range [0, 1]."


def test_select_relevance_score_fn_returns_correct_relevance_function(
    store: SQLServer_VectorStore,
    docs: List[Document],
) -> None:
    """Test that `_select_relevance_score_fn` uses the appropriate
    relevance function based on distance strategy provided."""
    store.add_documents(docs)
    result = store._select_relevance_score_fn()

    # This vectorstore is initialized to use `COSINE` distance strategy,
    # as such, the relevance function returned should be a
    # cosine relevance function.
    assert result == store._cosine_relevance_score_fn


def test_select_relevance_score_fn_raises_exception_with_invalid_value(
    store: SQLServer_VectorStore,
    docs: List[Document],
) -> None:
    """Test that `_select_relevance_score_fn` raises a value error
    if an invalid distance_strategy is provided."""
    store.add_documents(docs)
    store.distance_strategy = "InvalidDistanceStrategy"

    # Invocation of `_select_relevance_score_fn` should raise a ValueError
    # since the `distance_strategy` value is invalid.
    with pytest.raises(ValueError):
        store._select_relevance_score_fn()


# We need to mock this so that actual connection is not attempted
# after mocking _provide_token.
@mock.patch("sqlalchemy.dialects.mssql.dialect.initialize")
@mock.patch("langchain_sqlserver.vectorstores.SQLServer_VectorStore._provide_token")
@mock.patch(
    "langchain_sqlserver.vectorstores.SQLServer_VectorStore._prepare_json_data_type"
)
def test_that_given_a_valid_entra_id_connection_string_entra_id_authentication_is_used(
    prep_data_type: Mock,
    provide_token: Mock,
    dialect_initialize: Mock,
) -> None:
    """Test that if a valid entra_id connection string is passed in
    to SQLServer_VectorStore object, entra id authentication is used
    and connection is successful."""

    # Connection string is of the form below.
    # "mssql+pyodbc://lc-test.database.windows.net,1433/lcvectorstore
    # ?driver=ODBC+Driver+17+for+SQL+Server"
    store = connect_to_vector_store(_ENTRA_ID_CONNECTION_STRING_NO_PARAMS)
    # _provide_token is called only during Entra ID authentication.
    provide_token.assert_called()
    store.drop()

    # reset the mock so that it can be reused.
    provide_token.reset_mock()

    # "mssql+pyodbc://lc-test.database.windows.net,1433/lcvectorstore
    # ?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=no"
    store = connect_to_vector_store(_ENTRA_ID_CONNECTION_STRING_TRUSTED_CONNECTION_NO)
    provide_token.assert_called()
    store.drop()


# We need to mock this so that actual connection is not attempted
# after mocking _provide_token.
@mock.patch("sqlalchemy.dialects.mssql.dialect.initialize")
@mock.patch("langchain_sqlserver.vectorstores.SQLServer_VectorStore._provide_token")
@mock.patch(
    "langchain_sqlserver.vectorstores.SQLServer_VectorStore._prepare_json_data_type"
)
def test_that_given_a_connection_string_with_uid_and_pwd_entra_id_auth_is_not_used(
    prep_data_type: Mock,
    provide_token: Mock,
    dialect_initialize: Mock,
) -> None:
    """Test that if a connection string is provided to SQLServer_VectorStore object,
    and connection string has username and password, entra id authentication is not
    used and connection is successful."""

    # Connection string contains username and password,
    # mssql+pyodbc://username:password@lc-test.database.windows.net,1433/lcvectorstore
    # ?driver=ODBC+Driver+17+for+SQL+Server"
    store = connect_to_vector_store(_CONNECTION_STRING_WITH_UID_AND_PWD)
    # _provide_token is called only during Entra ID authentication.
    provide_token.assert_not_called()
    store.drop()


# We need to mock this so that actual connection is not attempted
# after mocking _provide_token.
@mock.patch("sqlalchemy.dialects.mssql.dialect.initialize")
@mock.patch("langchain_sqlserver.vectorstores.SQLServer_VectorStore._provide_token")
@mock.patch(
    "langchain_sqlserver.vectorstores.SQLServer_VectorStore._prepare_json_data_type"
)
def test_that_connection_string_with_trusted_connection_yes_does_not_use_entra_id_auth(
    prep_data_type: Mock,
    provide_token: Mock,
    dialect_initialize: Mock,
) -> None:
    """Test that if a connection string is provided to SQLServer_VectorStore object,
    and connection string has `trusted_connection` set to `yes`, entra id
    authentication is not used and connection is successful."""

    # Connection string is of the form below.
    # mssql+pyodbc://@lc-test.database.windows.net,1433/lcvectorstore
    # ?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    store = connect_to_vector_store(_CONNECTION_STRING)
    # _provide_token is called only during Entra ID authentication.
    provide_token.assert_not_called()
    store.drop()


def connect_to_vector_store(conn_string: str) -> SQLServer_VectorStore:
    return SQLServer_VectorStore(
        connection_string=conn_string,
        embedding_length=EMBEDDING_LENGTH,
        # DeterministicFakeEmbedding returns embeddings of the same
        # size as `embedding_length`.
        embedding_function=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
        table_name=_TABLE_NAME,
    )
