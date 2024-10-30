"""Test SQLServer_VectorStore functionality."""

import os
from unittest import mock
from unittest.mock import Mock

import pytest

from langchain_sqlserver.vectorstores import SQLServer_VectorStore
from tests.utils.fake_embeddings import DeterministicFakeEmbedding

pytest.skip(
    "Skipping these tests pending resource availability", allow_module_level=True
)

# Connection String values should be provided in the
# environment running this test suite.
#
_CONNECTION_STRING_WITH_UID_AND_PWD = str(
    os.environ.get("TEST_AZURESQLSERVER_CONNECTION_STRING_WITH_UID")
)
_CONNECTION_STRING_WITH_TRUSTED_CONNECTION = str(
    os.environ.get("TEST_AZURESQLSERVER_TRUSTED_CONNECTION")
)
_ENTRA_ID_CONNECTION_STRING_NO_PARAMS = str(
    os.environ.get("TEST_ENTRA_ID_CONNECTION_STRING_NO_PARAMS")
)
_ENTRA_ID_CONNECTION_STRING_TRUSTED_CONNECTION_NO = str(
    os.environ.get("TEST_ENTRA_ID_CONNECTION_STRING_TRUSTED_CONNECTION_NO")
)
_TABLE_NAME = "langchain_vector_store_tests"
EMBEDDING_LENGTH = 1536


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
    store = connect_to_vector_store(_CONNECTION_STRING_WITH_TRUSTED_CONNECTION)
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
