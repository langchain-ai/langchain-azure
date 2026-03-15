"""Test binary collation behavior for the custom_id column."""

from unittest import mock
from unittest.mock import MagicMock

from langchain_sqlserver.vectorstores import BINARY_COLLATION, SQLServer_VectorStore
from tests.utils.fake_embeddings import DeterministicFakeEmbedding

_CONNECTION_STRING = (
    "mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server"
)
_TABLE_NAME = "test_table"
EMBEDDING_LENGTH = 1536


def _make_store(
    use_binary_collation_on_custom_id: bool = True,
) -> SQLServer_VectorStore:
    """Create a SQLServer_VectorStore with mocked DB interactions."""
    with (
        mock.patch(
            "langchain_sqlserver.vectorstores.SQLServer_VectorStore._create_engine",
            return_value=MagicMock(),
        ),
        mock.patch(
            "langchain_sqlserver.vectorstores.SQLServer_VectorStore._prepare_json_data_type"
        ),
        mock.patch(
            "langchain_sqlserver.vectorstores.SQLServer_VectorStore"
            "._create_table_if_not_exists"
        ),
    ):
        return SQLServer_VectorStore(
            connection_string=_CONNECTION_STRING,
            embedding_length=EMBEDDING_LENGTH,
            embedding_function=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
            table_name=_TABLE_NAME,
            use_binary_collation_on_custom_id=use_binary_collation_on_custom_id,
        )


def test_custom_id_uses_binary_collation_by_default() -> None:
    """Test that custom_id column uses binary collation when not opted out."""
    store = _make_store(use_binary_collation_on_custom_id=True)
    custom_id_col = store._embedding_store.__table__.c.custom_id
    assert custom_id_col.type.collation == BINARY_COLLATION


def test_custom_id_uses_binary_collation_when_not_specified() -> None:
    """Test that custom_id column uses binary collation when parameter is omitted."""
    store = _make_store()
    custom_id_col = store._embedding_store.__table__.c.custom_id
    assert custom_id_col.type.collation == BINARY_COLLATION


def test_custom_id_does_not_use_binary_collation_when_opted_out() -> None:
    """Test that custom_id column has no forced collation when opted out."""
    store = _make_store(use_binary_collation_on_custom_id=False)
    custom_id_col = store._embedding_store.__table__.c.custom_id
    assert custom_id_col.type.collation is None


def test_binary_collation_constant_value() -> None:
    """Test that the BINARY_COLLATION constant has the expected value."""
    assert BINARY_COLLATION == "Latin1_General_100_BIN2_UTF8"
