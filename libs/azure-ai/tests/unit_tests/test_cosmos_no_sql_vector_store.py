"""Unit tests for AzureCosmosDBNoSqlVectorSearch field validation and projection."""

import pytest

from langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql import (
    AzureCosmosDBNoSqlVectorSearch,
    _validate_sql_identifier,
)


# ---------------------------------------------------------------------------
# _validate_sql_identifier – valid identifiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "page_content",
        "embedding",
        "my_metadata",
        "text",
        "_private",
        "field123",
        "A",
    ],
)
def test_validate_sql_identifier_valid(name: str) -> None:
    """Valid identifiers should not raise."""
    _validate_sql_identifier(name, "test_field")  # no exception expected


# ---------------------------------------------------------------------------
# _validate_sql_identifier – invalid identifier patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "my-field",      # hyphen
        "my field",      # space
        "123abc",        # starts with digit
        "field.name",    # dot
        "field@name",    # @
        "",              # empty string
    ],
)
def test_validate_sql_identifier_invalid_pattern(name: str) -> None:
    """Identifiers with invalid characters or patterns should raise ValueError."""
    with pytest.raises(ValueError, match="not a valid CosmosDB NoSQL identifier"):
        _validate_sql_identifier(name, "test_field")


# ---------------------------------------------------------------------------
# _validate_sql_identifier – reserved keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "SELECT",
        "select",
        "From",
        "WHERE",
        "NULL",
        "ORDER",
        "VALUE",
        "top",
    ],
)
def test_validate_sql_identifier_reserved_keyword(name: str) -> None:
    """Reserved keywords should raise ValueError regardless of case."""
    with pytest.raises(ValueError, match="reserved CosmosDB NoSQL keyword"):
        _validate_sql_identifier(name, "test_field")


# ---------------------------------------------------------------------------
# _generate_projection_fields – non-default metadata_key / embedding_field
# ---------------------------------------------------------------------------


def _make_store_stub(
    text_field: str = "text",
    embedding_field: str = "embedding",
    metadata_key: str = "metadata",
    table_alias: str = "c",
) -> object:
    """Return a minimal object that mimics the attributes used by
    _generate_projection_fields without constructing a full store."""

    class _Stub:
        _vector_search_fields = {
            "text_field": text_field,
            "embedding_field": embedding_field,
        }
        _metadata_key = metadata_key
        _table_alias = table_alias

        # Bind the method directly so we can call it without a real instance
        _generate_projection_fields = (
            AzureCosmosDBNoSqlVectorSearch._generate_projection_fields
        )

    return _Stub()


def test_projection_defaults_vector() -> None:
    """Default field names should appear verbatim in the projection alias."""
    stub = _make_store_stub()
    projection = stub._generate_projection_fields(None, "vector")
    assert "as text," in projection or "as text " in projection
    assert "as metadata" in projection
    assert "as SimilarityScore" in projection


def test_projection_custom_metadata_key() -> None:
    """Custom metadata_key should be used as the SQL alias for the metadata field."""
    stub = _make_store_stub(metadata_key="my_meta")
    projection = stub._generate_projection_fields(None, "vector")
    assert "as my_meta" in projection
    assert "as metadata" not in projection


def test_projection_custom_embedding_field_with_embedding() -> None:
    """Custom embedding_field should be used as the SQL alias when with_embedding=True."""
    stub = _make_store_stub(embedding_field="content_vector")
    projection = stub._generate_projection_fields(None, "vector", with_embedding=True)
    assert "as content_vector" in projection
    assert "as embedding" not in projection


def test_projection_custom_text_field() -> None:
    """Custom text_field should be used as the SQL alias for the text field."""
    stub = _make_store_stub(text_field="page_content")
    projection = stub._generate_projection_fields(None, "vector")
    assert "as page_content" in projection


def test_projection_hybrid_custom_fields() -> None:
    """Hybrid search with non-default fields should alias metadata and embedding correctly."""
    stub = _make_store_stub(
        text_field="page_content",
        embedding_field="content_vector",
        metadata_key="doc_meta",
    )
    projection = stub._generate_projection_fields(
        None, "hybrid", with_embedding=True
    )
    assert "as page_content" in projection
    assert "as doc_meta" in projection
    assert "as content_vector" in projection
    # Hardcoded names must NOT appear when custom names differ
    assert "as metadata" not in projection
    assert "as embedding" not in projection
