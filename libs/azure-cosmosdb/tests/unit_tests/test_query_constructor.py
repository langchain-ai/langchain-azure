"""Unit tests for AzureCosmosDbNoSQLTranslator."""

import pytest
from langchain_azure_cosmosdb._query_constructor import (
    AzureCosmosDbNoSQLTranslator,
)
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)


@pytest.fixture()
def translator() -> AzureCosmosDbNoSQLTranslator:
    return AzureCosmosDbNoSQLTranslator(table_name="c")


# ---------------------------------------------------------------------------
# visit_comparison – standard comparators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "comparator, sql_op",
    [
        (Comparator.EQ, "="),
        (Comparator.NE, "!="),
        (Comparator.GT, ">"),
        (Comparator.GTE, ">="),
        (Comparator.LT, "<"),
        (Comparator.LTE, "<="),
        (Comparator.LIKE, "LIKE"),
    ],
)
def test_visit_comparison_numeric(
    translator: AzureCosmosDbNoSQLTranslator,
    comparator: Comparator,
    sql_op: str,
) -> None:
    comp = Comparison(comparator=comparator, attribute="age", value=30)
    result = translator.visit_comparison(comp)
    assert result == f"c.age {sql_op} 30"


def test_visit_comparison_string_quoted(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    comp = Comparison(comparator=Comparator.EQ, attribute="name", value="Alice")
    result = translator.visit_comparison(comp)
    assert result == "c.name = 'Alice'"


# ---------------------------------------------------------------------------
# visit_comparison – IN / NIN with list values
# ---------------------------------------------------------------------------


def test_visit_comparison_in_list(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    comp = Comparison(
        comparator=Comparator.IN,
        attribute="color",
        value=["red", "blue"],
    )
    result = translator.visit_comparison(comp)
    assert result == "c.color IN ('red', 'blue')"


def test_visit_comparison_nin_list(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    comp = Comparison(
        comparator=Comparator.NIN,
        attribute="color",
        value=["red", "blue"],
    )
    result = translator.visit_comparison(comp)
    assert result == "c.color NOT IN ('red', 'blue')"


def test_visit_comparison_in_with_int_list(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    comp = Comparison(
        comparator=Comparator.IN,
        attribute="id",
        value=[1, 2, 3],
    )
    result = translator.visit_comparison(comp)
    assert result == "c.id IN (1, 2, 3)"


# ---------------------------------------------------------------------------
# visit_comparison – unsupported operator
# ---------------------------------------------------------------------------


def test_visit_comparison_unsupported_operator(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    comp = Comparison(comparator=Comparator.CONTAIN, attribute="x", value="y")
    with pytest.raises(ValueError, match="Unsupported operator"):
        translator.visit_comparison(comp)


# ---------------------------------------------------------------------------
# visit_operation – AND, OR, NOT
# ---------------------------------------------------------------------------


def test_visit_operation_and(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="a", value=1),
            Comparison(comparator=Comparator.EQ, attribute="b", value=2),
        ],
    )
    result = translator.visit_operation(op)
    assert result == "(c.a = 1 AND c.b = 2)"


def test_visit_operation_or(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    op = Operation(
        operator=Operator.OR,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="a", value=1),
            Comparison(comparator=Comparator.EQ, attribute="b", value=2),
        ],
    )
    result = translator.visit_operation(op)
    assert result == "(c.a = 1 OR c.b = 2)"


def test_visit_operation_not(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    op = Operation(
        operator=Operator.NOT,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="a", value=1),
        ],
    )
    result = translator.visit_operation(op)
    assert result == "NOT (c.a = 1)"


# ---------------------------------------------------------------------------
# visit_structured_query
# ---------------------------------------------------------------------------


def test_visit_structured_query_with_filter(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    sq = StructuredQuery(
        query="find documents",
        filter=Comparison(comparator=Comparator.EQ, attribute="status", value="active"),
    )
    query_str, kwargs = translator.visit_structured_query(sq)
    assert query_str == "find documents"
    assert kwargs == {"where": "c.status = 'active'"}


def test_visit_structured_query_without_filter(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    sq = StructuredQuery(query="find all", filter=None)
    query_str, kwargs = translator.visit_structured_query(sq)
    assert query_str == "find all"
    assert kwargs == {}


# ---------------------------------------------------------------------------
# None value handling
# ---------------------------------------------------------------------------


def test_none_eq_emits_is_null(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    comp = Comparison(comparator=Comparator.EQ, attribute="status", value=None)
    result = translator.visit_comparison(comp)
    assert result == "IS_NULL(c.status)"


def test_none_ne_emits_is_not_null(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    comp = Comparison(comparator=Comparator.NE, attribute="status", value=None)
    result = translator.visit_comparison(comp)
    assert result == "NOT IS_NULL(c.status)"


def test_none_gt_raises(
    translator: AzureCosmosDbNoSQLTranslator,
) -> None:
    comp = Comparison(comparator=Comparator.GT, attribute="age", value=None)
    with pytest.raises(ValueError, match="Cannot use comparator"):
        translator.visit_comparison(comp)
