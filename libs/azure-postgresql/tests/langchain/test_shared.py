"""Unit tests for the structured filter to SQL conversion."""

from typing import Any

import pytest
from psycopg import sql

from langchain_azure_postgresql.langchain._shared import _filter_to_sql


def _compose(filter: Any, metadata_columns: Any = "metadata") -> sql.Composable:
    return _filter_to_sql(filter, metadata_columns)


def _render(filter: Any, metadata_columns: Any = "metadata") -> str:
    return _compose(filter, metadata_columns).as_string(None)


class TestFilterToSQL:
    def test_no_filter(self):
        assert _render(None) == "true"

    @pytest.mark.parametrize(
        ("filter", "expected"),
        [
            (
                {"column": "tenant", "operator": "=", "value": "a"},
                "\"metadata\"->>'tenant' = 'a'",
            ),
            (
                {"column": "tenant", "operator": "!=", "value": "a"},
                "\"metadata\"->>'tenant' != 'a'",
            ),
            (
                {"column": "title", "operator": "ilike", "value": "%x%"},
                "\"metadata\"->>'title' ilike '%x%'",
            ),
            (
                {"column": "tenant", "operator": "=", "value": None},
                "\"metadata\"->>'tenant' = NULL",
            ),
            (
                {"column": "tenant", "operator": "in", "value": ["a", "b"]},
                "\"metadata\"->>'tenant' in ('a', 'b')",
            ),
            (
                {"column": "tenant", "operator": "not in", "value": ("a",)},
                "\"metadata\"->>'tenant' not in ('a')",
            ),
            (
                {"column": "age", "operator": "between", "value": [1, 5]},
                "\"metadata\"->>'age' between 1 and 5",
            ),
            (
                {"column": "age", "operator": "is null"},
                "\"metadata\"->>'age' is null",
            ),
            (
                {"column": "age", "operator": "is not null"},
                "\"metadata\"->>'age' is not null",
            ),
        ],
    )
    def test_jsonb_key_operators(self, filter: Any, expected: str):
        assert _render(filter) == expected

    def test_jsonb_key_is_literal(self):
        # Quotes in the key are escaped as part of a literal, not spliced in.
        filter = {"column": "a' or 1=1 --", "operator": "is null"}
        assert _render(filter) == "\"metadata\"->>'a'' or 1=1 --' is null"

    def test_custom_jsonb_column_is_identifier(self):
        filter = {"column": "k", "operator": "=", "value": 1}
        assert _render(filter, 'meta"data') == '"meta""data"->>\'k\' = 1'

    @pytest.mark.parametrize(
        ("cast", "expected_type"),
        [
            ("int", "integer"),
            ("bigint", "bigint"),
            ("numeric", "numeric"),
            ("float", "double precision"),
            ("boolean", "boolean"),
            ("date", "date"),
            ("timestamptz", "timestamptz"),
            ("text", "text"),
        ],
    )
    def test_cast(self, cast: str, expected_type: str):
        filter = {"column": "age", "operator": ">", "value": 3, "cast": cast}
        assert _render(filter) == f"(\"metadata\"->>'age')::{expected_type} > 3"

    def test_list_mode_renders_identifier(self):
        filter = {"column": "tenant", "operator": "in", "value": ["a"]}
        assert _render(filter, ["tenant", "source"]) == "\"tenant\" in ('a')"

    def test_list_mode_with_cast(self):
        filter = {"column": "age", "operator": "<=", "value": 3, "cast": "int"}
        assert _render(filter, ["age"]) == '("age")::integer <= 3'

    def test_nested_filters(self):
        filter = {
            "AND": [
                {"column": "tenant", "operator": "=", "value": "a"},
                {
                    "OR": [
                        {"column": "source", "operator": "is null"},
                        {"column": "source", "operator": "in", "value": ["x"]},
                    ]
                },
            ]
        }
        assert _render(filter, ["tenant", "source"]) == (
            '("tenant" = \'a\' and ("source" is null or "source" in (\'x\')))'
        )

    def test_composable_column_escape_hatch(self):
        column = sql.SQL("lower({})").format(sql.Identifier("content"))
        filter = {"column": column, "operator": "like", "value": "a%"}
        assert _render(filter) == "lower(\"content\") like 'a%'"
        assert _render(filter, ["tenant"]) == "lower(\"content\") like 'a%'"


class TestFilterToSQLRejects:
    @pytest.mark.parametrize(
        "operator",
        [
            "in (select 1) or true or 1 in",
            "= 'a' or true or 1 =",
            "IN",
            "<>",
            ["in"],
            1,
        ],
    )
    @pytest.mark.parametrize("metadata_columns", ["metadata", ["tenant"]])
    def test_operator_not_in_allowlist(self, operator: Any, metadata_columns: Any):
        filter = {"column": "tenant", "operator": operator, "value": ["a"]}
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            _compose(filter, metadata_columns)

    @pytest.mark.parametrize(
        "column",
        [
            "1=1 OR metadata->>'tenant'",
            "metadata->>'tenant'",
            "(metadata->>'age')::int",
            "metadata#>>'{a,b}'",
            "true or (select 1)",
        ],
    )
    def test_sql_expression_column_in_string_mode(self, column: str):
        filter = {"column": column, "operator": "in", "value": ["a"]}
        with pytest.raises(ValueError, match="looks like a SQL expression"):
            _compose(filter)

    def test_column_not_in_allowlist(self):
        filter = {"column": "secret", "operator": "is null"}
        with pytest.raises(ValueError, match="not in the list of metadata columns"):
            _compose(filter, ["tenant"])

    @pytest.mark.parametrize("branch", ["AND", "OR"])
    def test_allowlist_enforced_in_nested_branches(self, branch: str):
        filter = {
            branch: [
                {"column": "tenant", "operator": "is null"},
                {"AND": [{"column": "1=1 or tenant", "operator": "is null"}]},
            ]
        }
        with pytest.raises(ValueError, match="not in the list of metadata columns"):
            _compose(filter, ["tenant"])

    def test_str_column_without_metadata_columns(self):
        filter = {"column": "tenant", "operator": "is null"}
        with pytest.raises(ValueError, match="requires the store to have"):
            _compose(filter, None)

    @pytest.mark.parametrize("column", [1, ["tenant"], {"a": 1}])
    def test_column_wrong_type(self, column: Any):
        filter = {"column": column, "operator": "is null"}
        with pytest.raises(TypeError, match="must be a str or a psycopg"):
            _compose(filter)

    @pytest.mark.parametrize("cast", ["int; drop table x", "integer", ["int"]])
    def test_cast_not_in_allowlist(self, cast: Any):
        filter = {"column": "age", "operator": "=", "value": 1, "cast": cast}
        with pytest.raises(ValueError, match="Unsupported filter cast"):
            _compose(filter)

    @pytest.mark.parametrize(
        "filter",
        [
            {"operator": "is null"},
            {"column": "tenant"},
            {"AND": "tenant"},
        ],
    )
    def test_malformed_filter(self, filter: Any):
        with pytest.raises(ValueError):
            _compose(filter)

    def test_filter_not_a_dict(self):
        with pytest.raises(TypeError, match="Filter must be a dict"):
            _compose("tenant = 'a'")
