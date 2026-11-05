from typing import Any, Literal, Required, TypedDict, Union, get_args

import numpy as np
from psycopg import sql

ScalarType = str | int | float | bool


def _embedding_to_numpy(embedding: Any | None) -> np.ndarray | None:
    """Return an embedding as a float32 NumPy array."""
    if embedding is None:
        return None
    if hasattr(embedding, "to_numpy"):
        embedding = embedding.to_numpy()
    return np.array(embedding, dtype=np.float32)


FilterOperator = Literal[
    "=",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "like",
    "ilike",
    "is null",
    "is not null",
    "between",
    "not between",
    "in",
    "not in",
]

FilterCast = Literal[
    "int", "bigint", "numeric", "float", "boolean", "date", "timestamptz", "text"
]

_FILTER_OPERATORS: frozenset[str] = frozenset(get_args(FilterOperator))

_FILTER_CASTS: dict[str, sql.SQL] = {
    "int": sql.SQL("integer"),
    "bigint": sql.SQL("bigint"),
    "numeric": sql.SQL("numeric"),
    "float": sql.SQL("double precision"),
    "boolean": sql.SQL("boolean"),
    "date": sql.SQL("date"),
    "timestamptz": sql.SQL("timestamptz"),
    "text": sql.SQL("text"),
}

# Substrings that indicate a caller passed a SQL expression (e.g., the
# ``(metadata->>'key')::int`` form accepted by earlier versions) rather than a
# plain JSONB key. Rejecting them makes such filters fail loudly instead of
# silently matching a key with that literal name.
_SQL_EXPRESSION_MARKERS = ("->", "#>", "::", "(", ")")


class FilterCondition(TypedDict, total=False):
    """A single filter condition.

    ``column`` is never inserted into the query as raw SQL when it is a
    ``str``: it names either a JSONB key (when the store's ``metadata_columns``
    is a string) or one of the configured metadata columns (when it is a list).
    To filter on a custom SQL expression, pass a trusted
    :class:`psycopg.sql.Composable` instead. ``operator`` must be one of the
    supported operators and ``cast`` one of the supported casts; both are
    validated at runtime. ``value`` is always passed as a SQL literal.
    """

    column: Required[str | sql.Composable]
    operator: Required[FilterOperator]
    value: ScalarType | list[ScalarType] | tuple[ScalarType, ScalarType]
    cast: FilterCast


class AndFilter(TypedDict):
    AND: list[Union["AndFilter", "OrFilter", FilterCondition]]


class OrFilter(TypedDict):
    OR: list[Union["AndFilter", "OrFilter", FilterCondition]]


# Define the top-level filter type
Filter = AndFilter | OrFilter | FilterCondition


def _filter_column_to_sql(
    column: Any, metadata_columns: list[str] | str | None
) -> sql.Composable:
    """Render a filter condition's ``column`` without treating strings as SQL."""
    if isinstance(column, sql.Composable):
        # Explicit escape hatch: the caller built this SQL themselves.
        return column
    if not isinstance(column, str):
        raise TypeError(
            "Filter 'column' must be a str or a psycopg sql.Composable, "
            f"got {type(column).__name__}."
        )
    if isinstance(metadata_columns, list):
        if column not in metadata_columns:
            raise ValueError(
                f"Column '{column}' is not in the list of metadata columns: {metadata_columns}"
            )
        return sql.Identifier(column)
    if isinstance(metadata_columns, str):
        if any(marker in column for marker in _SQL_EXPRESSION_MARKERS):
            raise ValueError(
                f"Filter column {column!r} looks like a SQL expression. A str column "
                f"names a key in the {metadata_columns!r} JSONB column; use the "
                "'cast' field for casts, or pass a psycopg sql.Composable for a "
                "trusted custom expression."
            )
        return sql.SQL("{metadata}->>{key}").format(
            metadata=sql.Identifier(metadata_columns), key=sql.Literal(column)
        )
    raise ValueError(
        "Filtering on a str column requires the store to have metadata_columns configured."
    )


def _filter_to_sql(
    filter: Filter | None,
    /,
    metadata_columns: list[str] | str | None = "metadata",
) -> sql.Composed | sql.SQL:
    """Convert a structured filter into a safe SQL expression.

    The filter DSL supports nested boolean logic via ``AND``/``OR`` and leaf
    conditions with an operator applied to a column. How a ``str`` column is
    rendered depends on ``metadata_columns``:

    - As a list of column names: the ``column`` in each condition must be one
      of these names and is rendered as a quoted identifier.
    - As a single string (default ``"metadata"``): the ``column`` is a key in
      that JSONB column and is rendered as ``"metadata"->>'key'``, with the key
      passed as a literal. Strings that look like SQL expressions are rejected.
    - As ``None``: ``str`` columns are rejected.

    In every mode, a :class:`psycopg.sql.Composable` column is used as-is. This
    is the escape hatch for custom expressions and must be built by trusted
    code, never from request data.

    An optional ``cast`` (``int``, ``bigint``, ``numeric``, ``float``,
    ``boolean``, ``date``, ``timestamptz``, ``text``) casts the column before
    comparison, e.g. so JSONB values compare numerically.

    Supported operators
    -------------------
    - Comparison: ``=``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``like``, ``ilike``
    - Null checks: ``is null``, ``is not null`` (ignore ``value``)
    - Ranges: ``between``, ``not between`` (``value`` must be 2-tuple/list)
    - Membership: ``in``, ``not in`` (``value`` must be list/tuple)

    :param filter: Structured filter dict or ``None`` for no-op (always true).
    :type filter: Filter | None
    :param metadata_columns: A list of allowed metadata column names, the name
        of a single JSONB metadata column, or ``None`` if metadata is disabled.
    :type metadata_columns: list[str] | str | None
    :return: A psycopg ``SQL``/``Composed`` object safe to use in queries.
    :rtype: sql.Composed | sql.SQL
    :raises TypeError: If the filter is not a dict or a column is neither a
        ``str`` nor a ``Composable``.
    :raises ValueError: If the filter is malformed, the column is not allowed
        or looks like a SQL expression, or the operator, cast, or value shape
        is invalid.
    """
    if filter is None:
        # No filter, return a condition that always evaluates to true
        return sql.SQL("true")

    if not isinstance(filter, dict):
        raise TypeError(f"Filter must be a dict, got {type(filter).__name__}.")

    if "AND" in filter or "OR" in filter:
        key = "AND" if "AND" in filter else "OR"
        branches = filter[key]  # type: ignore[literal-required]
        if not isinstance(branches, list | tuple):
            raise ValueError(f"Value for '{key}' must be a list or tuple.")
        conditions = [_filter_to_sql(cond, metadata_columns) for cond in branches]
        return sql.SQL("").join(
            (
                sql.SQL("("),
                sql.SQL(" and " if key == "AND" else " or ").join(conditions),
                sql.SQL(")"),
            )
        )

    column = filter.get("column")
    operator = filter.get("operator")

    if column is None or operator is None:
        raise ValueError("Filter must contain 'column' and 'operator' keys.")
    if not isinstance(operator, str) or operator not in _FILTER_OPERATORS:
        raise ValueError(
            f"Unsupported filter operator {operator!r}. Supported operators: "
            f"{sorted(_FILTER_OPERATORS)}"
        )

    column_sql = _filter_column_to_sql(column, metadata_columns)

    cast = filter.get("cast")
    if cast is not None:
        if not isinstance(cast, str) or cast not in _FILTER_CASTS:
            raise ValueError(
                f"Unsupported filter cast {cast!r}. Supported casts: "
                f"{sorted(_FILTER_CASTS)}"
            )
        column_sql = sql.SQL("({column})::{type}").format(
            column=column_sql, type=_FILTER_CASTS[cast]
        )

    value = filter.get("value")

    if operator in ["in", "not in"]:
        if isinstance(value, list | tuple):
            return sql.SQL("{column} {operator} ({value})").format(
                column=column_sql,
                operator=sql.SQL(operator),
                value=sql.SQL(", ").join(map(sql.Literal, value)),
            )
        else:
            raise ValueError("Value for 'in' or 'not in' must be a list or tuple.")
    elif operator in ["between", "not between"]:
        if isinstance(value, list | tuple) and len(value) == 2:
            return sql.SQL("{column} {operator} {lower} and {upper}").format(
                column=column_sql,
                operator=sql.SQL(operator),
                lower=sql.Literal(value[0]),
                upper=sql.Literal(value[1]),
            )
        else:
            raise ValueError(
                "Value for 'between' or 'not between' must be a list or tuple of two elements."
            )
    elif operator in ["is null", "is not null"]:
        return sql.SQL("{column} {operator}").format(
            column=column_sql,
            operator=sql.SQL(operator),
        )
    else:
        return sql.SQL("{column} {operator} {value}").format(
            column=column_sql,
            operator=sql.SQL(operator),
            value=sql.Literal(value) if value is not None else sql.NULL,
        )
