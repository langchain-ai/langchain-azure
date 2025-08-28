import sys
from typing import Literal, TypedDict, Union

if sys.version_info < (3, 11):
    from typing_extensions import Required
else:
    from typing import Required

from psycopg import sql

ScalarType = str | int | float | bool


class FilterCondition(TypedDict, total=False):
    column: Required[str]
    operator: Required[
        Literal[
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
    ]
    value: ScalarType | list[ScalarType] | tuple[ScalarType, ScalarType]


class AndFilter(TypedDict):
    AND: list[Union["AndFilter", "OrFilter", FilterCondition]]


class OrFilter(TypedDict):
    OR: list[Union["AndFilter", "OrFilter", FilterCondition]]


# Define the top-level filter type
Filter = AndFilter | OrFilter | FilterCondition


def _filter_to_sql(
    filter: Filter | None,
    /,
    metadata_columns: list[str] | str = "metadata",
) -> sql.Composed | sql.SQL:
    """Convert a structured filter into a safe SQL expression.

    The filter DSL supports nested boolean logic via ``AND``/``OR`` and leaf
    conditions with an operator applied to a column. Columns can refer to
    configured metadata columns in two ways:

    - As a list of column names (``metadata_columns`` is a list[str]): the
    ``column`` in each condition must be one of these names. The column is
    rendered safely using identifiers.
    - As a single string (default ``"metadata"``): the ``column`` may instead be
    a raw SQL snippet such as ``(metadata->>'key')::int`` to target JSONB
    fields. In this mode the snippet is inserted verbatim; ensure it is trusted
    and validated by the caller.

    Supported operators
    -------------------
    - Comparison: ``=``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``like``, ``ilike``
    - Null checks: ``is null``, ``is not null`` (ignore ``value``)
    - Ranges: ``between``, ``not between`` (``value`` must be 2-tuple/list)
    - Membership: ``in``, ``not in`` (``value`` must be list/tuple)

    :param filter: Structured filter dict or ``None`` for no-op (always true).
    :type filter: Filter | None
    :param metadata_columns: Either a list of allowed metadata column names or a
        string indicating a single JSONB column name where callers may supply
        raw SQL paths/casts in the ``column`` field.
    :type metadata_columns: list[str] | str
    :return: A psycopg ``SQL``/``Composed`` object safe to use in queries.
    :rtype: sql.Composed | sql.SQL
    :raises ValueError: If required keys are missing, the column is not allowed
        (when a list is provided), or the operator/value shape is invalid.
    """
    if filter is None:
        # No filter, return a condition that always evaluates to true
        return sql.SQL("true")

    if "AND" in filter:
        conditions = [_filter_to_sql(cond) for cond in filter["AND"]]  # type: ignore[typeddict-item]
        return sql.SQL("").join(
            (sql.SQL("("), sql.SQL(" and ").join(conditions), sql.SQL(")"))
        )

    elif "OR" in filter:
        conditions = [_filter_to_sql(cond) for cond in filter["OR"]]  # type: ignore[typeddict-item]
        return sql.SQL("").join(
            (sql.SQL("("), sql.SQL(" or ").join(conditions), sql.SQL(")"))
        )

    else:
        column = filter.get("column")
        operator = filter.get("operator")

        if column is None or operator is None:
            raise ValueError("Filter must contain 'column' and 'operator' keys.")
        elif isinstance(metadata_columns, list) and column not in metadata_columns:
            raise ValueError(
                f"Column '{column}' is not in the list of metadata columns: {metadata_columns}"
            )

        value = filter.get("value")

        if operator in ["in", "not in"]:
            if isinstance(value, list | tuple):
                return sql.SQL("{column} {operator} ({value})").format(
                    column=sql.Identifier(column)
                    if isinstance(metadata_columns, list)
                    else sql.SQL(
                        column
                    ),  # to allow for (metadata->'key')::int style columns/keys
                    operator=sql.SQL(operator),
                    value=sql.SQL(", ").join(map(sql.Literal, value)),
                )
            else:
                raise ValueError("Value for 'in' or 'not in' must be a list or tuple.")
        elif operator in ["between", "not between"]:
            if isinstance(value, list | tuple) and len(value) == 2:
                return sql.SQL("{column} {operator} {lower} and {upper}").format(
                    column=sql.Identifier(column)
                    if isinstance(metadata_columns, list)
                    else sql.SQL(
                        column
                    ),  # to allow for (metadata->'key')::int style columns/keys
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
                column=sql.Identifier(column)
                if isinstance(metadata_columns, list)
                else sql.SQL(
                    column
                ),  # to allow for (metadata->'key')::int style columns/keys
                operator=sql.SQL(operator),
            )
        else:
            return sql.SQL("{} {} {}").format(
                column=sql.Identifier(column)
                if isinstance(metadata_columns, list)
                else sql.SQL(
                    column
                ),  # to allow for (metadata->'key')::int style columns/keys
                operator=sql.SQL(operator),
                value=sql.Literal(value) if value is not None else sql.NULL,
            )
