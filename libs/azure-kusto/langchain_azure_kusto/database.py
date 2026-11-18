"""Read-only connection to an Azure Data Explorer or Fabric Eventhouse database."""

from __future__ import annotations

import asyncio
import csv
import inspect
import io
import json
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from types import TracebackType
from typing import Any

import azure.identity
import azure.identity.aio
from azure.core.credentials import AccessToken, TokenCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.kusto.data import (
    ClientRequestProperties,
    KustoClient,
    KustoConnectionStringBuilder,
)
from azure.kusto.data.aio import KustoClient as AsyncKustoClient
from langchain_core._api import beta

from langchain_azure_kusto._tracing import CONNECTOR_NAME, CONNECTOR_VERSION

logger = logging.getLogger(__name__)

# KQL has only line comments, so a management command is the first `.` after
# any leading whitespace and `//` comment lines.
_MANAGEMENT_COMMAND = re.compile(r"\A\s*(?://[^\n]*(?:\n|\Z)\s*)*\.")

# Kusto entity names are letters, digits, `_`, ` `, `.` and `-`. Rejecting
# everything else, quotes and backslashes in particular, makes the bracket
# quoting below injection-safe without having to escape anything.
_TABLE_NAME = re.compile(r"\A[\w .\-]{1,1024}\Z")


class InvalidKustoInputError(ValueError):
    """Raised for a query or table name the caller (or agent) can correct."""


@dataclass(frozen=True)
class _ResultTable:
    columns: list[str]
    rows: list[list[Any]]


class _BorrowedCredential:
    """Lends a caller's sync credential to the SDK without handing over ownership.

    Closing a Kusto client also closes its credential. The caller owns any
    credential it passes in, so ``close`` is deliberately a no-op here.
    """

    def __init__(self, credential: TokenCredential) -> None:
        self._credential = credential

    def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        return self._credential.get_token(*scopes, **kwargs)

    def close(self) -> None:
        """Leave the caller's credential open."""


class _BorrowedAsyncCredential:
    """Async counterpart of :class:`_BorrowedCredential`."""

    def __init__(self, credential: AsyncTokenCredential) -> None:
        self._credential = credential

    async def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        return await self._credential.get_token(*scopes, **kwargs)

    async def close(self) -> None:
        """Leave the caller's credential open."""


def _is_async_credential(credential: object) -> bool:
    # `isinstance(..., TokenCredential)` is true for async credentials too, so
    # check how `get_token` is defined instead.
    return inspect.iscoroutinefunction(getattr(credential, "get_token", None))


def _validate_table_names(table_names: Sequence[str]) -> list[str]:
    names = [name.strip() for name in table_names if name.strip()]
    if not names:
        raise InvalidKustoInputError("Provide at least one table name.")
    invalid = [name for name in names if not _TABLE_NAME.match(name)]
    if invalid:
        raise InvalidKustoInputError(
            f"Invalid table name(s): {', '.join(invalid)}. Use names exactly as "
            "returned by the list-tables tool."
        )
    return names


def _check_read_only_query(query: str) -> None:
    if _MANAGEMENT_COMMAND.match(query):
        raise InvalidKustoInputError(
            "Management commands (starting with '.') are not allowed. "
            "Write a KQL query instead."
        )


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _format_table(table: _ResultTable, max_rows: int | None) -> str:
    rows = table.rows if max_rows is None else table.rows[:max_rows]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(table.columns)
    writer.writerows([_format_value(value) for value in row] for row in rows)
    text = buffer.getvalue().rstrip("\n")
    if not rows:
        text += "\n(no rows)"
    elif max_rows is not None and len(table.rows) > max_rows:
        text += (
            f"\n(Showing the first {max_rows} rows only. Filter or aggregate in "
            "the query to see the rest.)"
        )
    return text


@beta(
    message=(
        "`KustoDatabase` is in public preview. "
        "Its API is not stable and may change in future versions."
    )
)
class KustoDatabase:
    """A read-only connection to one Azure Data Explorer or Fabric Eventhouse database.

    Every query runs with the ``request_readonly`` request property, so it
    cannot change data, and management commands (those starting with ``.``) are
    rejected before they reach the service. Still, connect with an identity
    that has only the *Viewer* role on the database.

    The underlying Kusto clients are created on first use and reused. Close
    them with :meth:`close` / :meth:`aclose`, or use the object as a context
    manager. A credential passed in by the caller is never closed.

    Example:
        .. code-block:: python

            from langchain_azure_kusto import KustoDatabase

            db = KustoDatabase(
                cluster_uri="https://help.kusto.windows.net",
                database="Samples",
            )
            print(db.run("StormEvents | count"))
    """

    def __init__(
        self,
        cluster_uri: str,
        database: str,
        *,
        credential: TokenCredential | AsyncTokenCredential | None = None,
        max_rows: int = 100,
        sample_rows_in_table_info: int = 3,
        query_timeout: timedelta = timedelta(minutes=1),
    ) -> None:
        """Initialize `KustoDatabase`.

        Args:
            cluster_uri: Query URI of the cluster or Eventhouse, for example
                ``https://<cluster>.<region>.kusto.windows.net`` or
                ``https://<id>.<region>.kusto.fabric.microsoft.com``.
            database: Name of the database to query.
            credential: Credential used to sign in. A sync ``TokenCredential``
                works for both sync and async methods; an
                ``AsyncTokenCredential`` works for async methods only. If
                ``None``, ``DefaultAzureCredential`` is used.
            max_rows: Maximum number of rows :meth:`run` returns per result.
            sample_rows_in_table_info: Number of sample rows
                :meth:`get_table_info` includes per table. ``0`` disables them.
            query_timeout: Server-side timeout for each request.
        """
        if not cluster_uri.lower().startswith("https://"):
            raise ValueError(
                "cluster_uri must be an https:// URI, for example "
                "https://<cluster>.<region>.kusto.windows.net."
            )
        if not database:
            raise ValueError("database must not be empty.")
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1.")
        if sample_rows_in_table_info < 0:
            raise ValueError("sample_rows_in_table_info must not be negative.")

        self._cluster_uri = cluster_uri.rstrip("/")
        self._database = database
        self._credential = credential
        self._max_rows = max_rows
        self._sample_rows = sample_rows_in_table_info
        self._query_timeout = query_timeout
        self._client: KustoClient | None = None
        self._async_client: AsyncKustoClient | None = None
        self._async_client_loop: asyncio.AbstractEventLoop | None = None

    @property
    def cluster_uri(self) -> str:
        """Query URI of the cluster or Eventhouse."""
        return self._cluster_uri

    @property
    def database(self) -> str:
        """Name of the database."""
        return self._database

    @property
    def max_rows(self) -> int:
        """Maximum number of rows :meth:`run` returns per result."""
        return self._max_rows

    def __repr__(self) -> str:
        """Show the cluster and database, never the credential."""
        return (
            f"{type(self).__name__}(cluster_uri={self._cluster_uri!r}, "
            f"database={self._database!r})"
        )

    # Lifecycle

    def close(self) -> None:
        """Close the sync client, if one was created."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        """Close both the async and the sync client, if they were created."""
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None
            self._async_client_loop = None
        self.close()

    def __enter__(self) -> KustoDatabase:
        """Return this database."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the sync client."""
        self.close()

    async def __aenter__(self) -> KustoDatabase:
        """Return this database."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close both clients."""
        await self.aclose()

    # Public API

    def get_table_names(self) -> list[str]:
        """Return the names of the tables in the database."""
        response = self._get_client().execute_mgmt(
            self._database, ".show tables | project TableName", self._properties()
        )
        return [str(row[0]) for row in _tables(response)[0].rows]

    async def aget_table_names(self) -> list[str]:
        """Return the names of the tables in the database."""
        client = await self._get_async_client()
        response = await client.execute_mgmt(
            self._database, ".show tables | project TableName", self._properties()
        )
        return [str(row[0]) for row in _tables(response)[0].rows]

    def get_table_info(self, table_names: Sequence[str]) -> str:
        """Describe the columns of each table, followed by a few sample rows.

        Raises:
            InvalidKustoInputError: If a table name is malformed.
        """
        names = _validate_table_names(table_names)
        response = self._get_client().execute_query(
            self._database, self._table_info_query(names), self._properties()
        )
        return self._format_table_info(names, _tables(response))

    async def aget_table_info(self, table_names: Sequence[str]) -> str:
        """Describe the columns of each table, followed by a few sample rows.

        Raises:
            InvalidKustoInputError: If a table name is malformed.
        """
        names = _validate_table_names(table_names)
        response = await (await self._get_async_client()).execute_query(
            self._database, self._table_info_query(names), self._properties()
        )
        return self._format_table_info(names, _tables(response))

    def run(self, query: str) -> str:
        """Run a read-only KQL query and return the results as CSV text.

        At most ``max_rows`` rows are returned per result; a note is appended
        when rows were left out.

        Raises:
            InvalidKustoInputError: If the query is a management command.
        """
        tables = self._execute_query(query, max_rows=self._max_rows)
        return self._format_results(tables)

    async def arun(self, query: str) -> str:
        """Run a read-only KQL query and return the results as CSV text.

        At most ``max_rows`` rows are returned per result; a note is appended
        when rows were left out.

        Raises:
            InvalidKustoInputError: If the query is a management command.
        """
        tables = await self._aexecute_query(query, max_rows=self._max_rows)
        return self._format_results(tables)

    # Internals shared with the document loader

    def _execute_query(
        self, query: str, *, max_rows: int | None = None
    ) -> list[_ResultTable]:
        _check_read_only_query(query)
        response = self._get_client().execute_query(
            self._database, query, self._properties(read_only=True, max_rows=max_rows)
        )
        return _tables(response)

    async def _aexecute_query(
        self, query: str, *, max_rows: int | None = None
    ) -> list[_ResultTable]:
        _check_read_only_query(query)
        response = await (await self._get_async_client()).execute_query(
            self._database, query, self._properties(read_only=True, max_rows=max_rows)
        )
        return _tables(response)

    def _properties(
        self, *, read_only: bool = False, max_rows: int | None = None
    ) -> ClientRequestProperties:
        properties = ClientRequestProperties()
        properties.client_request_id = f"{CONNECTOR_NAME};{uuid.uuid4()}"
        properties.set_option(
            ClientRequestProperties.request_timeout_option_name, self._query_timeout
        )
        if read_only:
            properties.set_option("request_readonly", True)
        if max_rows is not None:
            # One extra row tells us whether the result was cut short.
            properties.set_option("query_take_max_records", max_rows + 1)
        return properties

    def _table_info_query(self, names: list[str]) -> str:
        statements = []
        for name in names:
            statements.append(
                f"['{name}'] | getschema | project ColumnName, ColumnType"
            )
            if self._sample_rows:
                statements.append(f"['{name}'] | take {self._sample_rows}")
        return ";\n".join(statements)

    def _format_table_info(self, names: list[str], tables: list[_ResultTable]) -> str:
        per_table = 2 if self._sample_rows else 1
        sections = []
        for index, name in enumerate(names):
            schema = tables[index * per_table]
            lines = [f"Table: {name}", "Columns:"]
            lines += [f"- {row[0]}: {row[1]}" for row in schema.rows]
            if self._sample_rows:
                sample = tables[index * per_table + 1]
                lines += ["Sample rows:", _format_table(sample, None)]
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def _format_results(self, tables: list[_ResultTable]) -> str:
        if len(tables) == 1:
            return _format_table(tables[0], self._max_rows)
        return "\n\n".join(
            f"Result {index}:\n{_format_table(table, self._max_rows)}"
            for index, table in enumerate(tables, start=1)
        )

    def _connection_string(self, credential: Any) -> KustoConnectionStringBuilder:
        kcsb = KustoConnectionStringBuilder.with_azure_token_credential(
            self._cluster_uri, credential=credential
        )
        kcsb._set_connector_details(CONNECTOR_NAME, CONNECTOR_VERSION)
        return kcsb

    def _get_client(self) -> KustoClient:
        if self._client is None:
            credential: Any
            if self._credential is None:
                credential = azure.identity.DefaultAzureCredential()
            elif _is_async_credential(self._credential):
                raise ValueError(
                    "Synchronous methods need a synchronous TokenCredential, but "
                    "KustoDatabase was given an AsyncTokenCredential. Use the "
                    "async methods instead, or pass a synchronous credential."
                )
            else:
                credential = _BorrowedCredential(self._credential)  # type: ignore[arg-type]
            self._client = KustoClient(self._connection_string(credential))
        return self._client

    async def _get_async_client(self) -> AsyncKustoClient:
        loop = asyncio.get_running_loop()
        if self._async_client is not None and self._async_client_loop is not loop:
            # The client's HTTP session is bound to the loop that created it, so
            # a new loop (for example a second `asyncio.run`) needs a new client.
            # Close the old one as far as is still possible from this loop.
            old_client, self._async_client = self._async_client, None
            try:
                await old_client.close()
            except Exception:
                logger.debug(
                    "Could not close the previous async Kusto client.", exc_info=True
                )
        if self._async_client is None:
            credential: Any
            if self._credential is None:
                credential = azure.identity.aio.DefaultAzureCredential()
            elif _is_async_credential(self._credential):
                credential = _BorrowedAsyncCredential(self._credential)  # type: ignore[arg-type]
            else:
                credential = _BorrowedCredential(self._credential)  # type: ignore[arg-type]
            self._async_client = AsyncKustoClient(self._connection_string(credential))
            self._async_client_loop = loop
        return self._async_client


def _tables(response: Any) -> list[_ResultTable]:
    return [
        _ResultTable(
            columns=[column.column_name for column in table.columns],
            rows=list(table.raw_rows),
        )
        for table in response.primary_results
    ]
