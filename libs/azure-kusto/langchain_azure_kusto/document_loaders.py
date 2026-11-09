"""Load the results of a KQL query as LangChain documents."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_azure_kusto.database import KustoDatabase, _format_value, _ResultTable


class KustoLoader(BaseLoader):
    """Turn each row of a read-only KQL query result into a `Document`.

    The page content is one ``column: value`` line per content column. The
    metadata holds the raw values of the metadata columns. Management commands
    are rejected, and the query runs with the ``request_readonly`` property.

    All rows are fetched in a single request, so keep the result to a size that
    fits in memory, for example with ``take`` or a time filter.

    Example:
        .. code-block:: python

            from langchain_azure_kusto import KustoDatabase, KustoLoader

            db = KustoDatabase("https://help.kusto.windows.net", "Samples")
            loader = KustoLoader(
                "StormEvents | take 10",
                db,
                page_content_columns=["EpisodeNarrative"],
                metadata_columns=["State", "EventType", "StartTime"],
            )
            docs = loader.load()
    """

    def __init__(
        self,
        query: str,
        db: KustoDatabase,
        *,
        page_content_columns: Sequence[str] | None = None,
        metadata_columns: Sequence[str] | None = None,
    ) -> None:
        """Initialize `KustoLoader`.

        Args:
            query: Read-only KQL query whose rows become documents.
            db: Database to run the query against.
            page_content_columns: Columns that make up the page content. If
                ``None``, every column not in ``metadata_columns`` is used.
            metadata_columns: Columns copied into the document metadata.
        """
        self._query = query
        self._db = db
        self._page_content_columns = (
            None if page_content_columns is None else list(page_content_columns)
        )
        self._metadata_columns = list(metadata_columns or [])

    def lazy_load(self) -> Iterator[Document]:
        """Yield one document per result row.

        Raises:
            InvalidKustoInputError: If the query is a management command.
            ValueError: If a configured column is missing from the result.
        """
        for table in self._db._execute_query(self._query):
            yield from self._documents(table)

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Yield one document per result row.

        Raises:
            InvalidKustoInputError: If the query is a management command.
            ValueError: If a configured column is missing from the result.
        """
        for table in await self._db._aexecute_query(self._query):
            for document in self._documents(table):
                yield document

    def _documents(self, table: _ResultTable) -> Iterator[Document]:
        positions = {name: index for index, name in enumerate(table.columns)}
        content_columns = (
            [name for name in table.columns if name not in self._metadata_columns]
            if self._page_content_columns is None
            else self._page_content_columns
        )
        missing = [
            name
            for name in [*content_columns, *self._metadata_columns]
            if name not in positions
        ]
        if missing:
            raise ValueError(
                f"Column(s) not in the query result: {', '.join(missing)}. "
                f"Available columns: {', '.join(table.columns)}."
            )
        for row in table.rows:
            yield Document(
                page_content="\n".join(
                    f"{name}: {_format_value(row[positions[name]])}"
                    for name in content_columns
                ),
                metadata=_metadata(row, positions, self._metadata_columns),
            )


def _metadata(
    row: list[Any], positions: dict[str, int], columns: list[str]
) -> dict[str, Any]:
    return {name: row[positions[name]] for name in columns}
