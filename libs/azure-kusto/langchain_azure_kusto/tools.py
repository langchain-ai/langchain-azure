"""Tools that let an agent explore and query a Kusto database."""

from __future__ import annotations

from azure.kusto.data.exceptions import (
    KustoMultiApiError,
    KustoNetworkError,
    KustoServiceError,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, BaseToolkit, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_azure_kusto.database import InvalidKustoInputError, KustoDatabase

# Service errors are returned to the agent so it can fix its query; keep them
# short enough not to flood the model's context.
_MAX_ERROR_CHARS = 2000


def _error_message(error: KustoServiceError) -> str:
    if isinstance(error, KustoMultiApiError):
        message = "\n".join(
            item.description or item.message or "" for item in error.get_api_errors()
        )
    else:
        message = error.message_text
    message = (message or str(error)).strip()
    return message[:_MAX_ERROR_CHARS]


def _to_tool_exception(error: Exception) -> Exception:
    """Map errors the agent can act on to `ToolException`; leave the rest alone.

    Bad input and query errors (syntax, unknown table, permission on a table)
    go back to the agent. Network failures are not something a different query
    fixes, so they propagate.
    """
    if isinstance(error, InvalidKustoInputError):
        return ToolException(str(error))
    if isinstance(error, KustoServiceError) and not isinstance(
        error, KustoNetworkError
    ):
        return ToolException(_error_message(error))
    return error


def _split_table_names(table_names: str) -> list[str]:
    return table_names.split(",")


class _NoInput(BaseModel):
    pass


class _TableNamesInput(BaseModel):
    table_names: str = Field(
        description=(
            "Comma-separated list of table names, for example: "
            "StormEvents, PopulationData"
        )
    )


class _QueryInput(BaseModel):
    query: str = Field(description="A read-only KQL (Kusto Query Language) query.")


class _BaseKustoTool(BaseTool):
    db: KustoDatabase = Field(exclude=True)
    """The database the tool queries."""

    handle_tool_error: bool = True
    """Return query errors to the agent so it can correct its query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class KustoListTablesTool(_BaseKustoTool):
    """List the tables in a Kusto database.

    Example:
        .. code-block:: python

            from langchain_azure_kusto import KustoDatabase, KustoListTablesTool

            db = KustoDatabase("https://help.kusto.windows.net", "Samples")
            print(KustoListTablesTool(db=db).invoke({}))
    """

    name: str = "kusto_list_tables"
    description: str = (
        "List the tables in the Kusto database as a comma-separated list. "
        "Call this first to find out which tables exist."
    )
    args_schema: type[BaseModel] = _NoInput

    def _run(self, run_manager: CallbackManagerForToolRun | None = None) -> str:
        try:
            return ", ".join(self.db.get_table_names())
        except Exception as error:
            raise _to_tool_exception(error) from error

    async def _arun(
        self, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> str:
        try:
            return ", ".join(await self.db.aget_table_names())
        except Exception as error:
            raise _to_tool_exception(error) from error


class KustoTableSchemaTool(_BaseKustoTool):
    """Describe the columns of Kusto tables, with a few sample rows.

    Example:
        .. code-block:: python

            from langchain_azure_kusto import KustoDatabase, KustoTableSchemaTool

            db = KustoDatabase("https://help.kusto.windows.net", "Samples")
            tool = KustoTableSchemaTool(db=db)
            print(tool.invoke({"table_names": "StormEvents"}))
    """

    name: str = "kusto_table_schema"
    description: str = (
        "Get the column names, column types and a few sample rows of one or "
        "more tables. Input is a comma-separated list of table names. Call "
        "kusto_list_tables first to get valid names."
    )
    args_schema: type[BaseModel] = _TableNamesInput

    def _run(
        self,
        table_names: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            return self.db.get_table_info(_split_table_names(table_names))
        except Exception as error:
            raise _to_tool_exception(error) from error

    async def _arun(
        self,
        table_names: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            return await self.db.aget_table_info(_split_table_names(table_names))
        except Exception as error:
            raise _to_tool_exception(error) from error


class KustoQueryTool(_BaseKustoTool):
    """Run a read-only KQL query and return the results as CSV text.

    Example:
        .. code-block:: python

            from langchain_azure_kusto import KustoDatabase, KustoQueryTool

            db = KustoDatabase("https://help.kusto.windows.net", "Samples")
            tool = KustoQueryTool(db=db)
            print(tool.invoke({"query": "StormEvents | count"}))
    """

    name: str = "kusto_query"
    description: str = (
        "Run a read-only KQL (Kusto Query Language) query against the database "
        "and get the results as CSV. Write KQL, not SQL: start with a table name "
        "and chain operators with pipes, for example "
        "`StormEvents | where State == 'TEXAS' | summarize count() by EventType "
        "| top 5 by count_`. Only a limited number of rows is returned, so "
        "filter and aggregate in the query. Management commands (starting with "
        "'.') are not allowed. Call kusto_table_schema first to learn the "
        "columns. If the query fails, read the error, fix the query and retry."
    )
    args_schema: type[BaseModel] = _QueryInput

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            return self.db.run(query)
        except Exception as error:
            raise _to_tool_exception(error) from error

    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            return await self.db.arun(query)
        except Exception as error:
            raise _to_tool_exception(error) from error


class KustoToolkit(BaseToolkit):
    """Tools for an agent to explore and query one Kusto database.

    Returns :class:`KustoListTablesTool`, :class:`KustoTableSchemaTool` and
    :class:`KustoQueryTool`, all sharing the same :class:`KustoDatabase`.

    Example:
        .. code-block:: python

            from langchain.agents import create_agent

            from langchain_azure_kusto import KustoDatabase, KustoToolkit

            db = KustoDatabase("https://help.kusto.windows.net", "Samples")
            agent = create_agent(model, tools=KustoToolkit(db=db).get_tools())
    """

    db: KustoDatabase = Field(exclude=True)
    """The database the tools query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_tools(self) -> list[BaseTool]:
        """Return the list tables, table schema and query tools."""
        return [
            KustoListTablesTool(db=self.db),
            KustoTableSchemaTool(db=self.db),
            KustoQueryTool(db=self.db),
        ]
