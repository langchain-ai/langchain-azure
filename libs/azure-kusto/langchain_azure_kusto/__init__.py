"""LangChain integrations for Azure Data Explorer and Microsoft Fabric Eventhouse.

Both services run the Kusto engine and are queried with KQL (Kusto Query
Language), so one integration covers both:

- :class:`KustoDatabase` -- a read-only connection to one database.
- :class:`KustoToolkit` -- tools that let an agent list tables, inspect their
  schema, and run read-only KQL queries.
- :class:`KustoLoader` -- turns the rows of a KQL query into LangChain
  ``Document`` objects.
"""

from importlib import metadata

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:  # pragma: no cover
    # Only when running from a source tree that was never installed.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

from langchain_azure_kusto.database import (  # noqa: E402
    InvalidKustoInputError,
    KustoDatabase,
)
from langchain_azure_kusto.document_loaders import KustoLoader  # noqa: E402
from langchain_azure_kusto.tools import (  # noqa: E402
    KustoListTablesTool,
    KustoQueryTool,
    KustoTableSchemaTool,
    KustoToolkit,
)

__all__ = [
    "InvalidKustoInputError",
    "KustoDatabase",
    "KustoListTablesTool",
    "KustoLoader",
    "KustoQueryTool",
    "KustoTableSchemaTool",
    "KustoToolkit",
    "__version__",
]
