"""Connector identity sent with every request, for usage attribution."""

from importlib import metadata

# The Kusto SDK sends this as `Kusto.LangChain:<version>` in the
# `x-ms-app` header, the same mechanism the other Kusto connectors use.
CONNECTOR_NAME = "LangChain"

try:
    CONNECTOR_VERSION = metadata.version("langchain-azure-kusto")
except metadata.PackageNotFoundError:  # pragma: no cover
    CONNECTOR_VERSION = "unknown"
