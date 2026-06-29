"""Shared partner user-agent string for Azure Storage SDK clients.

All Azure SDK clients created by this package tag their requests with a common
partner-attribution user agent so usage is consistent across the document
loaders and the Deep Agents backend.
"""

from langchain_azure_storage import __version__

USER_AGENT = f"azpartner-langchain/{__version__}"
"""Partner user-agent applied to every Azure SDK client in this package."""
