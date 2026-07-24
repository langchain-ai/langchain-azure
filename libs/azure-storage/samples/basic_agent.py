# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain-anthropic",
#     "langchain-azure-storage[deepagents]",
# ]
#
# [tool.uv.sources]
# langchain-azure-storage = { path = "..", editable = true }
# ///
"""Basic example: a Deep Agent whose workspace persists in Azure Blob Storage.

Run from this directory:
    uv run --env-file .env basic_agent.py

Environment variables:
    ANTHROPIC_API_KEY                -- Required for the default Anthropic model.
    AZURE_STORAGE_CONNECTION_STRING  -- Connection string (e.g. Azurite).
                                        Takes precedence.
    AZURE_STORAGE_ACCOUNT_URL        -- Account URL for a live account;
                                        DefaultAzureCredential is used.
"""

import asyncio
import os

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from deepagents import create_deep_agent

from langchain_azure_storage.deepagents import AzureBlobBackend

CONTAINER_NAME = "agent-workspace"
PREFIX = "session-001/"  # Isolates this session's files within the container.


def build_backend() -> AzureBlobBackend:
    """Build a backend from environment variables."""
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        return AzureBlobBackend.from_connection_string(
            connection_string, CONTAINER_NAME, prefix=PREFIX
        )
    account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
    if not account_url:
        raise RuntimeError(
            "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL"
        )
    return AzureBlobBackend(
        account_url=account_url, container_name=CONTAINER_NAME, prefix=PREFIX
    )


def ensure_container() -> None:
    """Create the blob container if it doesn't exist (the backend won't)."""
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        client = BlobServiceClient.from_connection_string(connection_string)
    else:
        client = BlobServiceClient(
            os.environ["AZURE_STORAGE_ACCOUNT_URL"],
            credential=DefaultAzureCredential(),
        )
    with client:
        container = client.get_container_client(CONTAINER_NAME)
        if not container.exists():
            container.create_container()
            print(f"Created container '{CONTAINER_NAME}'")


async def main() -> None:
    """Run a Deep Agent against a blob-backed workspace."""
    ensure_container()

    # The async context manager releases the backend's cached async client
    # (and its aiohttp session) on exit; see the README's resource lifecycle
    # section.
    async with build_backend() as backend:
        agent = create_deep_agent(backend=backend)

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create a Python hello world script at /hello.py",
                    }
                ]
            }
        )
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
