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

The agent writes a file through its ``write_file`` tool; the backend persists
it as a blob. After the run, the script lists the workspace and prints where
each file physically lives, so you can see the durability for yourself.

Run from this directory (see README.md for environment setup):
    uv run --env-file .env basic_agent.py
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
    """Build a backend from environment variables (see README.md)."""
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


def ensure_container() -> str:
    """Create the blob container if it doesn't exist; return its URL."""
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        service = BlobServiceClient.from_connection_string(connection_string)
    else:
        service = BlobServiceClient(
            os.environ["AZURE_STORAGE_ACCOUNT_URL"],
            credential=DefaultAzureCredential(),
        )
    with service:
        container = service.get_container_client(CONTAINER_NAME)
        if not container.exists():
            container.create_container()
        return str(container.url)


async def main() -> None:
    """Run a Deep Agent, then show where its files landed in Azure."""
    container_url = ensure_container()

    # The async context manager releases the backend's cached async client
    # (and its aiohttp session) on exit.
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

        # The workspace is durable: list what the agent left behind and where
        # each file physically lives in the container.
        listing = await backend.als("/")
        print("\nWorkspace contents:")
        for entry in listing.entries or []:
            blob_url = f"{container_url}/{PREFIX}{entry['path'].lstrip('/')}"
            print(f"  {entry['path']}  ->  {blob_url}")


if __name__ == "__main__":
    asyncio.run(main())
