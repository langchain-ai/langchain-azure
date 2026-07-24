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
"""Example: workspace persistence across agent lifetimes.

This is the demo only a durable backend can run. Phase 1 creates an agent,
has it write research notes, and tears the agent and backend down completely.
Phase 2 constructs a brand-new backend and agent on the same prefix — the new
agent finds and summarizes the notes, because the workspace lives in Azure
Blob Storage rather than in process memory. In real use the two phases would
be separate runs, days apart, or on different machines.

Run from this directory (see README.md for environment setup):
    uv run --env-file .env resume_workspace.py
"""

import asyncio
import os

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from deepagents import create_deep_agent

from langchain_azure_storage.deepagents import AzureBlobBackend

CONTAINER_NAME = "agent-workspace"
PREFIX = "research-session/"  # Both phases attach to this workspace.


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


async def phase_1_take_notes() -> None:
    """First agent lifetime: write notes into the workspace, then shut down."""
    async with build_backend() as backend:
        agent = create_deep_agent(backend=backend)
        await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Write a file /notes/observations.md with three "
                            "bullet points on the benefits of durable agent "
                            "workspaces. If the file already exists, update it."
                        ),
                    }
                ]
            }
        )
    # Leaving the block closes the backend: nothing about the workspace
    # remains in this process.


async def phase_2_resume() -> None:
    """Second agent lifetime: a fresh backend and agent on the same prefix."""
    async with build_backend() as backend:
        agent = create_deep_agent(backend=backend)
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Look through the files in your workspace and "
                            "summarize what you find."
                        ),
                    }
                ]
            }
        )
        print(result["messages"][-1].content)


async def main() -> None:
    """Prove the workspace outlives the agent that created it."""
    ensure_container()

    await phase_1_take_notes()
    print("Phase 1 done: agent and backend discarded; notes persist in Azure.\n")

    await phase_2_resume()


if __name__ == "__main__":
    asyncio.run(main())
