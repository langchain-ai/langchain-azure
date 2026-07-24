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
"""Example: composite agent with memory and subagents.

Demonstrates:
  - Loading memory from an AGENTS.md file via the ``memory`` parameter
  - Defining specialized subagents the main agent can delegate to
  - Sharing one Azure Blob Storage workspace across all agents

Run from this directory:
    uv run --env-file .env composite_with_memories.py

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
from deepagents import SubAgent, create_deep_agent

from langchain_azure_storage.deepagents import AzureBlobBackend

CONTAINER_NAME = "agent-workspace"
PREFIX = "composite-demo/"  # Isolates this demo's files within the container.


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


async def seed_memory(backend: AzureBlobBackend) -> None:
    """Seed an AGENTS.md memory file with project conventions.

    ``awrite`` fails if the file already exists; that's fine here, since an
    existing AGENTS.md is sufficient for this example.
    """
    result = await backend.awrite(
        "/.deepagents/AGENTS.md",
        "# Project Conventions\n\n"
        "- Use snake_case for all Python identifiers\n"
        "- Include docstrings on every public function\n"
        "- Write pytest-style tests\n",
    )
    if result.error is not None and "already exists" not in result.error:
        print(f"Warning: failed to seed AGENTS.md: {result.error}")


async def main() -> None:
    """Run the composite agent against a shared blob-backed workspace."""
    ensure_container()

    async with build_backend() as backend:
        await seed_memory(backend)

        subagents = [
            SubAgent(
                name="coder",
                description="Writes Python source code modules.",
                system_prompt=(
                    "You are a Python developer. Write clean, well-documented "
                    "code. Always include type hints and docstrings."
                ),
            ),
            SubAgent(
                name="tester",
                description="Writes pytest test files for existing code.",
                system_prompt=(
                    "You are a test engineer. Read the source code and write "
                    "comprehensive pytest tests. Cover edge cases and error "
                    "conditions."
                ),
            ),
        ]

        agent = create_deep_agent(
            backend=backend,
            subagents=subagents,
            # Load project conventions from the AGENTS.md file seeded above.
            memory=["/.deepagents/AGENTS.md"],
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Create a Python module at /src/calculator.py with "
                            "add, subtract, multiply, and divide functions. "
                            "Then write tests for it at "
                            "/tests/test_calculator.py."
                        ),
                    }
                ]
            }
        )
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
