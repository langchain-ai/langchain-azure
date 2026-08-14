# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain-azure-ai",
#     "langchain-azure-storage[deepagents]",
#     "langchain[anthropic,openai]",
# ]
#
# [tool.uv.sources]
# langchain-azure-storage = { path = "../../libs/azure-storage", editable = true }
# ///
"""Basic example: a Deep Agent whose workspace persists in Azure Blob Storage.

The agent writes a file through its ``write_file`` tool; the backend persists
it as a blob. After the run, the script lists the workspace and prints where
each file physically lives, so you can see the durability for yourself.

Run from this directory (see README.md for environment setup):
    uv run --env-file .env basic_agent.py
"""

import asyncio

from _shared import build_backend, build_model, ensure_container
from deepagents import create_deep_agent

PREFIX = "session-001/"  # Isolates this session's files within the container.


async def main() -> None:
    """Run a Deep Agent, then show where its files landed in Azure."""
    container_url = ensure_container()

    # The async context manager releases the backend's cached async client
    # (and its aiohttp session) on exit.
    async with build_backend(PREFIX) as backend:
        agent = create_deep_agent(model=build_model(), backend=backend)

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
