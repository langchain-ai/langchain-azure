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

from _shared import build_blob_backend, build_model, ensure_container
from deepagents import create_deep_agent

PREFIX = "research-session/"  # Both phases attach to this workspace.


async def phase_1_take_notes() -> None:
    """First agent lifetime: write notes into the workspace, then shut down."""
    async with build_blob_backend(PREFIX) as backend:
        agent = create_deep_agent(model=build_model(), backend=backend)
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
    async with build_blob_backend(PREFIX) as backend:
        agent = create_deep_agent(model=build_model(), backend=backend)
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

    print("=== Phase 1: an agent takes research notes ===")
    print(f"Writing into the '{PREFIX}' workspace, then discarding the agent.\n")
    await phase_1_take_notes()
    print("Phase 1 done: agent and backend discarded; notes persist in Azure.\n")

    print("=== Phase 2: a brand-new agent resumes the same workspace ===")
    print(
        "This agent was never told what Phase 1 wrote, and shares no in-memory\n"
        f"state with it. Everything it reports below it discovered by listing\n"
        f"and reading the blobs under '{PREFIX}'.\n"
    )
    await phase_2_resume()


if __name__ == "__main__":
    asyncio.run(main())
