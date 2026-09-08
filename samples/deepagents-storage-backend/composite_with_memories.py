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
"""Example: routing part of an agent's filesystem to Azure Blob Storage.

Deep Agents can split one filesystem across several backends with
[`CompositeBackend`](https://docs.langchain.com/oss/python/deepagents/backends#compositebackend-router),
choosing storage per path prefix. This sample routes two prefixes to Azure Blob
Storage and leaves everything else ephemeral:

    /memories/   -> Azure Blob Storage   long-term memory, survives every run
    /workspace/  -> Azure Blob Storage   shared working files, durable
    everything else -> StateBackend      thread-scoped, discarded after the run

That last line is the reason to use a composite here rather than pointing the
whole filesystem at Azure. Deep Agents writes its own bookkeeping into the
backend — offloaded large tool results under ``/large_tool_results/`` and
conversation history under ``/conversation_history/`` — and with a bare backend
those land in your container next to the agent's real output. Routing only the
prefixes you care about keeps the container clean.

The Azure-specific payoff: the main agent, the coder, and the tester all share
one durable ``/workspace/``, so the coder's files are immediately visible to the
tester — and to any later session that attaches to the same prefix. The run is
streamed so the output attributes each file operation to the agent that
performed it (see ``run_with_attribution``).

Run from this directory (see README.md for environment setup):
    uv run --env-file .env composite_with_memories.py
"""

import asyncio
import textwrap
from contextlib import AsyncExitStack
from typing import Any

from _shared import build_backend, build_model, ensure_container
from deepagents import SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol
from langchain_core.messages import AIMessageChunk
from langchain_core.messages.tool import ToolCall

# Blob prefixes backing each route. Both live in the samples' container.
MEMORIES_PREFIX = "composite-demo/memories/"
WORKSPACE_PREFIX = "composite-demo/workspace/"

MEMORY_PATH = "/memories/AGENTS.md"

COORDINATOR_PROMPT = (
    "You are a coordinator. Do not write code or tests yourself. Delegate all "
    "implementation to the 'coder' subagent and all test writing to the "
    "'tester' subagent using the task() tool. Call the coder first, then the "
    "tester once the source file exists."
)

# Past-tense labels for the filesystem tools worth showing in the trace.
_FILE_TOOLS = {"write_file": "wrote", "edit_file": "edited", "read_file": "read"}


async def seed_memory(backend: BackendProtocol) -> None:
    """Seed an AGENTS.md memory file with project conventions, if absent.

    Written through the composite backend, so ``MEMORY_PATH`` is the same path
    the agents see. The ``/memories/`` route sends it to Azure Blob Storage.

    The agents can rewrite this file themselves — memory is left writable so the
    workspace can accumulate conventions across sessions. Seeding is therefore
    guarded by a read: since deepagents 0.7.0, ``awrite`` replaces an existing
    file instead of refusing to, so writing unconditionally would wipe whatever
    the agents learned on every re-run.
    """
    if (await backend.aread(MEMORY_PATH)).error is None:
        return

    result = await backend.awrite(
        MEMORY_PATH,
        "# Project Conventions\n\n"
        "- Use snake_case for all Python identifiers\n"
        "- Include docstrings on every public function\n"
        "- Write pytest-style tests\n",
    )
    if result.error is not None:
        print(f"Warning: failed to seed AGENTS.md: {result.error}")


def _describe_tool_call(agent_name: str, call: ToolCall) -> str | None:
    """Render one tool call as a trace line, or ``None`` to skip it."""
    args = call.get("args") or {}
    if call["name"] == "task":
        subagent = args.get("subagent_type", "?")
        # Delegated instructions run long; the first line is enough.
        description = str(args.get("description", "")).strip().splitlines()
        summary = textwrap.shorten(description[0], 88) if description else ""
        return f"{agent_name} -> delegated to '{subagent}': {summary}"
    verb = _FILE_TOOLS.get(call["name"])
    if verb is None:
        return None
    return f"{agent_name} -> {verb} {args.get('file_path', '?')}"


async def run_with_attribution(agent: Any, prompt: str) -> tuple[str, list[str]]:
    """Run ``agent``, recording which agent performed each file operation.

    Streaming with ``subgraphs=True`` surfaces the subagents' own messages
    alongside the main agent's, and each message's metadata carries the
    ``lc_agent_name`` of the agent that produced it. That is what lets the trace
    say *which* agent wrote a file, rather than only what the main agent reports
    back at the end.

    Args:
        agent: The compiled Deep Agent to run.
        prompt: The user request to send.

    Returns:
        A ``(final_response, trace)`` tuple.
    """
    # Tool calls arrive split across chunks, so accumulate each message by id
    # and only read `tool_calls` once the stream is done.
    messages: dict[str, tuple[str, AIMessageChunk]] = {}
    final_state: dict[str, Any] = {}

    async for namespace, stream_mode, data in agent.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode=["messages", "values"],
        subgraphs=True,
    ):
        if stream_mode == "values":
            # An empty namespace is the main agent, whose final state holds the
            # response to the user.
            if not namespace:
                final_state = data
            continue

        chunk, metadata = data
        if not isinstance(chunk, AIMessageChunk) or chunk.id is None:
            continue
        agent_name = metadata.get("lc_agent_name") or "main agent"
        previous = messages.get(chunk.id)
        messages[chunk.id] = (
            agent_name,
            previous[1] + chunk if previous is not None else chunk,
        )

    trace = [
        line
        for agent_name, message in messages.values()
        for call in message.tool_calls
        if (line := _describe_tool_call(agent_name, call)) is not None
    ]
    response = final_state.get("messages", [])
    return (str(response[-1].content) if response else ""), trace


async def show_route(label: str, route: str, backend: Any) -> None:
    """Print the files a route persisted to Azure Blob Storage."""
    result = await backend.aglob("**/*")
    print(f"  {route} ({label})")
    for match in result.matches or []:
        print(f"    {route.rstrip('/')}{match['path']}")


async def main() -> None:
    """Run the composite agent, then show what each route persisted."""
    ensure_container()

    # Two backends, one per durable route. AsyncExitStack closes both.
    async with AsyncExitStack() as stack:
        memories = await stack.enter_async_context(build_backend(MEMORIES_PREFIX))
        workspace = await stack.enter_async_context(build_backend(WORKSPACE_PREFIX))

        backend = CompositeBackend(
            # Unrouted paths — including Deep Agents' own /large_tool_results/
            # and /conversation_history/ — stay in thread-scoped state.
            default=StateBackend(),
            routes={"/memories/": memories, "/workspace/": workspace},
        )

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
            model=build_model(),
            backend=backend,
            subagents=subagents,
            system_prompt=COORDINATOR_PROMPT,
            # Load project conventions from the AGENTS.md file seeded above.
            memory=[MEMORY_PATH],
        )

        response, trace = await run_with_attribution(
            agent,
            "Create a Python module at /workspace/src/calculator.py with add, "
            "subtract, multiply, and divide functions. Then write tests for it "
            "at /workspace/tests/test_calculator.py.",
        )

        print("Who did what (one shared workspace in Azure Blob Storage):")
        for line in trace:
            print(f"  {line}")

        print("\nPersisted to Azure Blob Storage:")
        await show_route("long-term memory", "/memories/", memories)
        await show_route("shared working files", "/workspace/", workspace)
        print(
            "\nEverything outside those routes — including Deep Agents' offloaded\n"
            "tool results and conversation history — stayed in thread-scoped state\n"
            "and never reached the container."
        )

        print(f"\n{response}")


if __name__ == "__main__":
    asyncio.run(main())
