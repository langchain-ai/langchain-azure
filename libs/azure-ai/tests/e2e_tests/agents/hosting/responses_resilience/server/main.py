# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Deterministic ResponsesHostServer crash-recovery E2E server."""

from __future__ import annotations

import asyncio
import os

from azure.ai.agentserver.responses import ResponsesServerOptions

# Foundry executes this file from the server directory, where server_app is
# a top-level sibling package.
from server_app.crash_injection import (
    CrashInjectingResponsesHostServer,
)
from server_app.workflow import (  # pyright: ignore[reportMissingImports]
    build_graph,
    state_root,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def amain() -> None:
    """Run the deterministic resilient Responses host."""

    checkpoint_path = state_root() / "checkpoints.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        server = CrashInjectingResponsesHostServer(
            build_graph(checkpointer),
            options=ResponsesServerOptions(resilient_background=True),
        )
        await server.run_async(port=int(os.environ.get("PORT", "8088")))


if __name__ == "__main__":
    asyncio.run(amain())