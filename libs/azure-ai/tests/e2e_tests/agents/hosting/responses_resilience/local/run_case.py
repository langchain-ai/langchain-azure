# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Execute a shared resilience case against local server processes."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import BinaryIO

import pytest

from tests.e2e_tests.agents.hosting.responses_resilience.common.case_definitions import (  # noqa: E501
    RECOVERY_CASES,
)
from tests.e2e_tests.agents.hosting.responses_resilience.common.cases import (
    ResilienceCase,
    assert_case_outcome,
    assert_setup_outcome,
    assert_verification_outcome,
)
from tests.e2e_tests.agents.hosting.responses_resilience.common.responses_client import (  # noqa: E501
    create_openai_client,
    retrieve_stored_response,
    run_resilient_turn,
)
from tests.e2e_tests.agents.hosting.responses_resilience.local.process import (
    clear_stale_stream_locks,
    free_port,
    spawn_server,
    stop_process,
    wait_exit,
    wait_ready,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app import (
    crash_points,
)


async def run_local_case(
    case: ResilienceCase,
    tmp_path: Path,
) -> None:
    """Run one case with local-only process lifecycle assertions."""

    port = free_port()
    endpoint = f"http://127.0.0.1:{port}/responses"
    process: subprocess.Popen[bytes] | None = None
    client = None
    logs: list[BinaryIO] = []
    conversation_id = f"{case.name}-e2e"
    try:
        process, log = spawn_server(tmp_path, port, lifetime=0)
        logs.append(log)
        await wait_ready(process, port)
        client = create_openai_client(endpoint, "local")
        if case.setup_input_text is not None:
            setup_outcome = await run_resilient_turn(
                client,
                case.setup_input_text,
                conversation_id=conversation_id,
                reconnect_delay=0.1,
                reconnect_timeout=60.0,
            )
            assert_setup_outcome(case, setup_outcome)

        outcome_task = asyncio.create_task(
            run_resilient_turn(
                client,
                case.input_text,
                conversation_id=conversation_id,
                reconnect_delay=0.1,
                reconnect_timeout=60.0,
            )
        )

        assert await wait_exit(process) == crash_points.CRASH_EXIT_CODE
        clear_stale_stream_locks(tmp_path)
        logs[-1].close()
        process, log = spawn_server(tmp_path, port, lifetime=1)
        logs.append(log)
        await wait_ready(process, port)

        outcome = await outcome_task
        assert_case_outcome(case, outcome)
        if case.verify_terminal_retrieval:
            stored = await retrieve_stored_response(client, outcome.response_id)
            assert stored.status == outcome.status
            assert stored.output_text == outcome.output_text
        if case.verification_input_text is not None:
            verification_outcome = await run_resilient_turn(
                client,
                case.verification_input_text,
                conversation_id=conversation_id,
                reconnect_delay=0.1,
                reconnect_timeout=60.0,
            )
            assert_verification_outcome(case, verification_outcome)
    finally:
        if client is not None:
            await client.close()
        stop_process(process)
        for log in logs:
            if not log.closed:
                log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", RECOVERY_CASES, ids=lambda case: case.name)
async def test_resilience_case(case: ResilienceCase, tmp_path: Path) -> None:
    await run_local_case(case, tmp_path)