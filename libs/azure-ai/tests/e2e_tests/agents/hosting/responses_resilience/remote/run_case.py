# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run the deterministic recovery case against a Foundry-hosted server."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

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
    final_result,
    retrieve_stored_response,
    run_resilient_turn,
)

AZURE_AI_SCOPE = "https://ai.azure.com/.default"
RECOVERY_CASES_BY_CRASH_POINT = {case.crash_point: case for case in RECOVERY_CASES}


async def run_case(
    endpoint: str,
    *,
    api_key: str | Any,
    case: ResilienceCase,
    reconnect_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Run one Foundry case using the shared resilient Responses client."""

    async with create_openai_client(endpoint, api_key) as client:
        conversation_id: str | None = None
        setup_response_id: str | None = None
        conversation = await client.conversations.create()
        conversation_id = conversation.id
        if case.setup_input_text is not None:
            setup_turn = await run_resilient_turn(
                client,
                case.setup_input_text,
                conversation_id=conversation_id,
                reconnect_timeout=reconnect_timeout_seconds,
            )
            assert_setup_outcome(case, setup_turn)
            setup_response_id = setup_turn.response_id

        turn = await run_resilient_turn(
            client,
            case.input_text,
            conversation_id=conversation_id,
            reconnect_timeout=reconnect_timeout_seconds,
        )
        assert_case_outcome(case, turn)
        terminal_retrieval_verified = False
        if case.verify_terminal_retrieval:
            stored = await retrieve_stored_response(
                client,
                turn.response_id,
                timeout=reconnect_timeout_seconds,
            )
            assert stored.status == turn.status
            assert stored.output_text == turn.output_text
            terminal_retrieval_verified = True
        verification_response_id: str | None = None
        if case.verification_input_text is not None:
            verification_turn = await run_resilient_turn(
                client,
                case.verification_input_text,
                conversation_id=conversation_id,
                reconnect_timeout=reconnect_timeout_seconds,
            )
            assert_verification_outcome(case, verification_turn)
            verification_response_id = verification_turn.response_id
    result = final_result(turn.output_text)
    return {
        "crash_point": case.crash_point,
        "conversation_id": conversation_id,
        "setup_response_id": setup_response_id,
        "verification_response_id": verification_response_id,
        "terminal_retrieval_verified": terminal_retrieval_verified,
        "recovery_started_seconds": turn.recovery_started_seconds,
        "requested_response_id": turn.requested_response_id,
        "response_id": turn.response_id,
        "response_status": turn.status,
        "pre_recovery_output_text": turn.pre_recovery_output_text,
        "pre_reset_output_text": turn.pre_reset_output_text,
        "output_text": turn.output_text,
        "result": result,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("case", RECOVERY_CASES, ids=lambda case: case.name)
async def test_resilience_case(
    case: ResilienceCase,
    responses_endpoint: str,
    foundry_api_key: Callable[[], Awaitable[str]],
    reconnect_timeout: float,
) -> None:
    outcome = await run_case(
        responses_endpoint,
        api_key=foundry_api_key,
        case=case,
        reconnect_timeout_seconds=reconnect_timeout,
    )

    if case.setup_input_text is not None:
        assert outcome["conversation_id"] is not None
        assert outcome["setup_response_id"] is not None
    if case.verification_input_text is not None:
        assert outcome["verification_response_id"] is not None
    if case.verify_terminal_retrieval:
        assert outcome["terminal_retrieval_verified"] is True


async def _amain(args: argparse.Namespace) -> dict[str, Any]:
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, AZURE_AI_SCOPE)

    async def api_key() -> str:
        return await asyncio.to_thread(token_provider)

    try:
        return await run_case(
            args.url,
            api_key=api_key,
            case=RECOVERY_CASES_BY_CRASH_POINT[args.crash_point],
            reconnect_timeout_seconds=args.reconnect_timeout,
        )
    finally:
        credential.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Full Responses endpoint URL")
    parser.add_argument(
        "--crash-point",
        choices=RECOVERY_CASES_BY_CRASH_POINT,
        required=True,
    )
    parser.add_argument("--reconnect-timeout", type=float, default=60.0)
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()

    outcome = asyncio.run(_amain(args))
    rendered = json.dumps(outcome, indent=2, sort_keys=True)
    print(rendered)
    if args.result_file is not None:
        args.result_file.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
