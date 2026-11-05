# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared inputs and assertions for local and Foundry resilience tests."""

from __future__ import annotations

from dataclasses import dataclass

from tests.e2e_tests.agents.hosting.responses_resilience.common.responses_client import (  # noqa: E501
    TurnResult,
    final_result,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app.workflow import (  # noqa: E501
    EXECUTE_OUTPUT,
    PLAN_OUTPUT,
    RESEARCH_OUTPUT,
)

NODE_NAMES = ("1plan", "2research", "3execute", "4summarize")
EXPECTED_RESPONSE = f"{PLAN_OUTPUT}\n{RESEARCH_OUTPUT}\n{EXECUTE_OUTPUT}\n"


@dataclass(frozen=True)
class ResilienceCase:
    """Environment-independent input and expected Responses outcome."""

    name: str
    input_text: str
    crash_point: str
    expected_node_runs: tuple[int, int, int, int]
    expected_checkpoint_writes: tuple[int, int, int, int]
    expected_response: str
    verify_terminal_retrieval: bool = False
    setup_input_text: str | None = None
    setup_expected_node_runs: tuple[int, int, int, int] | None = None
    setup_expected_checkpoint_writes: tuple[int, int, int, int] | None = None
    verification_input_text: str | None = None
    verification_expected_node_runs: tuple[int, int, int, int] | None = None
    verification_expected_checkpoint_writes: tuple[int, int, int, int] | None = None
    expected_pre_reset_output_text: str | None = None


def _assert_response_parts(output_text: str, expected_response: str) -> None:
    initial_parts, separator, _ = output_text.rpartition("\n")
    assert separator
    assert f"{initial_parts}\n" == expected_response


def assert_case_outcome(case: ResilienceCase, outcome: TurnResult) -> None:
    """Assert every environment-independent expectation for a case."""

    assert outcome.status == "completed"
    _assert_response_parts(outcome.output_text, case.expected_response)
    result = final_result(outcome.output_text)
    actual_node_runs = tuple(result["node_runs"][node] for node in NODE_NAMES)
    assert actual_node_runs == case.expected_node_runs
    actual_checkpoint_writes = tuple(
        result["checkpoint_writes"][node] for node in NODE_NAMES
    )
    assert actual_checkpoint_writes == case.expected_checkpoint_writes
    if case.expected_pre_reset_output_text is not None:
        assert outcome.pre_reset_output_text == case.expected_pre_reset_output_text


def assert_setup_outcome(case: ResilienceCase, outcome: TurnResult) -> None:
    """Assert the completed setup turn for a multi-turn case."""

    assert case.setup_expected_node_runs is not None
    assert case.setup_expected_checkpoint_writes is not None
    assert outcome.status == "completed"
    _assert_response_parts(outcome.output_text, EXPECTED_RESPONSE)
    result = final_result(outcome.output_text)
    actual_node_runs = tuple(result["node_runs"][node] for node in NODE_NAMES)
    assert actual_node_runs == case.setup_expected_node_runs
    actual_checkpoint_writes = tuple(
        result["checkpoint_writes"][node] for node in NODE_NAMES
    )
    assert actual_checkpoint_writes == case.setup_expected_checkpoint_writes


def assert_verification_outcome(case: ResilienceCase, outcome: TurnResult) -> None:
    """Assert a post-recovery turn continued from the repaired state pointer."""

    assert case.verification_expected_node_runs is not None
    assert case.verification_expected_checkpoint_writes is not None
    assert outcome.status == "completed"
    _assert_response_parts(outcome.output_text, EXPECTED_RESPONSE)
    result = final_result(outcome.output_text)
    actual_node_runs = tuple(result["node_runs"][node] for node in NODE_NAMES)
    assert actual_node_runs == case.verification_expected_node_runs
    actual_checkpoint_writes = tuple(
        result["checkpoint_writes"][node] for node in NODE_NAMES
    )
    assert actual_checkpoint_writes == case.verification_expected_checkpoint_writes
