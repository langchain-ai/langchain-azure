"""Crash window 3b: second turn after its first E2 and before E3."""

import json

from tests.e2e_tests.agents.hosting.responses_resilience.common.cases import (
    EXPECTED_RESPONSE,
    ResilienceCase,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app import (
    crash_points,
)

SECOND_TURN_STATE_STORE_WINDOW = ResilienceCase(
    name="case_window03b_second_turn_first_checkpoint",
    crash_point=crash_points.SECOND_TURN_FIRST_CHECKPOINT_BEFORE_METADATA,
    input_text=json.dumps(
        {
            crash_points.KEY:
                crash_points.SECOND_TURN_FIRST_CHECKPOINT_BEFORE_METADATA
        }
    ),
    expected_node_runs=(2, 2, 2, 2),
    expected_checkpoint_writes=(2, 2, 2, 2),
    expected_response=EXPECTED_RESPONSE,
    setup_input_text=json.dumps(
        {crash_points.KEY: crash_points.NO_CRASH}
    ),
    setup_expected_node_runs=(1, 1, 1, 1),
    setup_expected_checkpoint_writes=(1, 1, 1, 1),
)