"""Crash window 6: crash after final E3 and before E4 State Store commit."""

import json

from tests.e2e_tests.agents.hosting.responses_resilience.common.cases import (
    EXPECTED_RESPONSE,
    ResilienceCase,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app import (
    crash_points,
)
BEFORE_STATE_STORE = ResilienceCase(
    name="case_window06_before_state_store",
    crash_point=crash_points.AFTER_FINAL_RESPONSES_CHECKPOINT_BEFORE_STATE_STORE,
    input_text=json.dumps(
        {
            crash_points.KEY:
                crash_points.AFTER_FINAL_RESPONSES_CHECKPOINT_BEFORE_STATE_STORE
        }
    ),
    expected_node_runs=(1, 1, 1, 1),
    expected_checkpoint_writes=(1, 1, 1, 1),
    expected_response=EXPECTED_RESPONSE,
    verification_input_text=json.dumps(
        {crash_points.KEY: crash_points.NO_CRASH}
    ),
    verification_expected_node_runs=(2, 2, 2, 2),
    verification_expected_checkpoint_writes=(2, 2, 2, 2),
)