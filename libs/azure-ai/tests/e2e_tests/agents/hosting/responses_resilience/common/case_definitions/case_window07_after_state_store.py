"""Crash window 7: crash after E4 and before E5 terminal persistence."""

import json

from tests.e2e_tests.agents.hosting.responses_resilience.common.cases import (
    EXPECTED_RESPONSE,
    ResilienceCase,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app import (
    crash_points,
)

AFTER_STATE_STORE = ResilienceCase(
    name="case_window07_after_state_store",
    crash_point=crash_points.AFTER_STATE_STORE_BEFORE_COMPLETED,
    input_text=json.dumps(
        {
            crash_points.KEY:
                crash_points.AFTER_STATE_STORE_BEFORE_COMPLETED
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