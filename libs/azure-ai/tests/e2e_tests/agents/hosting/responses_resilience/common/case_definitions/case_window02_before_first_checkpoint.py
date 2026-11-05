"""Crash window 2: crash after E1 and before the first E2 commit."""

import json

from tests.e2e_tests.agents.hosting.responses_resilience.common.cases import (
    EXPECTED_RESPONSE,
    ResilienceCase,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app import (
    crash_points,
)

AFTER_FIRST_NODE_START = ResilienceCase(
    name="case_window02_before_first_checkpoint",
    crash_point=crash_points.AFTER_RESPONSE_CREATED_BEFORE_FIRST_CHECKPOINT,
    input_text=json.dumps(
        {crash_points.KEY: crash_points.AFTER_RESPONSE_CREATED_BEFORE_FIRST_CHECKPOINT}
    ),
    expected_node_runs=(1, 1, 1, 1),
    expected_checkpoint_writes=(1, 1, 1, 1),
    expected_response=EXPECTED_RESPONSE,
)
