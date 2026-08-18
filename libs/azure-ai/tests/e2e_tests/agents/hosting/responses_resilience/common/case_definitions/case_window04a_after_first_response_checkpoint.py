"""Crash window 4a: crash just after E3 for superstep 1."""

import json

from tests.e2e_tests.agents.hosting.responses_resilience.common.cases import (
    EXPECTED_RESPONSE,
    ResilienceCase,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app import (
    crash_points,
)

AFTER_FIRST_SUPERSTEP_COMMIT = ResilienceCase(
    name="case_window04a_after_first_response_checkpoint",
    crash_point=crash_points.AFTER_1PLAN_RESPONSES_CHECKPOINT,
    input_text=json.dumps(
        {crash_points.KEY: crash_points.AFTER_1PLAN_RESPONSES_CHECKPOINT}
    ),
    expected_node_runs=(1, 1, 1, 1),
    expected_checkpoint_writes=(1, 1, 1, 1),
    expected_response=EXPECTED_RESPONSE,
)
