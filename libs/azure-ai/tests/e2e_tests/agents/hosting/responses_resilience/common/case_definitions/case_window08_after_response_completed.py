"""Crash window 8: crash after E5 persists the terminal response."""

import json

from tests.e2e_tests.agents.hosting.responses_resilience.common.cases import (
    EXPECTED_RESPONSE,
    ResilienceCase,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app import (
    crash_points,
)

AFTER_RESPONSE_COMPLETED = ResilienceCase(
    name="case_window08_after_response_completed",
    crash_point=crash_points.AFTER_RESPONSE_COMPLETED,
    input_text=json.dumps({crash_points.KEY: crash_points.AFTER_RESPONSE_COMPLETED}),
    expected_node_runs=(1, 1, 1, 1),
    expected_checkpoint_writes=(1, 1, 1, 1),
    expected_response=EXPECTED_RESPONSE,
    verify_terminal_retrieval=True,
)
