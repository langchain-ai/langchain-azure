"""Crash window 1: crash before E1 creates a response record."""

import json

from tests.e2e_tests.agents.hosting.responses_resilience.common.cases import (
    EXPECTED_RESPONSE,
    ResilienceCase,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app import (
    crash_points,
)

BEFORE_RESPONSE_CREATED = ResilienceCase(
    name="case_window01_before_response_created",
    crash_point=crash_points.BEFORE_RESPONSE_CREATED,
    input_text=json.dumps(
        {crash_points.KEY: crash_points.BEFORE_RESPONSE_CREATED}
    ),
    expected_node_runs=(1, 1, 1, 1),
    expected_checkpoint_writes=(1, 1, 1, 1),
    expected_response=EXPECTED_RESPONSE,
)