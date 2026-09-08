"""Crash window 5: crash after superstep 2 E2 and before its E3."""

import json

from tests.e2e_tests.agents.hosting.responses_resilience.common.cases import (
    EXPECTED_RESPONSE,
    ResilienceCase,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app import (
    crash_points,
)

_CRASH_POINT = crash_points.AFTER_2RESEARCH_GRAPH_CHECKPOINT_BEFORE_METADATA

SECOND_CHECKPOINT_BEFORE_METADATA = ResilienceCase(
    name="case_window05_second_checkpoint_before_metadata",
    crash_point=_CRASH_POINT,
    input_text=json.dumps({crash_points.KEY: _CRASH_POINT}),
    expected_node_runs=(1, 2, 1, 1),
    expected_checkpoint_writes=(1, 1, 1, 1),
    expected_response=EXPECTED_RESPONSE,
)
