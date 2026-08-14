"""Crash window 5: crash after superstep 2 E2 and before its E3."""

import json

from tests.e2e_tests.agents.hosting.responses_resilience.common.cases import (
    EXPECTED_RESPONSE,
    ResilienceCase,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app import (
    crash_points,
)
SECOND_CHECKPOINT_BEFORE_METADATA = ResilienceCase(
    name="case_window05_second_checkpoint_before_metadata",
    crash_point=crash_points.AFTER_2RESEARCH_GRAPH_CHECKPOINT_BEFORE_METADATA,
    input_text=json.dumps(
        {
            crash_points.KEY:
                crash_points.AFTER_2RESEARCH_GRAPH_CHECKPOINT_BEFORE_METADATA
        }
    ),
    expected_node_runs=(1, 2, 1, 1),
    expected_checkpoint_writes=(1, 1, 1, 1),
    expected_response=EXPECTED_RESPONSE,
)