"""Crash window 4b: crash during superstep 2 before its E2 commit."""

import json

from tests.e2e_tests.agents.hosting.responses_resilience.common.cases import (
    EXPECTED_RESPONSE,
    ResilienceCase,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app import (
    crash_points,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app.workflow import (
    PLAN_OUTPUT,
    RESEARCH_OUTPUT,
)

CHECKPOINT_METADATA_WINDOW = ResilienceCase(
    name="case_window04b_during_second_superstep",
    crash_point=crash_points.AFTER_2RESEARCH_CHECKPOINT_BEFORE_METADATA,
    input_text=json.dumps(
        {
            crash_points.KEY:
                crash_points.AFTER_2RESEARCH_CHECKPOINT_BEFORE_METADATA
        }
    ),
    expected_node_runs=(1, 2, 1, 1),
    expected_checkpoint_writes=(1, 1, 1, 1),
    expected_response=EXPECTED_RESPONSE,
    expected_pre_reset_output_text=(
        f"{PLAN_OUTPUT}\n{RESEARCH_OUTPUT.split(' ')[0]} "
    ),
)