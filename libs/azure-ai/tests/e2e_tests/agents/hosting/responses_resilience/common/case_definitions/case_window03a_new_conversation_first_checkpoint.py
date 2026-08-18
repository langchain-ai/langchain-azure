"""Crash window 3a: new conversation after its first E2 and before E3."""

import json

from tests.e2e_tests.agents.hosting.responses_resilience.common.cases import (
    EXPECTED_RESPONSE,
    ResilienceCase,
)
from tests.e2e_tests.agents.hosting.responses_resilience.server.server_app import (
    crash_points,
)

_CRASH_POINT = crash_points.NEW_CONVERSATION_FIRST_CHECKPOINT_BEFORE_METADATA

NEW_CONVERSATION_FIRST_CHECKPOINT = ResilienceCase(
    name="case_window03a_new_conversation_first_checkpoint",
    crash_point=_CRASH_POINT,
    input_text=json.dumps({crash_points.KEY: _CRASH_POINT}),
    expected_node_runs=(1, 1, 1, 1),
    expected_checkpoint_writes=(1, 1, 1, 1),
    expected_response=EXPECTED_RESPONSE,
)
