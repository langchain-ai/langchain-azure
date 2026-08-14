"""Shared Responses resilience case definitions."""

from .case_window01_before_response_created import BEFORE_RESPONSE_CREATED
from .case_window02_before_first_checkpoint import AFTER_FIRST_NODE_START
from .case_window03a_new_conversation_first_checkpoint import (
    NEW_CONVERSATION_FIRST_CHECKPOINT,
)
from .case_window03b_second_turn_first_checkpoint import SECOND_TURN_STATE_STORE_WINDOW
from .case_window04a_after_first_response_checkpoint import (
    AFTER_FIRST_SUPERSTEP_COMMIT,
)
from .case_window04b_during_second_superstep import CHECKPOINT_METADATA_WINDOW
from .case_window05_second_checkpoint_before_metadata import (
    SECOND_CHECKPOINT_BEFORE_METADATA,
)
from .case_window06_before_state_store import BEFORE_STATE_STORE
from .case_window07_after_state_store import AFTER_STATE_STORE
from .case_window08_after_response_completed import AFTER_RESPONSE_COMPLETED

RECOVERY_CASES = (
    BEFORE_RESPONSE_CREATED,
    AFTER_FIRST_NODE_START,
    NEW_CONVERSATION_FIRST_CHECKPOINT,
    SECOND_TURN_STATE_STORE_WINDOW,
    AFTER_FIRST_SUPERSTEP_COMMIT,
    CHECKPOINT_METADATA_WINDOW,
    SECOND_CHECKPOINT_BEFORE_METADATA,
    BEFORE_STATE_STORE,
    AFTER_STATE_STORE,
    AFTER_RESPONSE_COMPLETED,
)

__all__ = [
    "AFTER_FIRST_NODE_START",
    "AFTER_FIRST_SUPERSTEP_COMMIT",
    "AFTER_RESPONSE_COMPLETED",
    "AFTER_STATE_STORE",
    "BEFORE_STATE_STORE",
    "BEFORE_RESPONSE_CREATED",
    "CHECKPOINT_METADATA_WINDOW",
    "NEW_CONVERSATION_FIRST_CHECKPOINT",
    "RECOVERY_CASES",
    "SECOND_CHECKPOINT_BEFORE_METADATA",
    "SECOND_TURN_STATE_STORE_WINDOW",
]