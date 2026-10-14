# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared client-facing message templates for agent hosting."""

from typing import Final

INTERNAL_SERVER_ERROR: Final = "Internal server error."
CHECKPOINT_CONFIG_MIGRATION: Final = (
    "Override build_checkpoint_config to select thread_id, checkpoint_ns, or "
    "checkpoint_id; execution config hooks must preserve that selection."
)
MISSING_FUNCTION_OUTPUT: Final = "No tool output found for function call {call_id}."
MISSING_MCP_APPROVAL: Final = (
    "The following MCP approval requests do not have an approval: "
    "{approval_request_ids}"
)
