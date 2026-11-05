# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Internal State Store client creation and handler-scoped reuse."""

from .state_store_factory import (
    _state_store_deletion_scope,
    get_state_store,
    state_store_client_scope,
)

__all__ = [
    "_state_store_deletion_scope",
    "get_state_store",
    "state_store_client_scope",
]
