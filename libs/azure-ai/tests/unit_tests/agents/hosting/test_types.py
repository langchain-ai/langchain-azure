# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for public hosting type exports."""

from __future__ import annotations

import pytest

pytest.importorskip("azure.ai.agentserver.invocations")
pytest.importorskip("azure.ai.agentserver.responses")

from langchain_azure_ai.agents import hosting  # noqa: E402
from langchain_azure_ai.agents.hosting import (  # noqa: E402
    CreateResponse,
    InvocationAgentServerHost,
    ResponseContext,
    ResponseEventStream,
    ResponseProviderProtocol,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)


def test_hosting_reexports_sdk_types() -> None:
    exported_types = [
        CreateResponse,
        InvocationAgentServerHost,
        ResponseContext,
        ResponseEventStream,
        ResponseProviderProtocol,
        ResponsesAgentServerHost,
        ResponsesServerOptions,
    ]

    assert all(
        exported_type.__name__ in hosting.__all__ for exported_type in exported_types
    )
    assert InvocationAgentServerHost.__module__.startswith(
        "azure.ai.agentserver.invocations"
    )
    assert all(
        exported_type.__module__.startswith("azure.ai.agentserver.responses")
        for exported_type in exported_types
        if exported_type is not InvocationAgentServerHost
    )
