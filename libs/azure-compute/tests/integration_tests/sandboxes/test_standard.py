"""Standard `SandboxBackendProtocol` conformance tests for `ACASandbox`.

Runs the shared suite from `langchain-tests` against a real Azure Container
Apps sandbox. Unlike the blob backend in `langchain-azure-storage`, which had to
fork the suite because it implements the filesystem-only `BackendProtocol`,
`ACASandbox` is a genuine sandbox and can subclass the suite directly.

Requires a sandbox group and the `Container Apps SandboxGroup Data Owner` role
on it. One sandbox is created for the whole class and deleted afterwards. These
cost real Azure resources, so they carry an `aca_sandbox` marker: run just them
with `-m aca_sandbox`, or skip them with `-m "not aca_sandbox"`.
"""

from __future__ import annotations

import os
import uuid
from contextlib import ExitStack
from typing import TYPE_CHECKING

import pytest

# importorskip throughout, not plain imports: `python-dotenv` and
# `langchain-tests` only come with the `test_integration` group, and a
# whole-tree run without them must skip at collection rather than error.
dotenv = pytest.importorskip("dotenv")
dotenv.load_dotenv()

deepagents = pytest.importorskip("deepagents")
pytest.importorskip("azure.containerapps.sandbox")
pytest.importorskip("langchain_tests")

from azure.containerapps.sandbox import (  # noqa: E402
    SandboxGroupClient,
    endpoint_for_region,
)
from azure.identity import DefaultAzureCredential  # noqa: E402
from langchain_tests.integration_tests import SandboxIntegrationTests  # noqa: E402

from langchain_azure_compute.sandboxes import ACASandbox  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZURE_CONTAINER_APPS_RESOURCE_GROUP")
SANDBOX_GROUP = os.getenv("AZURE_CONTAINER_APPS_SANDBOX_GROUP")
REGION = os.getenv("AZURE_CONTAINER_APPS_REGION")

# Release CI runs `make integration_tests` with no Azure resources, so these
# must skip rather than fail when unconfigured.
pytestmark = [
    pytest.mark.aca_sandbox,
    pytest.mark.skipif(
        not all([SUBSCRIPTION_ID, RESOURCE_GROUP, SANDBOX_GROUP, REGION]),
        reason=(
            "Set AZURE_SUBSCRIPTION_ID, AZURE_CONTAINER_APPS_RESOURCE_GROUP, "
            "AZURE_CONTAINER_APPS_SANDBOX_GROUP and AZURE_CONTAINER_APPS_REGION "
            "to run ACA sandbox integration tests"
        ),
    ),
]


class TestACASandboxStandard(SandboxIntegrationTests):
    """Run the shared sandbox suite against a live ACA sandbox."""

    @classmethod
    @pytest.fixture(scope="class")
    def sandbox(cls) -> Iterator[SandboxBackendProtocol]:
        """Create a sandbox for the class, then delete it.

        `python-3.12` rather than `ubuntu`: the suite runs `python -c ...`, and
        the `ubuntu` image provides only `python3`.

        `ExitStack` rather than nested `try`/`finally`: the credential and both
        clients must be closed even when creation itself fails, which a
        `finally` hung off a successful yield never reaches.

        Cleanup is keyed on a unique run label and registered *before* the
        create call: `begin_create_sandbox` issues its PUT synchronously, so a
        failure while waiting for Running (poller timeout, terminal `Failed`,
        Ctrl-C) happens with the sandbox already existing server-side. A
        delete callback registered only after `.result()` would never run on
        exactly those paths, stranding a billable sandbox.
        """
        labels = {
            "owner": "langchain-azure-compute",
            "purpose": "ci",
            "run": uuid.uuid4().hex,
        }
        with ExitStack() as stack:
            credential = stack.enter_context(DefaultAzureCredential())
            group = stack.enter_context(
                SandboxGroupClient(
                    endpoint_for_region(REGION),  # type: ignore[arg-type]
                    credential,
                    subscription_id=SUBSCRIPTION_ID,  # type: ignore[arg-type]
                    resource_group=RESOURCE_GROUP,  # type: ignore[arg-type]
                    sandbox_group=SANDBOX_GROUP,  # type: ignore[arg-type]
                )
            )

            def _delete_labeled() -> None:
                # Covers success and every creation-failure path alike; when
                # the PUT itself never landed, the listing is simply empty.
                for sbx in group.list_sandboxes(labels=labels):
                    group.begin_delete_sandbox(sbx.id).result()

            # Before the create call, so it runs even when `.result()` raises;
            # the stack unwinds LIFO, so the delete still precedes the group
            # client's close.
            stack.callback(_delete_labeled)
            # result() is a sandbox-scoped client, and the poller has already
            # waited for Running -- no get_sandbox_client()/ensure_running().
            client = stack.enter_context(
                group.begin_create_sandbox(disk="python-3.12", labels=labels).result()
            )
            yield ACASandbox(client)
