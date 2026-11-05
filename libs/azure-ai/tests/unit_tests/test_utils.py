"""Unit tests for `langchain_azure_ai.utils.utils`."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_projects(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub `azure.ai.projects`, which is an optional dependency."""
    client = MagicMock(name="AIProjectClient")
    # Force the non-hub branch, which is the one that builds the OpenAI client.
    client.connections.get_default.side_effect = KeyError("no connection")
    client.get_openai_client.return_value.base_url = "https://example.openai.azure.com"

    projects = ModuleType("azure.ai.projects")
    projects.AIProjectClient = MagicMock(return_value=client)  # type: ignore[attr-defined]

    models = ModuleType("azure.ai.projects.models")
    models.ApiKeyCredentials = MagicMock()  # type: ignore[attr-defined]
    models.Connection = MagicMock()  # type: ignore[attr-defined]
    models.ConnectionType = MagicMock()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "azure.ai.projects", projects)
    monkeypatch.setitem(sys.modules, "azure.ai.projects.models", models)
    return client


def test_unsupported_service_raises(fake_projects: MagicMock) -> None:
    """A service name that is merely a substring of "inference" is not valid."""
    from langchain_azure_ai.utils.utils import get_service_endpoint_from_project

    with pytest.raises(ValueError, match="is not supported"):
        get_service_endpoint_from_project(
            "https://example.services.ai.azure.com/api/projects/p",
            MagicMock(),
            service="in",
        )


def test_empty_service_raises(fake_projects: MagicMock) -> None:
    """The empty string is a substring of every string; it must not pass."""
    from langchain_azure_ai.utils.utils import get_service_endpoint_from_project

    with pytest.raises(ValueError, match="is not supported"):
        get_service_endpoint_from_project(
            "https://example.services.ai.azure.com/api/projects/p",
            MagicMock(),
            service="",
        )


def test_inference_service_still_resolves(fake_projects: MagicMock) -> None:
    """The exact name keeps working."""
    from langchain_azure_ai.utils.utils import get_service_endpoint_from_project

    endpoint, _ = get_service_endpoint_from_project(
        "https://example.services.ai.azure.com/api/projects/p",
        MagicMock(),
        service="inference",
    )

    assert endpoint == "https://example.openai.azure.com/v1"


def test_api_version_is_forwarded(fake_projects: MagicMock) -> None:
    """`api_version` must reach the OpenAI client, not be hard-coded."""
    from langchain_azure_ai.utils.utils import get_service_endpoint_from_project

    get_service_endpoint_from_project(
        "https://example.services.ai.azure.com/api/projects/p",
        MagicMock(),
        service="inference",
        api_version="2026-05-01-preview",
    )

    fake_projects.get_openai_client.assert_called_once_with(
        api_version="2026-05-01-preview"
    )


def test_api_version_defaults_to_v1(fake_projects: MagicMock) -> None:
    from langchain_azure_ai.utils.utils import get_service_endpoint_from_project

    get_service_endpoint_from_project(
        "https://example.services.ai.azure.com/api/projects/p",
        MagicMock(),
        service="inference",
    )

    fake_projects.get_openai_client.assert_called_once_with(api_version="v1")
