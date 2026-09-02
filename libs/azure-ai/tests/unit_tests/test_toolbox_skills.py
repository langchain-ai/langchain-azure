"""Unit tests for the Foundry Toolbox Deep Agents skills backend."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

pytest.importorskip("deepagents", reason="Deep Agents requires Python 3.11+")

from deepagents.backends.utils import create_file_data  # noqa: E402
from deepagents.middleware.skills import _list_skills_with_errors  # noqa: E402

from langchain_azure_ai.agents.deepagents import (
    FoundryToolboxSkillsBackend,
)  # noqa: E402

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_azure_ai._api.base.ExperimentalWarning"
)


class FakeToolbox:
    """Return versioned skill snapshots without making MCP requests."""

    toolbox_endpoint = (
        "https://resource.services.ai.azure.com/api/projects/p/"
        "toolboxes/tb/mcp?api-version=v1"
    )

    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    def get_skills(self, *, base_path: str = "/") -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("refresh failed")
        self.calls += 1
        content = (
            "---\n"
            "name: proof\n"
            f"description: version {self.calls}\n"
            "---\n"
            f"body-v{self.calls}"
        )
        return {f"{base_path}proof/SKILL.md": create_file_data(content)}


def _patch_toolbox(
    monkeypatch: pytest.MonkeyPatch,
) -> FakeToolbox:
    fake = FakeToolbox()
    monkeypatch.setattr(
        "langchain_azure_ai.agents.deepagents.toolbox.AzureAIProjectToolbox",
        lambda **kwargs: fake,
    )
    return fake


class TestFoundryToolboxSkillsBackend:
    def test_refreshes_metadata_and_keeps_last_good_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _patch_toolbox(monkeypatch)
        backend = FoundryToolboxSkillsBackend(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="token",
            cache_refresh_interval=timedelta(0),
        )

        first, error = _list_skills_with_errors(backend, "/")
        assert error is None
        assert first[0]["description"] == "version 1"

        second, error = _list_skills_with_errors(backend, "/")
        assert error is None
        assert second[0]["description"] == "version 2"

        fake.fail = True
        assert backend.refresh() is False
        result = backend.read("/proof/SKILL.md")
        assert result.file_data is not None
        assert "body-v2" in result.file_data["content"]

    async def test_async_scan_refreshes_once_when_cache_is_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _patch_toolbox(monkeypatch)
        backend = FoundryToolboxSkillsBackend(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="token",
            cache_refresh_interval=timedelta(0),
        )

        result = await backend.als("/")

        assert result.entries
        assert fake.calls == 1

    def test_is_read_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_toolbox(monkeypatch)
        backend = FoundryToolboxSkillsBackend(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="token",
        )

        assert backend.write("/proof/SKILL.md", "new").error
        assert backend.edit("/proof/SKILL.md", "a", "b").error
        assert backend.delete("/proof/SKILL.md").error
        assert backend.upload_files([("/proof/SKILL.md", b"new")])[0].error
