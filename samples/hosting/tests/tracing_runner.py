# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Launch an actual hosted sample with local substitutes for external services."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from tracing_scenarios import CASES, CONTENT_RECORDING_FLAGS, model_responses

SAMPLES = Path(__file__).resolve().parents[2]


def _start_sample(sample: str) -> None:
    from contextlib import ExitStack, nullcontext
    from copy import deepcopy
    from types import SimpleNamespace

    from azure.ai.agentserver.core.storage import FoundryStateStore
    from azure.ai.projects import AIProjectClient
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import messages_to_dict
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import MemorySaver
    from pydantic import Field

    stores: dict[str, dict[str, Any]] = {}

    class LocalStateStore:
        def __init__(self, name: str) -> None:
            self.name = name
            stores.setdefault(name, {})

        async def __aenter__(self) -> LocalStateStore:
            return self

        async def __aexit__(self, *_: Any) -> None:
            pass

        async def aclose(self) -> None:
            pass

        async def get_item(self, key: str) -> SimpleNamespace | None:
            value = stores[self.name].get(key)
            return SimpleNamespace(value=deepcopy(value)) if value is not None else None

        async def set_item(self, key: str, value: Any, **_: Any) -> None:
            stores[self.name][key] = deepcopy(value)

    async def get_store(name: str, *_: Any, **__: Any) -> LocalStateStore:
        return LocalStateStore(name)

    class Model(FakeMessagesListChatModel):
        root_client: Any = Field(default_factory=nullcontext)
        root_async_client: Any = Field(default_factory=nullcontext)
        calls: list[dict[str, Any]] = Field(default_factory=list)

        def bind_tools(self, *_: Any, **__: Any) -> Model:
            return self

        def _generate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
            result = super()._generate(messages, *args, **kwargs)
            self.calls.append(
                {
                    "inputs": messages_to_dict(messages),
                    "output": messages_to_dict([result.generations[0].message])[0],
                }
            )
            Path("model-calls.json").write_text(
                json.dumps(
                    {
                        "sample": sample,
                        "content_recording": {
                            name: os.environ.get(name)
                            for name in CONTENT_RECORDING_FLAGS
                        },
                        "calls": self.calls,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return result

    @tool
    def web_search(query: str) -> str:
        """Substitute the external MCP/Toolbox service."""
        return f"Local search result for {query}: https://example.invalid/tracing"

    protocol, scenario = CASES[sample]
    model = Model(responses=model_responses(scenario))
    directory = SAMPLES / sample
    sys.path.insert(0, str(directory))
    project = MagicMock()
    project.__enter__.return_value = project
    project.get_openai_client.return_value.base_url = (
        "https://example.invalid/openai/v1"
    )
    with ExitStack() as patches:
        for target, replacement in {
            "langchain_openai.ChatOpenAI": MagicMock(return_value=model),
            "langchain_azure_ai.chat_models.AzureAIOpenAIApiChatModel": MagicMock(
                return_value=model
            ),
            "langchain_azure_ai.agents.hosting.FoundryCheckpointSaver": lambda **_: (
                MemorySaver()
            ),
            "langchain_mcp_adapters.client.MultiServerMCPClient.get_tools": AsyncMock(
                return_value=[web_search]
            ),
            "langchain_azure_ai.tools.AzureAIProjectToolbox.get_tools": AsyncMock(
                return_value=[web_search]
            ),
        }.items():
            patches.enter_context(patch(target, replacement))
        patches.enter_context(
            patch.object(
                AIProjectClient,
                "get_openai_client",
                return_value=project.get_openai_client.return_value,
            )
        )
        patches.enter_context(
            patch.object(AIProjectClient, "__enter__", return_value=project)
        )
        patches.enter_context(
            patch.object(FoundryStateStore, "get_or_create", get_store)
        )
        if sample.endswith(("responses/09_run", "invocations/03_run")):
            from langchain_azure_ai.agents.hosting.run import main

            main(
                ["--protocol", protocol, "--config", str(directory / "langgraph.json")]
            )
        else:
            spec = importlib.util.spec_from_file_location(
                "sample", directory / "main.py"
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            entrypoint = getattr(module, "amain", None) or module.main
            if inspect.iscoroutinefunction(entrypoint):
                asyncio.run(entrypoint())
            else:
                entrypoint()


if __name__ == "__main__":
    _start_sample(sys.argv[1])
