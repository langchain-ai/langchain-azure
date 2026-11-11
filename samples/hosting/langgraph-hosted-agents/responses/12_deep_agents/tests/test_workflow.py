"""Exercise checkpoint restoration and handled publication failures."""

from __future__ import annotations

import json
import os
import sys
import unittest
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock, patch

from azure.core.exceptions import HttpResponseError, ServiceRequestError
from deepagents.backends.utils import create_file_data
from langchain_azure_ai.agents.hosting import (
    FoundryCheckpointSaver,
    ResponsesHostServer,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import build_agent
from report_export import create_report_export_tool

REPORT = "# Report v1\nEvidence [1].\n\n[1] https://example.org/source"
REVISION = REPORT + "\n\n## Recovery limitations\nOnly saved state is restored."
FAILURE = (
    "Publishing /report-v1.md failed. No download link was created. "
    "The saved report remains available for editing."
)


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedModel:
        return self

    def get_num_tokens_from_messages(
        self, messages: list[BaseMessage], tools: Sequence[Any] | None = None
    ) -> int:
        return sum(len(str(message.content)) // 4 for message in messages)


def call(name: str, arguments: dict[str, Any], identifier: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": arguments, "id": identifier, "type": "tool_call"}
        ],
    )


def response(client: TestClient, stream: bool, **request: Any) -> dict[str, Any]:
    result = client.post("/responses", json={"stream": stream, **request})
    result.raise_for_status()
    if not stream:
        return result.json()
    events = [
        json.loads(line.removeprefix("data: "))
        for line in result.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    if any(
        event.get("type") in {"error", "response.failed", "response.incomplete"}
        for event in events
    ):
        raise AssertionError("Unexpected terminal error in native host stream")
    return next(
        event["response"]
        for event in events
        if event.get("type") == "response.completed"
    )


class CheckpointRestoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_state_survives_saver_reopen_without_delta_history(self) -> None:
        with (
            TemporaryDirectory() as state_root,
            patch.dict(
                os.environ,
                {
                    "AGENTSERVER_STATE_ROOT": state_root,
                    "OTEL_SDK_DISABLED": "true",
                    "REPORT_STORAGE_ACCOUNT_URL": "https://reports.blob.core.windows.net",
                    "REPORT_STORAGE_CONTAINER": "reports",
                },
            ),
            patch.object(
                FoundryCheckpointSaver,
                "aget_delta_channel_history",
                side_effect=AssertionError(
                    "Checkpoint restore must not load delta history"
                ),
            ) as history,
        ):
            os.environ.pop("FOUNDRY_HOSTING_ENVIRONMENT", None)
            config: RunnableConfig = {"configurable": {"thread_id": "full-state"}}
            original = create_file_data(REPORT)
            revised = create_file_data(REVISION)
            model = ScriptedModel(
                messages=iter(
                    [
                        call(
                            "task",
                            {
                                "description": "Save research notes",
                                "subagent_type": "research-agent",
                            },
                            "research",
                        ),
                        call(
                            "write_file",
                            {"file_path": "/notes.md", "content": "Research notes"},
                            "notes",
                        ),
                        AIMessage(content="Research complete"),
                        *[AIMessage(content=f"reply-{i}") for i in range(11)],
                    ]
                )
            )
            export = create_report_export_tool(MagicMock())
            async with FoundryCheckpointSaver(MagicMock()) as saver:
                graph = build_agent(model, saver, [], export)
                for i in range(11):
                    inputs: dict[str, Any] = {
                        "messages": [HumanMessage(content=f"user-{i}", id=f"user-{i}")]
                    }
                    if i == 0:
                        inputs["files"] = {
                            "/report.md": original,
                            "/discard.md": original,
                        }
                    elif i == 4:
                        inputs["files"] = {"/report.md": revised}
                    elif i == 7:
                        inputs["files"] = {"/discard.md": None}
                    result = await graph.ainvoke(inputs, config)
                self.assertEqual(
                    [
                        message.content
                        for message in result["messages"]
                        if isinstance(message, HumanMessage)
                        or (isinstance(message, AIMessage) and not message.tool_calls)
                    ],
                    [text for i in range(11) for text in (f"user-{i}", f"reply-{i}")],
                )
                child_checkpoints = [
                    item.checkpoint["channel_values"]
                    async for item in saver.alist(config)
                    if item.config["configurable"]["checkpoint_ns"]
                ]
                self.assertTrue(child_checkpoints)
                self.assertEqual(
                    child_checkpoints[0]["files"]["/notes.md"]["content"],
                    "Research notes",
                )
                self.assertEqual(
                    child_checkpoints[0]["messages"][-1].content, "Research complete"
                )
                selected = await graph.aupdate_state(
                    config,
                    {
                        "messages": [
                            HumanMessage(content="updated user-3", id="user-3"),
                            RemoveMessage(id="user-0"),
                        ]
                    },
                )
                before = await graph.aget_state(selected)
                checkpoint = await saver.aget_tuple(selected)
                self.assertIsNotNone(checkpoint)
                assert checkpoint is not None
                for channel in ("messages", "files"):
                    self.assertEqual(
                        checkpoint.checkpoint["channel_values"][channel],
                        before.values[channel],
                    )
            async with FoundryCheckpointSaver(MagicMock()) as saver:
                graph = build_agent(
                    ScriptedModel(messages=iter([AIMessage(content="reply-11")])),
                    saver,
                    [],
                    export,
                )
                restored = await graph.aget_state(selected)
                self.assertEqual(restored.values["messages"], before.values["messages"])
                self.assertEqual(restored.values["files"]["/report.md"], revised)
                self.assertNotIn("/discard.md", restored.values["files"])
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content="user-11", id="user-11")]},
                    selected,
                )
                self.assertEqual(
                    [
                        m.content
                        for m in result["messages"]
                        if isinstance(m, HumanMessage)
                    ],
                    ["updated user-3" if i == 3 else f"user-{i}" for i in range(1, 12)],
                )
                self.assertEqual(result["messages"][-1].content, "reply-11")
                self.assertEqual(result["files"]["/report.md"], revised)
            history.assert_not_called()


class PublicationFailureWorkflowTests(unittest.TestCase):
    def test_named_versions_publish_selected_saved_bytes_without_overwriting(
        self,
    ) -> None:
        for stream in (False, True):
            with (
                self.subTest(stream=stream),
                patch.dict(
                    os.environ,
                    {
                        "OTEL_SDK_DISABLED": "true",
                        "REPORT_STORAGE_ACCOUNT_URL": "https://reports.blob.core.windows.net",
                        "REPORT_STORAGE_CONTAINER": "reports",
                    },
                ),
            ):
                storage = MagicMock()
                storage.__enter__.return_value = storage
                storage.account_name = "reports"
                blobs: list[MagicMock] = []

                def create_blob(
                    container: str, name: str, saved: list[MagicMock] = blobs
                ) -> MagicMock:
                    blob = MagicMock(
                        url=f"https://reports.blob.core.windows.net/{container}/{name}"
                    )
                    saved.append(blob)
                    return blob

                storage.get_blob_client.side_effect = create_blob
                with (
                    patch("report_export.BlobServiceClient", return_value=storage),
                    patch(
                        "report_export.generate_blob_sas", return_value="sp=r&sig=test"
                    ),
                ):
                    model = ScriptedModel(
                        disable_streaming=True,
                        messages=iter(
                            [
                                call(
                                    "write_file",
                                    {"file_path": "/report-v1.md", "content": REPORT},
                                    "save",
                                ),
                                call(
                                    "publish_report",
                                    {"report_name": "report-v1.md"},
                                    "publish-v1",
                                ),
                                AIMessage(content=REPORT),
                                call(
                                    "write_file",
                                    {"file_path": "/report-v2.md", "content": REPORT},
                                    "copy",
                                ),
                                call(
                                    "edit_file",
                                    {
                                        "file_path": "/report-v2.md",
                                        "old_string": REPORT,
                                        "new_string": REVISION,
                                    },
                                    "revise",
                                ),
                                call(
                                    "publish_report",
                                    {"report_name": "report-v2.md"},
                                    "publish-v2",
                                ),
                                AIMessage(content=REVISION),
                                call(
                                    "publish_report",
                                    {"report_name": "report-v1.md"},
                                    "republish-original",
                                ),
                                AIMessage(content=REPORT),
                            ]
                        ),
                    )
                    graph = build_agent(
                        model,
                        InMemorySaver(),
                        [],
                        create_report_export_tool(MagicMock()),
                    )
                    with TestClient(ResponsesHostServer(graph)) as client:
                        first = response(client, stream, input="Save and publish v1")
                        second = response(
                            client,
                            stream,
                            previous_response_id=first["id"],
                            input="Keep v1 and revise a copy as v2",
                        )
                        third = response(
                            client,
                            stream,
                            previous_response_id=second["id"],
                            input="Publish original v1 again",
                        )
                        self.assertTrue(
                            all(
                                result["status"] == "completed"
                                for result in (first, second, third)
                            )
                        )
                self.assertEqual(
                    [blob.upload_blob.call_args.args[0] for blob in blobs],
                    [REPORT.encode(), REVISION.encode(), REPORT.encode()],
                )
                names = [
                    invocation.args[1]
                    for invocation in storage.get_blob_client.call_args_list
                ]
                self.assertEqual(len(set(names)), 3)
                self.assertTrue(
                    names[0].endswith("/report-v1.md")
                    and names[2].endswith("/report-v1.md")
                )
                self.assertTrue(names[1].endswith("/report-v2.md"))

    def test_host_completes_and_preserves_editable_state_after_publication_error(
        self,
    ) -> None:
        for stream in (False, True):
            for error in (HttpResponseError("denied"), ServiceRequestError("offline")):
                with (
                    self.subTest(stream=stream, error=type(error).__name__),
                    patch.dict(
                        os.environ,
                        {
                            "OTEL_SDK_DISABLED": "true",
                            "REPORT_STORAGE_ACCOUNT_URL": "https://reports.blob.core.windows.net",
                            "REPORT_STORAGE_CONTAINER": "reports",
                        },
                    ),
                ):
                    storage = MagicMock()
                    storage.__enter__.return_value = storage
                    storage.account_name = "reports"
                    storage.get_user_delegation_key.side_effect = error
                    with patch("report_export.BlobServiceClient", return_value=storage):
                        model = ScriptedModel(
                            disable_streaming=True,
                            messages=iter(
                                [
                                    call(
                                        "write_file",
                                        {
                                            "file_path": "/report-v1.md",
                                            "content": REPORT,
                                        },
                                        "save",
                                    ),
                                    call(
                                        "publish_report",
                                        {"report_name": "report-v1.md"},
                                        "publish",
                                    ),
                                    AIMessage(content=FAILURE),
                                    call(
                                        "read_file",
                                        {"file_path": "/report-v1.md"},
                                        "read-original",
                                    ),
                                    call(
                                        "write_file",
                                        {
                                            "file_path": "/report-v2.md",
                                            "content": REPORT,
                                        },
                                        "copy",
                                    ),
                                    call(
                                        "edit_file",
                                        {
                                            "file_path": "/report-v2.md",
                                            "old_string": REPORT,
                                            "new_string": REVISION,
                                        },
                                        "edit",
                                    ),
                                    call(
                                        "read_file",
                                        {"file_path": "/report-v2.md"},
                                        "read-revised",
                                    ),
                                    AIMessage(content=REVISION),
                                ]
                            ),
                        )
                        graph = build_agent(
                            model,
                            InMemorySaver(),
                            [],
                            create_report_export_tool(MagicMock()),
                        )
                        with TestClient(ResponsesHostServer(graph)) as client:
                            first = response(
                                client, stream, input="Save and publish report-v1.md"
                            )
                            self.assertEqual(first["status"], "completed")
                            self.assertIn(FAILURE, json.dumps(first))
                            self.assertNotIn("download_url", json.dumps(first))
                            revised = response(
                                client,
                                stream,
                                previous_response_id=first["id"],
                                input="Read v1, then revise a copy as v2",
                            )
                            self.assertEqual(revised["status"], "completed")
                            assert client.portal is not None
                            config: RunnableConfig = {
                                "configurable": {"thread_id": first["id"]}
                            }
                            state = client.portal.call(graph.aget_state, config)
                            self.assertEqual(
                                state.values["files"]["/report-v1.md"]["content"],
                                REPORT,
                            )
                            self.assertEqual(
                                state.values["files"]["/report-v2.md"]["content"],
                                REVISION,
                            )
                            errors = [
                                message
                                for message in state.values["messages"]
                                if isinstance(message, ToolMessage)
                                and message.name == "publish_report"
                            ]
                            self.assertEqual(len(errors), 1)
                            self.assertEqual(errors[0].status, "error")
                            self.assertIn("No download link", errors[0].content)
                    storage.get_blob_client.assert_not_called()
                    storage.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
