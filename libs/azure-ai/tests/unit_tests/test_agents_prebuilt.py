"""Unit tests for agent prebuilt helper functions."""

import base64
from typing import Any
from unittest.mock import MagicMock

import pytest
from azure.ai.agents.models import (
    CodeInterpreterToolDefinition,
    FilePurpose,
    FunctionToolDefinition,
)
from langchain_core.messages import HumanMessage

from langchain_azure_ai.agents.prebuilt.declarative import (
    _agent_has_code_interpreter,
    _upload_file_blocks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(tools: Any = None) -> MagicMock:
    """Return a mock Agent with the given tools list."""
    agent = MagicMock()
    agent.tools = tools
    return agent


def _make_client(file_id: str = "file-abc") -> MagicMock:
    """Return a mock AIProjectClient whose file upload returns a FileInfo stub."""
    file_info = MagicMock()
    file_info.id = file_id
    client = MagicMock()
    client.agents.files.upload_and_poll.return_value = file_info
    return client


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


CSV_BYTES = b"col1,col2\n1,2\n3,4"
CSV_B64 = _b64(CSV_BYTES)


# ---------------------------------------------------------------------------
# _agent_has_code_interpreter
# ---------------------------------------------------------------------------


class TestAgentHasCodeInterpreter:
    """Tests for _agent_has_code_interpreter."""

    def test_returns_false_when_tools_is_none(self) -> None:
        assert _agent_has_code_interpreter(_make_agent(tools=None)) is False

    def test_returns_false_when_tools_is_empty(self) -> None:
        assert _agent_has_code_interpreter(_make_agent(tools=[])) is False

    def test_returns_false_when_no_code_interpreter(self) -> None:
        func_tool = MagicMock(spec=FunctionToolDefinition)
        assert _agent_has_code_interpreter(_make_agent(tools=[func_tool])) is False

    def test_returns_true_when_code_interpreter_present(self) -> None:
        ci_tool = MagicMock(spec=CodeInterpreterToolDefinition)
        assert _agent_has_code_interpreter(_make_agent(tools=[ci_tool])) is True

    def test_returns_true_when_mixed_tools_include_code_interpreter(self) -> None:
        ci_tool = MagicMock(spec=CodeInterpreterToolDefinition)
        func_tool = MagicMock(spec=FunctionToolDefinition)
        assert (
            _agent_has_code_interpreter(_make_agent(tools=[func_tool, ci_tool])) is True
        )


# ---------------------------------------------------------------------------
# _upload_file_blocks
# ---------------------------------------------------------------------------


class TestUploadFileBlocks:
    """Tests for _upload_file_blocks."""

    def test_string_content_returns_original_message(self) -> None:
        """String-only messages are returned unchanged with no file IDs."""
        msg = HumanMessage(content="hello")
        client = _make_client()

        result_msg, file_ids = _upload_file_blocks(msg, client)

        assert result_msg is msg
        assert file_ids == []
        client.agents.files.upload_and_poll.assert_not_called()

    def test_no_file_blocks_returns_original_message(self) -> None:
        """Messages with only text blocks produce no uploads."""
        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        client = _make_client()

        result_msg, file_ids = _upload_file_blocks(msg, client)

        assert result_msg is msg
        assert file_ids == []
        client.agents.files.upload_and_poll.assert_not_called()

    def test_file_block_is_uploaded(self) -> None:
        """A file block triggers upload_and_poll and returns the file ID."""
        msg = HumanMessage(
            content=[{"type": "file", "mime_type": "text/csv", "base64": CSV_B64}]
        )
        client = _make_client(file_id="file-xyz")

        result_msg, file_ids = _upload_file_blocks(msg, client)

        assert file_ids == ["file-xyz"]
        client.agents.files.upload_and_poll.assert_called_once()

    def test_upload_uses_tuple_form_not_separate_filename_kwarg(self) -> None:
        """upload_and_poll must be called with file=(filename, bytes) tuple.

        The Azure AI Agents server rejects requests that include a top-level
        'filename' property in the JSON body.  Passing the file as a tuple
        embeds the name inside the multipart form data instead.
        """
        msg = HumanMessage(
            content=[{"type": "file", "mime_type": "text/csv", "base64": CSV_B64}]
        )
        client = _make_client(file_id="file-tuple")

        _upload_file_blocks(msg, client)

        actual_call = client.agents.files.upload_and_poll.call_args
        # Should be called with keyword 'file' whose value is a tuple
        file_arg = actual_call.kwargs.get("file")
        assert isinstance(
            file_arg, tuple
        ), "file must be a (filename, bytes) tuple, not a bare bytes object"
        assert isinstance(file_arg[0], str), "first element of tuple must be filename"
        assert isinstance(file_arg[1], bytes), "second element of tuple must be bytes"
        # 'filename' must NOT appear as a separate keyword argument
        assert (
            "filename" not in actual_call.kwargs
        ), "filename must not be passed as a separate kwarg (server rejects it)"
        # purpose must be set
        assert actual_call.kwargs.get("purpose") == FilePurpose.AGENTS

    def test_file_block_content_matches_original_bytes(self) -> None:
        """The bytes passed to upload_and_poll match the decoded base64 data."""
        msg = HumanMessage(
            content=[{"type": "file", "mime_type": "text/csv", "base64": CSV_B64}]
        )
        client = _make_client()

        _upload_file_blocks(msg, client)

        file_arg = client.agents.files.upload_and_poll.call_args.kwargs["file"]
        assert file_arg[1] == CSV_BYTES

    def test_file_block_stripped_from_returned_message(self) -> None:
        """The file block is removed from the returned message content."""
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "analyze this"},
                {"type": "file", "mime_type": "text/csv", "base64": CSV_B64},
            ]
        )
        client = _make_client()

        result_msg, _ = _upload_file_blocks(msg, client)

        assert result_msg is not msg
        assert len(result_msg.content) == 1  # type: ignore[arg-type]
        assert result_msg.content[0]["type"] == "text"  # type: ignore[index]

    def test_text_blocks_preserved_after_upload(self) -> None:
        """Non-file blocks are kept in the returned message."""
        text_block = {"type": "text", "text": "please analyze"}
        msg = HumanMessage(
            content=[
                text_block,
                {"type": "file", "mime_type": "text/csv", "base64": CSV_B64},
            ]
        )
        client = _make_client()

        result_msg, _ = _upload_file_blocks(msg, client)

        assert result_msg.content == [text_block]  # type: ignore[comparison-overlap]

    def test_message_metadata_preserved(self) -> None:
        """model_copy preserves the original message id and other fields."""
        msg = HumanMessage(
            content=[{"type": "file", "mime_type": "text/csv", "base64": CSV_B64}],
            id="msg-123",
        )
        client = _make_client()

        result_msg, _ = _upload_file_blocks(msg, client)

        assert result_msg.id == "msg-123"

    def test_multiple_file_blocks_all_uploaded(self) -> None:
        """Multiple file blocks produce one upload call each."""
        b64_1 = _b64(b"data1")
        b64_2 = _b64(b"data2")

        file_info_1 = MagicMock()
        file_info_1.id = "file-1"
        file_info_2 = MagicMock()
        file_info_2.id = "file-2"
        client = MagicMock()
        client.agents.files.upload_and_poll.side_effect = [file_info_1, file_info_2]

        msg = HumanMessage(
            content=[
                {"type": "file", "mime_type": "text/plain", "base64": b64_1},
                {"type": "file", "mime_type": "text/plain", "base64": b64_2},
            ]
        )

        result_msg, file_ids = _upload_file_blocks(msg, client)

        assert file_ids == ["file-1", "file-2"]
        assert client.agents.files.upload_and_poll.call_count == 2
        # All file blocks stripped; remaining content is empty list
        assert result_msg.content == []

    def test_extension_derived_from_mime_type(self) -> None:
        """The filename sent to upload_and_poll has the correct extension."""
        msg = HumanMessage(
            content=[
                {"type": "file", "mime_type": "application/pdf", "base64": CSV_B64}
            ]
        )
        client = _make_client()

        _upload_file_blocks(msg, client)

        file_arg = client.agents.files.upload_and_poll.call_args.kwargs["file"]
        filename = file_arg[0]
        assert filename.endswith(".pdf"), f"Expected .pdf extension, got: {filename}"

    def test_invalid_base64_raises_value_error(self) -> None:
        """Malformed base64 data raises ValueError with context."""
        msg = HumanMessage(
            content=[
                {"type": "file", "mime_type": "text/csv", "base64": "!!!not-base64!!!"}
            ]
        )
        client = _make_client()

        with pytest.raises(ValueError, match="Failed to decode base64"):
            _upload_file_blocks(msg, client)

    def test_upload_failure_raises_runtime_error(self) -> None:
        """If upload_and_poll raises, a RuntimeError with context is raised."""
        msg = HumanMessage(
            content=[{"type": "file", "mime_type": "text/csv", "base64": CSV_B64}]
        )
        client = MagicMock()
        client.agents.files.upload_and_poll.side_effect = Exception("network error")

        with pytest.raises(RuntimeError, match="Failed to upload file block"):
            _upload_file_blocks(msg, client)

    def test_file_block_without_base64_not_uploaded(self) -> None:
        """A file block that lacks base64 data is kept as-is and not uploaded."""
        file_block = {"type": "file", "mime_type": "text/csv", "file_id": "existing"}
        msg = HumanMessage(content=[file_block])
        client = _make_client()

        result_msg, file_ids = _upload_file_blocks(msg, client)

        assert file_ids == []
        assert result_msg is msg
        client.agents.files.upload_and_poll.assert_not_called()

    def test_extension_sanitized_against_path_traversal(self) -> None:
        """Path-traversal characters in mime-type are stripped from the filename."""
        msg = HumanMessage(
            content=[
                {
                    "type": "file",
                    "mime_type": "application/../../../etc/passwd",
                    "base64": CSV_B64,
                }
            ]
        )
        client = _make_client()

        _upload_file_blocks(msg, client)

        file_arg = client.agents.files.upload_and_poll.call_args.kwargs["file"]
        filename = file_arg[0]
        # The extension must contain only alphanumeric characters
        ext = filename.rsplit(".", 1)[-1]
        assert ext.isalnum(), f"Extension contains non-alphanumeric characters: {ext!r}"
        assert "/" not in filename
        assert ".." not in filename
