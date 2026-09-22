# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Lossless content mappings for the Responses protocol (not Invocations)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from azure.ai.agentserver.responses.models import (
    Annotation,
    InputFileContentParam,
    InputImageContentParamAutoParam,
    InputTextContentParam,
    OutputMessageContent,
)
from langchain_core.messages import AIMessage, ToolMessage

ToolOutput = (
    str
    | list[
        InputTextContentParam | InputImageContentParamAutoParam | InputFileContentParam
    ]
)


def _string(block: dict[str, Any], key: str) -> str:
    value = block.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Responses content requires a string '{key}'.")
    return value


def _fields(block: dict[str, Any], allowed: set[str]) -> None:
    unknown = block.keys() - allowed
    if unknown:
        raise ValueError(
            f"Responses cannot preserve fields {sorted(unknown)} "
            f"on content type {block.get('type')!r}."
        )


def _reference(block: dict[str, Any]) -> dict[str, Any]:
    kind = block["type"]
    keys = (
        ("image_url", "file_id")
        if kind == "input_image"
        else ("file_url", "file_id", "file_data")
    )
    _fields(
        block,
        {"type", "detail", *keys} | ({"filename"} if kind == "input_file" else set()),
    )
    if sum(block.get(key) is not None for key in keys) != 1:
        raise ValueError(f"Responses {kind} requires exactly one data reference.")
    for key in (*keys, "filename"):
        if key in block and block[key] is not None:
            _string(block, key)
            if key in keys and not block[key]:
                raise ValueError(f"Responses {kind} requires a nonempty '{key}'.")
    if "detail" in block and block["detail"] is not None:
        valid = (
            {"low", "high", "auto", "original"}
            if kind == "input_image"
            else {"low", "high"}
        )
        if block["detail"] not in valid:
            raise ValueError(f"Responses {kind} has an unsupported detail.")
    return deepcopy(block)


def input_content(content: Any) -> str | list[str | dict[str, Any]]:
    """Preserve rich request/history parts, retaining the text-only string path."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Responses message content must be a string or a list.")
    parts: list[str | dict[str, Any]] = []
    plain = True
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            kind = part.get("type")
            if kind in {"input_text", "output_text", "text"}:
                _string(part, "text")
                _fields(part, {"type", "text", "annotations", "logprobs"})
                parts.append({**deepcopy(part), "type": "text"})
                plain = (
                    plain and not part.get("annotations") and not part.get("logprobs")
                )
            elif kind in {"input_image", "input_file"}:
                parts.append(_reference(part))
                plain = False
            elif kind == "refusal":
                _fields(part, {"type", "refusal"})
                _string(part, "refusal")
                parts.append(deepcopy(part))
                plain = False
            else:
                raise ValueError(
                    f"Responses input content type {kind!r} is unsupported."
                )
        else:
            raise ValueError("Responses content parts must be strings or dictionaries.")
    if plain:
        return "".join(cast(dict[str, Any], part)["text"] for part in parts)
    return parts


def _data_block(part: dict[str, Any]) -> dict[str, Any]:
    """Map supported LangChain v1 data blocks without inventing file names."""
    kind = part["type"]
    _fields(
        part, {"type", "url", "base64", "file_id", "mime_type", "filename", "extras"}
    )
    extras = part.get("extras", {})
    if not isinstance(extras, dict):
        raise ValueError("Responses data block extras must be a dictionary.")
    _fields(extras, {"detail", "filename"})
    sources = [key for key in ("url", "base64", "file_id") if part.get(key) is not None]
    if len(sources) != 1:
        raise ValueError("Responses data blocks require exactly one data reference.")
    source = sources[0]
    data = _string(part, source)
    if source == "base64":
        mime = _string(part, "mime_type")
        if not mime or "/" not in mime:
            raise ValueError("Responses base64 content requires a MIME type.")
        data = f"data:{mime};base64,{data}"
    elif part.get("mime_type") is not None:
        raise ValueError("Responses URL/file ID references cannot carry a MIME field.")
    result: dict[str, Any] = {"type": f"input_{kind}"}
    key = (
        "file_id"
        if source == "file_id"
        else (
            "image_url"
            if kind == "image"
            else "file_data"
            if source == "base64"
            else "file_url"
        )
    )
    result[key] = data
    filename = part.get("filename", extras.get("filename"))
    if (
        "filename" in part
        and "filename" in extras
        and part["filename"] != extras["filename"]
    ):
        raise ValueError("Responses file block has conflicting filename values.")
    if filename is not None:
        if kind != "file":
            raise ValueError("Responses image references cannot carry a filename.")
        result["filename"] = filename
    if "detail" in extras:
        result["detail"] = extras["detail"]
    return _reference(result)


def tool_output(message: ToolMessage) -> ToolOutput:
    """Export public tool content; never expose arbitrary application artifacts."""
    if message.artifact is not None:
        raise ValueError(
            "Responses cannot export ToolMessage.artifact separately from model "
            "content. Keep private artifacts in application storage; explicitly "
            "place public text/image/file blocks in ToolMessage.content."
        )
    if isinstance(message.content, str):
        return message.content
    parts: list[
        InputTextContentParam | InputImageContentParamAutoParam | InputFileContentParam
    ] = []
    for part in message.content:
        if isinstance(part, str):
            parts.append({"type": "input_text", "text": part})
            continue
        kind = part.get("type")
        if kind in {"text", "input_text"}:
            _fields(part, {"type", "text"})
            parts.append({"type": "input_text", "text": _string(part, "text")})
        elif kind in {"input_image", "input_file"}:
            parts.append(
                cast(
                    InputImageContentParamAutoParam | InputFileContentParam,
                    _reference(part),
                )
            )
        elif kind == "file" and "file" in part:
            _fields(part, {"type", "file"})
            file = part["file"]
            if not isinstance(file, dict):
                raise ValueError("Responses file content requires a dictionary.")
            parts.append(
                cast(InputFileContentParam, _reference({**file, "type": "input_file"}))
            )
        elif kind in {"image", "file"}:
            parts.append(
                cast(
                    InputImageContentParamAutoParam | InputFileContentParam,
                    _data_block(part),
                )
            )
        elif kind == "image_url":
            _fields(part, {"type", "image_url"})
            image = part.get("image_url")
            if not isinstance(image, dict):
                raise ValueError("Responses image_url content requires a dictionary.")
            _fields(image, {"url", "detail"})
            native = {"type": "input_image", "image_url": _string(image, "url")}
            if "detail" in image:
                native["detail"] = image["detail"]
            parts.append(cast(InputImageContentParamAutoParam, _reference(native)))
        else:
            raise ValueError(f"Responses tool content type {kind!r} is unsupported.")
    return parts


def _annotation(annotation: Any) -> Annotation:
    if not isinstance(annotation, dict):
        raise ValueError("Responses text annotations must be dictionaries.")
    kind = annotation.get("type")
    if kind == "file_citation" and "file_index" in annotation:
        annotation = deepcopy(annotation)
        annotation["index"] = annotation.pop("file_index")
    if kind == "citation":
        _fields(annotation, {"type", "url", "title", "start_index", "end_index"})
        annotation = {**annotation, "type": "url_citation"}
        kind = "url_citation"
    if kind not in {
        "url_citation",
        "file_citation",
        "container_file_citation",
        "file_path",
    }:
        raise ValueError(f"Responses annotation type {kind!r} is unsupported.")
    return cast(Annotation, deepcopy(annotation))


def assistant_content(message: AIMessage) -> list[OutputMessageContent]:
    """Validate and preserve the assistant content representable by Responses."""
    content = message.content
    if isinstance(content, str):
        return (
            [
                {
                    "type": "output_text",
                    "text": content,
                    "annotations": [],
                    "logprobs": [],
                }
            ]
            if content
            else []
        )
    parts: list[OutputMessageContent] = []
    for part in content:
        if isinstance(part, str):
            parts.append(
                {"type": "output_text", "text": part, "annotations": [], "logprobs": []}
            )
            continue
        kind = part.get("type")
        if kind in {"text", "output_text"}:
            _fields(part, {"type", "text", "annotations", "logprobs", "index", "id"})
            annotations = part.get("annotations", [])
            if not isinstance(annotations, list):
                raise ValueError("Responses text annotations must be a list.")
            logprobs = part.get("logprobs", [])
            if not isinstance(logprobs, list):
                raise ValueError("Responses text logprobs must be a list.")
            parts.append(
                {
                    "type": "output_text",
                    "text": _string(part, "text"),
                    "annotations": [
                        _annotation(annotation) for annotation in annotations
                    ],
                    "logprobs": deepcopy(logprobs),
                }
            )
        elif kind == "refusal":
            _fields(part, {"type", "refusal", "index", "id"})
            parts.append({"type": "refusal", "refusal": _string(part, "refusal")})
        elif kind == "reasoning":
            # Reasoning uses its own output item, not message content.
            continue
        elif kind in {"function_call", "tool_call"} and any(
            call.get("id") == part.get("call_id", part.get("id"))
            for call in message.tool_calls
        ):
            continue
        else:
            raise ValueError(
                f"Responses assistant content type {kind!r} is unsupported."
            )
    return parts
