# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Convert public tool content to Responses function-call output parts."""

from copy import deepcopy
from typing import Any, cast

from azure.ai.agentserver.responses.models import (
    InputFileContentParam,
    InputImageContentParamAutoParam,
    InputTextContentParam,
)
from langchain_core.messages.block_translators.openai import (
    convert_to_openai_data_block,
)

from ._utils import extract_text


def tool_output(
    content: str | list[str | dict[str, Any]],
) -> (
    str
    | list[
        InputTextContentParam | InputImageContentParamAutoParam | InputFileContentParam
    ]
):
    """Preserve supported tool attachments, retaining the text-only wire format.

    Args:
        content: Model-visible ToolMessage content, excluding application artifacts.

    Returns:
        A string for text-only content, or ordered Responses text/image/file parts.

    Raises:
        ValueError: A standard image/file block lacks its required data source.
    """
    if not isinstance(content, list) or not any(
        isinstance(p, dict)
        and p.get("type") in {"input_image", "input_file", "image", "file", "image_url"}
        for p in content
    ):
        return extract_text(content)
    parts: list[dict[str, Any]] = []
    for part in deepcopy(content):
        kind = part.get("type") if isinstance(part, dict) else None
        if isinstance(part, dict) and kind in {"input_image", "input_file"}:
            parts.append(part)
        elif isinstance(part, dict) and kind == "image_url":
            image = part["image_url"]
            parts.append(
                {
                    "type": "input_image",
                    "image_url": image["url"],
                    **({"detail": image["detail"]} if "detail" in image else {}),
                }
            )
        elif isinstance(part, dict) and kind == "file" and "file" in part:
            parts.append({**part["file"], "type": "input_file"})
        elif isinstance(part, dict) and kind in {"image", "file"}:
            # The upstream image converter supports URL/base64 but not file IDs.
            if kind == "image" and (
                "file_id" in part or part.get("source_type") == "id"
            ):
                converted = {
                    "type": "input_image",
                    "file_id": part["file_id"] if "file_id" in part else part["id"],
                }
            else:
                converted = convert_to_openai_data_block(part, api="responses")
            extras = part.get("extras") or part.get("metadata") or {}
            for field in ("detail", "filename") if kind == "file" else ("detail",):
                if field in part or field in extras:
                    converted[field] = part.get(field, extras.get(field))
            parts.append(converted)
        else:
            text = extract_text([part])
            if text:
                parts.append({"type": "input_text", "text": text})
    for part in parts:
        if part["type"] == "input_image":
            part.setdefault("detail", "auto")
    return cast(
        list[
            InputTextContentParam
            | InputImageContentParamAutoParam
            | InputFileContentParam
        ],
        parts,
    )
