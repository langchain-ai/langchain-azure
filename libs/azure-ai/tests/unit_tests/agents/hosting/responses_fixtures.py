# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model-side Responses fixtures independent of hosting conversion and builders."""

import json
from collections.abc import Iterator
from typing import Any


def model_response(
    parts: list[dict[str, Any]], name: str = "provider"
) -> dict[str, Any]:
    return {
        "id": f"resp-{name}",
        "object": "response",
        "created_at": 0,
        "model": "test",
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": f"msg-{name}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": parts,
            }
        ],
    }


def model_events(response: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield {
        "type": "response.created",
        "response": {**response, "status": "in_progress", "output": []},
    }
    for output_index, item in enumerate(response["output"]):
        yield {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {**item, "status": "in_progress", "content": []},
        }
        for content_index, part in enumerate(item["content"]):
            location = {
                "item_id": item["id"],
                "output_index": output_index,
                "content_index": content_index,
            }
            yield {
                **location,
                "type": "response.content_part.added",
                "part": {**part, "text": "", "annotations": []},
            }
            yield {
                **location,
                "type": "response.output_text.delta",
                "delta": part["text"],
                "logprobs": [],
            }
            for annotation_index, annotation in enumerate(part["annotations"]):
                yield {
                    **location,
                    "type": "response.output_text.annotation.added",
                    "annotation_index": annotation_index,
                    "annotation": annotation,
                }
            yield {
                **location,
                "type": "response.output_text.done",
                "text": part["text"],
                "logprobs": [],
            }
            yield {**location, "type": "response.content_part.done", "part": part}
        yield {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": item,
        }
    yield {"type": "response.completed", "response": response}


def sse_event(event: dict[str, Any], sequence: int) -> bytes:
    return f"data: {json.dumps({**event, 'sequence_number': sequence})}\n\n".encode()
