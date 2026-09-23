# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared Responses text-part conversion and citation lifecycle."""

from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

from azure.ai.agentserver.responses import ResponseEventStream
from azure.ai.agentserver.responses.models import Annotation as WireAnnotation
from azure.ai.agentserver.responses.models import (
    MessageContent,
    MessageContentOutputTextContent,
)
from azure.ai.agentserver.responses.streaming import (
    OutputItemBuilder,
    OutputItemMessageBuilder,
    TextContentBuilder,
)
from openai.types.responses.response_output_text import Annotation
from pydantic import TypeAdapter, ValidationError

_ANNOTATION_ADAPTER: TypeAdapter[Annotation] = TypeAdapter(Annotation)


def _response_annotation(annotation: Any) -> dict[str, Any] | None:
    """Validate native Responses annotations from ``output_version=responses/v1``.

    Other provider or LangChain standard annotations are omitted without
    changing the answer text or inventing citation titles and offsets.
    """
    if not isinstance(annotation, dict):
        return None
    value = deepcopy(annotation)
    if value.get("type") == "file_citation" and "file_index" in value:
        value["index"] = value.pop("file_index")
    try:
        return _ANNOTATION_ADAPTER.validate_python(value, strict=True).model_dump()
    except ValidationError:
        return None


@dataclass
class _TextPart:
    builder: TextContentBuilder
    annotations: list[WireAnnotation] = field(default_factory=list)


class _TextMessageEmitter:
    """Keep text deltas and per-part citations in the same Responses message."""

    def __init__(self, stream: ResponseEventStream) -> None:
        self.stream = stream
        self.message: OutputItemMessageBuilder | None = None
        self.item: OutputItemBuilder | None = None
        self.parts: dict[object, _TextPart] = {}

    async def add(self, content: Any) -> AsyncIterator[Any]:
        """Emit text immediately and retain citations until the part completes."""
        blocks: Any = (
            [{"type": "text", "text": content, "index": 0}]
            if isinstance(content, str)
            else content
        )
        for block in blocks:
            if isinstance(block, str):
                block = {"type": "text", "text": block}
            if not isinstance(block, dict) or block.get("type", "text") not in {
                "text",
                "output_text",
            }:
                continue
            text = block.get("text", "")
            annotations = [
                cast(WireAnnotation, value)
                for annotation in block.get("annotations") or []
                if (value := _response_annotation(annotation)) is not None
            ]
            if not text and not annotations:
                continue
            index = block.get("index", object())
            if self.message is None:
                self.message = self.stream.add_output_item_message()
                self.item = OutputItemBuilder(
                    self.stream, self.message.output_index, self.message.item_id
                )
                yield self.item.emit_added(
                    {
                        "type": "message",
                        "id": self.item.item_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    }
                )
            if index not in self.parts:
                self.parts[index] = _TextPart(self.message.add_text_content())
                yield self.parts[index].builder.emit_added()
            part = self.parts[index]
            if text:
                yield part.builder.emit_delta(text)
            part.annotations.extend(annotations)

    async def close(self) -> AsyncIterator[Any]:
        """Complete the message with the exact content sent in part events."""
        content: list[MessageContent] = []
        for part in self.parts.values():
            yield part.builder.emit_text_done()
            for annotation in part.annotations:
                yield part.builder.emit_annotation_added(annotation)
            # SDK 2.1.0b2 clears annotations on completion. Enrich its public
            # event and complete the item with the same parts.
            event = part.builder.emit_done()
            completed = cast(MessageContentOutputTextContent, event["part"])
            completed["annotations"] = deepcopy(part.annotations)
            content.append(deepcopy(completed))
            yield event
        if self.item is not None:
            yield self.item.emit_done(
                {
                    "type": "message",
                    "id": self.item.item_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": content,
                }
            )
