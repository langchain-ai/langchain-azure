# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared Responses text-part conversion and citation lifecycle."""

from collections.abc import Iterable, Iterator
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
    TextContentBuilder,
)
from openai.types.responses.response_output_text import Annotation
from pydantic import TypeAdapter, ValidationError

_ANNOTATION_ADAPTER: TypeAdapter[Annotation] = TypeAdapter(Annotation)


def _response_annotation(annotation: Any) -> WireAnnotation | None:
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
        return cast(
            WireAnnotation,
            _ANNOTATION_ADAPTER.validate_python(value, strict=True).model_dump(),
        )
    except ValidationError:
        return None


@dataclass(frozen=True)
class _TextDelta:
    index: object
    text: str
    annotations: list[WireAnnotation]


def text_deltas(content: str | list[str | dict[str, Any]]) -> Iterator[_TextDelta]:
    """Normalize supported text blocks before changing any stream state."""
    blocks: list[str | dict[str, Any]] = (
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
            value
            for annotation in block.get("annotations") or []
            if (value := _response_annotation(annotation)) is not None
        ]
        if text or annotations:
            yield _TextDelta(block.get("index", object()), text, annotations)


@dataclass
class _TextPart:
    builder: TextContentBuilder
    annotations: list[WireAnnotation] = field(default_factory=list)


class _TextMessageEmitter:
    """Own one output item and its indexed parts using public SDK builders.

    SDK 2.1.0b2 discards annotations in its message completion builders.
    Keep that workaround here until supported SDK versions preserve them.
    """

    def __init__(self, stream: ResponseEventStream) -> None:
        self._stream = stream
        self._item: OutputItemBuilder | None = None
        self._parts: dict[object, _TextPart] = {}

    def add(self, deltas: Iterable[_TextDelta]) -> Iterator[Any]:
        """Emit text immediately and retain citations until the part completes."""
        for delta in deltas:
            if self._item is None:
                # Reserve the SDK-generated message ID and output position.
                message = self._stream.add_output_item_message()
                self._item = OutputItemBuilder(
                    self._stream, message.output_index, message.item_id
                )
                yield self._item.emit_added(
                    {
                        "type": "message",
                        "id": self._item.item_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    }
                )
            if delta.index not in self._parts:
                builder = TextContentBuilder(
                    self._stream,
                    self._item.output_index,
                    len(self._parts),
                    self._item.item_id,
                )
                self._parts[delta.index] = _TextPart(builder)
                yield builder.emit_added()
            part = self._parts[delta.index]
            if delta.text:
                yield part.builder.emit_delta(delta.text)
            part.annotations.extend(delta.annotations)

    def close(self) -> Iterator[Any]:
        """Complete the message with the exact content sent in part events."""
        content: list[MessageContent] = []
        for part in self._parts.values():
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
        if self._item is not None:
            yield self._item.emit_done(
                {
                    "type": "message",
                    "id": self._item.item_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": content,
                }
            )
