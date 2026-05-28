"""Helpers for Azure AI Foundry Memory message items."""

from typing import Literal

from openai.types.responses import EasyInputMessageParam


def build_foundry_message_item(
    content: str, role: Literal["user", "assistant", "system", "developer"]
) -> EasyInputMessageParam:
    """Build a Foundry Memory message item with Azure's required explicit type.

    Args:
        content: The text content to send to the memory store.
        role: The Foundry/OpenAI message role associated with the content.

    Returns:
        An ``EasyInputMessageParam`` shaped for Azure AI Foundry Memory.

    Azure AI Foundry Memory expects message items to include
    ``type="message"`` explicitly, even though some upstream helpers omit it.
    """
    return EasyInputMessageParam(content=content, role=role, type="message")
