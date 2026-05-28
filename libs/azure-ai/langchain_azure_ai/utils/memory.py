"""Helpers for Azure AI Foundry Memory message items."""

from typing import Literal

from openai.types.responses import EasyInputMessageParam


def build_foundry_message_item(
    content: str, role: Literal["user", "assistant", "system", "developer"]
) -> EasyInputMessageParam:
    """Build a Foundry memory message item with the required explicit type."""
    return EasyInputMessageParam(content=content, role=role, type="message")
