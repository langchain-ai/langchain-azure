"""A tiny fake chat model used by the resilient sample.

It streams a fixed ``reply`` one word or whitespace run at a time (no network,
no real LLM). A custom LangChain chat model emits ``messages`` events by
implementing ``_astream`` and calling ``run_manager.on_llm_new_token`` for
every token.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any
import os
import asyncio

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

TOKEN_DELAY_SECONDS = float(os.environ.get("TOKEN_DELAY_SECONDS", "0.05"))

class FakeChatModel(BaseChatModel):
    reply: str = "This is a fake answer."

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for token in re.findall(r"\S+|\s+", self.reply):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            # This callback is what makes the token show up in stream_mode="messages".
            if run_manager is not None:
                await run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
            await asyncio.sleep(TOKEN_DELAY_SECONDS)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Non-streaming fallback: return the whole reply at once.
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.reply))]
        )
