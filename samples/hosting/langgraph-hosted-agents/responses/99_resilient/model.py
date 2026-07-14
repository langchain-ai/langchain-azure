"""A tiny fake chat model used by the resilient sample.

It streams a fixed ``reply`` one word or whitespace run at a time (no network,
no real LLM). A custom LangChain chat model emits ``messages`` events by
implementing ``_astream`` and calling ``run_manager.on_llm_new_token`` for
every token.
"""
from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class FakeChatModel(BaseChatModel):
    reply: str = "This is a fake answer."
    token_delay_seconds: float = float(os.environ.get("TOKEN_DELAY_SECONDS", "0.05"))
    cancellation_signal: Any = None

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
            if self._is_cancelled():
                return
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            # This callback is what makes the token show up in stream_mode="messages".
            if run_manager is not None:
                await run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
            if await self._wait_for_cancellation():
                return

    def _is_cancelled(self) -> bool:
        return bool(
            getattr(self.cancellation_signal, "is_set", lambda: False)()
        )

    async def _wait_for_cancellation(self) -> bool:
        if self.token_delay_seconds <= 0:
            return self._is_cancelled()
        wait = getattr(self.cancellation_signal, "wait", None)
        if wait is None:
            await asyncio.sleep(self.token_delay_seconds)
            return False
        try:
            await asyncio.wait_for(wait(), timeout=self.token_delay_seconds)
            return True
        except TimeoutError:
            return False

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
