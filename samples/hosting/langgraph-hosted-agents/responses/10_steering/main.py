"""Sample 10 - Steerable long-running Responses API host."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from langchain_azure_ai.agents.hosting import ResponsesHostServer

load_dotenv()
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

_AZURE_AI_SCOPE = "https://ai.azure.com/.default"


def _build_chat_model() -> ChatOpenAI:
    project_endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/")
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    credential = DefaultAzureCredential()
    project = AIProjectClient(endpoint=project_endpoint, credential=credential)
    openai_client = project.get_openai_client()
    token_provider = get_bearer_token_provider(credential, _AZURE_AI_SCOPE)

    return ChatOpenAI(
        model=deployment,
        base_url=str(openai_client.base_url),
        api_key=token_provider,
    )


def _last_user_text(state: MessagesState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    return "the current topic"


async def steerable_work(
    state: MessagesState, config: RunnableConfig, model: ChatOpenAI
) -> dict:
    """Run a long workflow node that cooperates with steering cancellation."""
    topic = _last_user_text(state)
    steps = int(os.environ.get("STEERING_STEPS", "60"))
    delay = float(os.environ.get("STEERING_STEP_SECONDS", "1"))
    cancellation = (config.get("configurable") or {}).get(
        "foundry_cancellation_signal"
    )

    logger.info("steering_sample_started", extra={"steps": steps, "delay": delay})
    notes: list[str] = []
    for step in range(steps):
        if _is_set(cancellation):
            logger.warning("steering_sample_cancellation_observed")
            notes.append(
                f"Stopped at step {step + 1}/{steps} because a newer turn was queued."
            )
            return {"messages": [AIMessage(content="\n".join(notes))]}
        await asyncio.sleep(delay)
        notes.append(f"step {step + 1}/{steps}: working on {topic!r}")

    logger.info("steering_sample_invoking_foundry_model")
    notes_text = "\n".join(notes)
    completion = await model.ainvoke(
        [
            HumanMessage(
                content=(
                    "Summarize this steerable workflow in one sentence:\n"
                    f"{notes_text}"
                )
            )
        ]
    )
    content = f"{notes_text}\n\n{completion.content}"
    return {"messages": [AIMessage(content=content)]}


def _is_set(value: Any) -> bool:
    return bool(getattr(value, "is_set", lambda: False)())


def _build_graph():
    model = _build_chat_model()

    async def steerable_node(state: MessagesState, config: RunnableConfig) -> dict:
        return await steerable_work(state, config, model)

    builder = StateGraph(MessagesState)
    builder.add_node("steerable_work", steerable_node)
    builder.add_edge(START, "steerable_work")
    builder.add_edge("steerable_work", END)
    return builder.compile(checkpointer=MemorySaver())


def main() -> None:
    """Start the Responses API host."""
    port = int(os.environ.get("PORT", "8088"))
    logger.info("steering_sample_host_starting", extra={"port": port})
    ResponsesHostServer(
        _build_graph(),
        resilient_background=True,
        steerable_conversations=True,
    ).run(port=port)


if __name__ == "__main__":
    main()
