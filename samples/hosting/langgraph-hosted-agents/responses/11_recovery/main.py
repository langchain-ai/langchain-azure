"""Sample 11 - Server-side recovery for background Responses."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from langchain_azure_ai.agents.hosting import ResponsesHostServer, RetryPolicy

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


def _state_dir() -> Path:
    default_state_dir = Path.home() / ".langgraph-task-api-recovery"
    path = Path(os.environ.get("RECOVERY_STATE_DIR") or str(default_state_dir))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_thread_id(config: RunnableConfig) -> str:
    raw = str((config.get("configurable") or {}).get("thread_id") or "default")
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)


async def recoverable_work(
    state: MessagesState, config: RunnableConfig, model: ChatOpenAI
) -> dict:
    """Run a workflow node that uses a durable watermark for recovery."""
    del state
    marker = _state_dir() / f"{_safe_thread_id(config)}.phase1"

    if not marker.exists():
        logger.info("recovery_sample_phase1_committing", extra={"marker": str(marker)})
        marker.write_text("phase 1 committed\n", encoding="utf-8")
        if os.environ.get("DEMO_CRASH_ONCE") == "1":
            logger.critical("recovery_sample_demo_crash_requested")
            logging.shutdown()
            os._exit(137)
        completion = await model.ainvoke(
            [
                HumanMessage(
                    content=(
                        "In one sentence, confirm phase 1 of a recoverable "
                        "workflow committed successfully."
                    )
                )
            ]
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Phase 1 committed; no crash requested.\n\n"
                        f"{completion.content}"
                    )
                )
            ]
        }

    logger.warning("recovery_sample_phase1_marker_found", extra={"marker": str(marker)})
    completion = await model.ainvoke(
        [
            HumanMessage(
                content=(
                    "In one sentence, explain that a recoverable workflow "
                    "resumed from an existing durable watermark."
                )
            )
        ]
    )
    return {
        "messages": [
            AIMessage(
                content=(
                    "Recovered server-side: phase 1 watermark already existed, "
                    "so the graph completed phase 2 after re-entry.\n\n"
                    f"{completion.content}"
                )
            )
        ]
    }


def _build_graph():
    model = _build_chat_model()

    async def recoverable_node(state: MessagesState, config: RunnableConfig) -> dict:
        return await recoverable_work(state, config, model)

    builder = StateGraph(MessagesState)
    builder.add_node("recoverable_work", recoverable_node)
    builder.add_edge(START, "recoverable_work")
    builder.add_edge("recoverable_work", END)
    return builder.compile()


def main() -> None:
    """Start the Responses API host."""
    port = int(os.environ.get("PORT", "8088"))
    logger.info("recovery_sample_host_starting", extra={"port": port})
    ResponsesHostServer(
        _build_graph(),
        resilient_background=True,
        task_retry=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=2,
            retry_on=(TimeoutError, ConnectionError),
        ),
    ).run(port=port)


if __name__ == "__main__":
    main()
