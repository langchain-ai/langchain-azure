"""Sample 09 - Foundry short-term + long-term memory over the Responses API.

Combines the **two** Foundry-managed memory tiers into one hosted
LangGraph agent, and demonstrates the **two access patterns** for the
long-term tier side-by-side:

* **Short-term (thread state)** —
  :class:`langchain_azure_ai.checkpointers.FoundryCheckpointSaver`
  persists the per-``conversation.id`` checkpoint chain to Azure AI
  Foundry, so a single conversation survives server restarts (this is
  what sample 08 demonstrates on its own).

* **Long-term — Pattern B (context injection)** — a ``recall`` graph
  node always runs before the LLM, queries
  :class:`langchain_azure_ai.retrievers.AzureAIMemoryRetriever`, and
  feeds the top-k recalled facts as a *transient* ``SystemMessage``
  (stored in a side state field so it never pollutes the persisted
  message history).

* **Long-term — Pattern A (memory as a tool)** — the LLM is also bound
  with two ``@tool`` functions backed by the same store:

    - ``search_memory(query)`` — explicit recall the model can trigger
      mid-conversation (e.g., "let me check what I know about ...").
    - ``save_memory(fact)`` — explicit write the model can use to
      record derived insights it wants to remember next time.

  The model chooses when to call these; pattern B guarantees baseline
  recall even if it doesn't.

Both patterns use the **same** ``store_name`` + ``scope`` — the store
provider is agnostic to how it's consumed. See README's pattern A vs.
B discussion.

Graph shape::

    START → recall → agent ⇄ tools → END
                       ↑       │
                       └───────┘

Required environment variables (set in ``.env`` or your shell):

    AZURE_AI_PROJECT_ENDPOINT                       e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME                  e.g. gpt-4o   (defaults to "gpt-4o")
    MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME         e.g. gpt-4o   (used by the Memory Store for extraction)
    MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME    e.g. text-embedding-3-large
    MEMORY_STORE_NAME                               optional, defaults to ``langchain-azure-ai-demo-store``
    MEMORY_USER_SCOPE                               optional, defaults to ``user:demo-user``
    PORT                                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    python sample_09_responses_foundry_memory.py

Then drive **two separate conversations** for the same user, and watch
the agent recall a fact from the first conversation in the second::

    # Conversation A — introduce a preference and ask the model to remember it
    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"I prefer dark roast coffee with oat milk. Please remember that.","conversation":{"id":"ltm-demo-conv-A"}}'
    # The model is expected to call save_memory("User prefers dark roast ...")
    # via Pattern A.

    # ... wait ~30s for the Foundry Memory Store to finish extraction ...

    # Conversation B — fresh thread, same user → memory should recall
    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"What kind of coffee do I like?","conversation":{"id":"ltm-demo-conv-B"}}'
    # The recall node (Pattern B) injects the saved fact as context
    # before the model is even invoked.

Conversation B has no shared *checkpoint state* with conversation A —
the recall comes purely from Foundry's long-term Memory Store.

Notes
-----

* The Foundry Memory Store is **per-project**, identified by ``store_name``
  and scoped by the ``scope`` string. This sample uses a fixed
  ``MEMORY_USER_SCOPE`` so all conversations share the same long-term
  memory; in production, derive ``scope`` from your authenticated user
  (e.g. ``"user:{sub}"``).
* Memory extraction is **asynchronous** on the Foundry side. The first
  fact you mention may not be recalled for ~30s. ``update_delay=0`` in
  this sample asks for immediate extraction.
* ``FoundryCheckpointSaver`` and ``AzureAIMemoryRetriever`` are both
  **experimental**.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
)
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from openai.types.responses import EasyInputMessageParam
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import LangGraphResponsesAgentHost
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_azure_ai.checkpointers import FoundryCheckpointSaver
from langchain_azure_ai.retrievers import AzureAIMemoryRetriever

load_dotenv()

logger = logging.getLogger(__name__)

_AAD_SCOPE = "https://ai.azure.com/.default"
_DEFAULT_STORE_NAME = "langchain-azure-ai-demo-store"
_DEFAULT_USER_SCOPE = "user:demo-user"


# ---------------------------------------------------------------------------
# Memory Store setup (idempotent)
# ---------------------------------------------------------------------------


def _ensure_memory_store(
    project_endpoint: str, credential: Any, store_name: str
) -> None:
    """Create the Foundry Memory Store if it does not already exist."""
    client = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential,
        user_agent="langchain-azure-ai",
    )
    try:
        client.beta.memory_stores.get(store_name)
        logger.info("Memory store %r already exists.", store_name)
        return
    except ResourceNotFoundError:
        logger.info("Creating memory store %r ...", store_name)

    client.beta.memory_stores.create(
        name=store_name,
        description="LangChain Azure AI hosting sample 09 memory store.",
        definition=MemoryStoreDefaultDefinition(
            chat_model=os.environ["MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME"],
            embedding_model=os.environ[
                "MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME"
            ],
            options=MemoryStoreDefaultOptions(
                user_profile_enabled=True,
                chat_summary_enabled=True,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------


def _build_chat_model() -> ChatOpenAI:
    project_endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"].rstrip("/")
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    credential = DefaultAzureCredential()
    token = credential.get_token(_AAD_SCOPE).token
    return ChatOpenAI(
        model=deployment,
        api_key=token,  # type: ignore[arg-type]
        base_url=f"{project_endpoint}/openai/v1",
    )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


_AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to long-term memory about the "
    "current user.\n\n"
    "Two channels of memory are available to you:\n"
    "1. Before every turn the application may prepend a SystemMessage with "
    "facts already recalled from long-term memory. Use them when relevant.\n"
    "2. You also have two tools:\n"
    "   - search_memory(query): explicit recall when the recalled facts "
    "above are not enough.\n"
    "   - save_memory(fact): record an important derived insight (e.g., a "
    "stable user preference, a decision) that you want to remember in "
    "future conversations. Call this whenever the user states a clear "
    "preference or fact about themselves worth keeping.\n\n"
    "Prefer to answer from the recalled facts. Only call search_memory if "
    "the user asks about something not covered by the recalled facts."
)


class _AgentState(MessagesState):
    """Graph state with a transient ``recalled`` field for Pattern B.

    Storing recalled context in a separate field (instead of injecting a
    ``SystemMessage`` into ``messages``) keeps the persisted thread state
    free of per-turn recall noise.
    """

    recalled: List[str]


def _last_human_text(messages: List[Any]) -> str:
    """Return the most recent human message's text content (or empty)."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _build_graph(
    project_endpoint: str,
    store_name: str,
    scope: str,
    saver: FoundryCheckpointSaver,
) -> Any:
    llm = _build_chat_model()

    # Standalone retriever — incremental search would require sticky per-user
    # caching across requests, which is overkill for this demo. ``k=5`` keeps
    # the recall short enough to fit in the system prompt comfortably.
    retriever = AzureAIMemoryRetriever(
        project_endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
        store_name=store_name,
        scope=scope,
        k=5,
    )

    # Reused by the save_memory tool (Pattern A write).
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
        user_agent="langchain-azure-ai",
    )

    # -- Pattern A: memory exposed as tools --------------------------------

    @tool
    def search_memory(query: str) -> str:
        """Search the current user's long-term memory for facts about them.

        Use this when the facts already recalled above are not sufficient to
        answer the user's question.

        Args:
            query: Natural-language description of what to look up.
        """
        try:
            docs = retriever.invoke(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_memory failed: %s", exc)
            return f"Memory lookup failed: {exc}"
        if not docs:
            return "No matching memories."
        return "\n".join(f"- {d.page_content}" for d in docs)

    @tool
    def save_memory(fact: str) -> str:
        """Persist a derived fact about the user to long-term memory.

        Use this when the user states a stable preference, decision, or
        identity fact worth remembering across conversations.

        Args:
            fact: A single concise statement to remember. Phrase as a
                third-person fact (e.g. "User prefers dark roast coffee.").
        """
        try:
            project_client.beta.memory_stores.begin_update_memories(
                name=store_name,
                scope=scope,
                items=[EasyInputMessageParam(content=fact, role="user")],
                update_delay=0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("save_memory failed: %s", exc)
            return f"Failed to save: {exc}"
        return "Saved."

    tools = [search_memory, save_memory]
    llm_with_tools = llm.bind_tools(tools)

    # -- Pattern B: implicit context injection -----------------------------

    def recall(state: _AgentState) -> Dict[str, Any]:
        """Populate ``state['recalled']`` with top-k facts for this turn."""
        query = _last_human_text(state["messages"])
        if not query:
            return {"recalled": []}
        try:
            docs = retriever.invoke(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory recall failed: %s", exc)
            return {"recalled": []}
        return {"recalled": [d.page_content for d in docs]}

    def call_model(state: _AgentState) -> Dict[str, Any]:
        """Build the prompt = [agent system + recalled context] + messages.

        ``recalled`` is read from state but **not** persisted into the
        message history — it's regenerated fresh on every turn.
        """
        prefix: List[Any] = [SystemMessage(content=_AGENT_SYSTEM_PROMPT)]
        recalled = state.get("recalled") or []
        if recalled:
            recalled_text = "\n".join(f"- {r}" for r in recalled)
            prefix.append(
                SystemMessage(
                    content=(
                        "Recalled long-term memories about the current user "
                        "(from Azure AI Foundry Memory):\n\n" + recalled_text
                    )
                )
            )
        return {"messages": [llm_with_tools.invoke(prefix + state["messages"])]}

    workflow = StateGraph(_AgentState)
    workflow.add_node("recall", recall)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "recall")
    workflow.add_edge("recall", "agent")
    workflow.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=saver)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _amain() -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(auto_configure_azure_monitor=True)

    project_endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
    store_name = os.environ.get("MEMORY_STORE_NAME", _DEFAULT_STORE_NAME)
    scope = os.environ.get("MEMORY_USER_SCOPE", _DEFAULT_USER_SCOPE)
    port = int(os.environ.get("PORT", "8088"))

    # Idempotent one-time setup of the Foundry Memory Store.
    _ensure_memory_store(project_endpoint, DefaultAzureCredential(), store_name)

    async with AsyncDefaultAzureCredential() as cred:
        async with FoundryCheckpointSaver(
            project_endpoint=project_endpoint,
            credential=cred,
        ) as saver:
            graph = _build_graph(project_endpoint, store_name, scope, saver)
            await LangGraphResponsesAgentHost(graph).run_async(
                host="127.0.0.1", port=port
            )


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
