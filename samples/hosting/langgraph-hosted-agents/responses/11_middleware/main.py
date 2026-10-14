"""Configure the model and checkpoint store; the host loads build_graph."""

from __future__ import annotations

import logging
import os

from agent import build_agent
from azure.ai.agentserver.core import AgentConfig
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_azure_ai.agents.hosting import FoundryCheckpointSaver
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph


def build_graph() -> CompiledStateGraph:
    """Build the unchanged middleware agent for local or Foundry execution."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    credential = DefaultAzureCredential()
    project = AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"], credential=credential
    )
    openai_client = project.get_openai_client()
    model = ChatOpenAI(
        model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini"),
        base_url=str(openai_client.base_url),
        api_key=get_bearer_token_provider(credential, "https://ai.azure.com/.default"),
        use_responses_api=True,
        output_version="responses/v1",
    )
    checkpointer: BaseCheckpointSaver = (
        FoundryCheckpointSaver()
        if AgentConfig.from_env().is_hosted
        else InMemorySaver()
    )
    return build_agent(model, checkpointer)
