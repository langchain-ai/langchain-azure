"""Azure AI Foundry integration with LangChain/LangGraph.

This package provides integration between LangChain/LangGraph agents
and Azure AI Foundry, enabling enterprise features such as:

- Azure AI Foundry agent creation and management
- OpenTelemetry tracing with Application Insights
- Azure RBAC and managed identity integration
- Unified REST API and chat UI endpoints

Modules:
- agents: Azure AI Foundry agent creation and management
- wrappers: Wrapper classes for IT, Enterprise, and DeepAgents
- server: FastAPI server with REST endpoints and chat UI
- config: Configuration management
- chat_models: Azure Chat model integrations
- embeddings: Azure embedding integrations
- tools: Azure AI tools (Document Intelligence, etc.)
- vectorstores: Azure vector store integrations

Usage:
    from langchain_azure_ai.wrappers import (
        ITAgentWrapper,
        EnterpriseAgentWrapper,
        DeepAgentWrapper,
    )
    
    # Create an IT Helpdesk agent
    helpdesk = ITAgentWrapper(
        name="it-helpdesk",
        agent_subtype="helpdesk",
        tools=[...],
    )
    
    # Chat with the agent
    response = helpdesk.chat("I need help with my password")
"""

__version__ = "1.0.4"

__all__ = ["__version__"]

