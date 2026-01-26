"""Azure AI Foundry wrappers for LangChain agents.

This module provides wrapper classes that enable existing LangChain/LangGraph agents
to be exposed through Azure AI Foundry with enterprise features like:
- Azure RBAC and managed identity
- OpenTelemetry tracing to Application Insights
- Copilot Studio integration
- Unified REST API endpoints

The wrappers are designed to be non-invasive - they don't modify the original agents,
but instead create a thin proxy layer that routes requests through Azure AI Foundry
while preserving all original functionality.

Wrapper Types:
- FoundryAgentWrapper: Base wrapper class
- ITAgentWrapper: IT support agents (helpdesk, servicenow, hitl)
- EnterpriseAgentWrapper: Enterprise agents (research, content, data, etc.)
- DeepAgentWrapper: Multi-agent systems (it_ops, sales, recruitment)

Usage:
    from langchain_azure_ai.wrappers import (
        FoundryAgentWrapper,
        ITAgentWrapper,
        EnterpriseAgentWrapper,
        DeepAgentWrapper,
    )

    # Wrap an existing LangChain agent
    wrapped_agent = FoundryAgentWrapper(
        agent=my_langchain_agent,
        name="my-agent",
        description="My wrapped agent",
    )

    # Use through Azure AI Foundry
    result = await wrapped_agent.invoke({"messages": [...]})
"""

from langchain_azure_ai.wrappers.base import (
    AgentType,
    FoundryAgentWrapper,
    WrapperConfig,
)
from langchain_azure_ai.wrappers.it_agents import (
    ITAgentWrapper,
    ITHelpdeskWrapper,
    ServiceNowWrapper,
    HITLSupportWrapper,
)
from langchain_azure_ai.wrappers.enterprise_agents import (
    EnterpriseAgentWrapper,
    ResearchAgentWrapper,
    ContentAgentWrapper,
    DataAnalystWrapper,
    DocumentAgentWrapper,
    CodeAssistantWrapper,
    RAGAgentWrapper,
    DocumentIntelligenceWrapper,
)
from langchain_azure_ai.wrappers.deep_agents import (
    DeepAgentWrapper,
    SubAgentConfig,
    DeepAgentState,
    ITOperationsWrapper,
    SalesIntelligenceWrapper,
    RecruitmentWrapper,
)

__all__ = [
    # Base classes
    "AgentType",
    "FoundryAgentWrapper",
    "WrapperConfig",
    # IT Agent wrappers
    "ITAgentWrapper",
    "ITHelpdeskWrapper",
    "ServiceNowWrapper",
    "HITLSupportWrapper",
    # Enterprise Agent wrappers
    "EnterpriseAgentWrapper",
    "ResearchAgentWrapper",
    "ContentAgentWrapper",
    "DataAnalystWrapper",
    "DocumentAgentWrapper",
    "CodeAssistantWrapper",
    "RAGAgentWrapper",
    "DocumentIntelligenceWrapper",
    # DeepAgent wrappers
    "DeepAgentWrapper",
    "SubAgentConfig",
    "DeepAgentState",
    "ITOperationsWrapper",
    "SalesIntelligenceWrapper",
    "RecruitmentWrapper",
]
