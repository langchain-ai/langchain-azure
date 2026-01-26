"""IT Agent wrappers for Azure AI Foundry integration.

This module provides wrappers for IT support agents:
- IT Helpdesk Agent: General IT support with knowledge base
- ServiceNow Agent: ITSM integration with ticket management
- HITL Support Agent: Human-in-the-loop support escalation

These wrappers preserve the original agent functionality while adding
Azure AI Foundry enterprise features.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from langchain_azure_ai.wrappers.base import (
    AgentType,
    FoundryAgentWrapper,
    WrapperConfig,
)

logger = logging.getLogger(__name__)


class ITAgentWrapper(FoundryAgentWrapper):
    """Wrapper for IT support agents with Azure AI Foundry integration.

    This wrapper is designed for IT support use cases:
    - IT Helpdesk: General IT support, password resets, access requests
    - ServiceNow: ITSM ticket management, incident handling
    - HITL: Human-in-the-loop escalation support

    Example:
        ```python
        from langchain_azure_ai.wrappers import ITAgentWrapper

        # Create IT Helpdesk agent
        helpdesk = ITAgentWrapper(
            name="it-helpdesk",
            instructions="You are an IT Helpdesk agent...",
            agent_subtype="helpdesk",
            tools=[search_kb, create_ticket, check_status],
        )

        response = helpdesk.chat("I need to reset my password")
        ```
    """

    # Default instructions for different IT agent subtypes
    DEFAULT_INSTRUCTIONS = {
        "helpdesk": """You are an IT Helpdesk Agent. Your role is to:
1. Help users with common IT issues (password resets, access requests, software installation)
2. Search the knowledge base for solutions
3. Create support tickets when issues need escalation
4. Provide clear, step-by-step instructions

Always be helpful, professional, and patient. If you cannot resolve an issue, 
create a ticket and provide the ticket number to the user.""",
        "servicenow": """You are a ServiceNow ITSM Agent. Your role is to:
1. Manage IT service requests and incidents
2. Create, update, and track tickets
3. Query ticket status and history
4. Escalate critical issues appropriately

Use the ServiceNow tools to interact with the ITSM system. Always provide
ticket numbers and status updates to users.""",
        "hitl": """You are a Human-in-the-Loop Support Agent. Your role is to:
1. Handle complex issues that require human judgment
2. Prepare issue summaries for human reviewers
3. Route issues to appropriate support tiers
4. Track escalation status

When uncertain, escalate to human review with a clear summary of the issue.""",
    }

    def __init__(
        self,
        name: str,
        instructions: Optional[str] = None,
        agent_subtype: str = "helpdesk",
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        config: Optional[WrapperConfig] = None,
        existing_agent: Optional[Union[CompiledStateGraph, Any]] = None,
        enable_memory: bool = True,
        **kwargs: Any,
    ):
        """Initialize IT agent wrapper.

        Args:
            name: Name of the agent.
            instructions: System instructions. If None, uses default for subtype.
            agent_subtype: Type of IT agent (helpdesk, servicenow, hitl).
            tools: List of tools for the agent.
            model: Model deployment name.
            temperature: Temperature for responses.
            config: Wrapper configuration.
            existing_agent: Existing agent to wrap.
            enable_memory: Whether to enable conversation memory.
            **kwargs: Additional arguments.
        """
        self.agent_subtype = agent_subtype.lower()
        self.enable_memory = enable_memory
        self._memory = MemorySaver() if enable_memory else None

        # Use default instructions if not provided
        if instructions is None:
            instructions = self.DEFAULT_INSTRUCTIONS.get(
                self.agent_subtype,
                self.DEFAULT_INSTRUCTIONS["helpdesk"],
            )

        super().__init__(
            name=name,
            instructions=instructions,
            agent_type=AgentType.IT_AGENT,
            description=f"IT Agent ({agent_subtype}): {name}",
            tools=tools,
            model=model,
            temperature=temperature,
            config=config,
            existing_agent=existing_agent,
        )

    def _create_agent_impl(
        self,
        llm: BaseChatModel,
        tools: List[Union[BaseTool, Callable]],
    ) -> CompiledStateGraph:
        """Create IT agent using LangGraph's create_react_agent.

        IT agents use the ReAct pattern with:
        - System prompt defining the agent's role
        - Tools for IT operations
        - Memory for conversation context

        Args:
            llm: The language model to use.
            tools: List of tools for the agent.

        Returns:
            A compiled LangGraph StateGraph.
        """
        # Create the agent with optional memory
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SystemMessage(content=self.instructions),
            checkpointer=self._memory,
        )

        return agent

    def chat_with_session(
        self,
        message: str,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Chat with session management for IT support workflows.

        Args:
            message: User message.
            session_id: Session ID for conversation continuity.
            user_id: Optional user ID for tracking.
            metadata: Optional metadata.

        Returns:
            Response dictionary with session info and agent response.
        """
        configurable: Dict[str, Any] = {
            "thread_id": session_id,
        }

        if metadata:
            configurable["metadata"] = metadata
        if user_id:
            configurable["user_id"] = user_id

        config: Dict[str, Any] = {"configurable": configurable}

        # Extract response
        response_text = self.chat(message, thread_id=session_id)

        return {
            "session_id": session_id,
            "agent_type": self.agent_subtype,
            "response": response_text,
            "user_id": user_id,
        }


class ITHelpdeskWrapper(ITAgentWrapper):
    """Specialized wrapper for IT Helpdesk agents.

    This is a convenience class that pre-configures the ITAgentWrapper
    for helpdesk use cases.
    """

    def __init__(
        self,
        name: str = "it-helpdesk",
        instructions: Optional[str] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="helpdesk",
            tools=tools,
            **kwargs,
        )


class ServiceNowWrapper(ITAgentWrapper):
    """Specialized wrapper for ServiceNow ITSM agents.

    This is a convenience class that pre-configures the ITAgentWrapper
    for ServiceNow integration.
    """

    def __init__(
        self,
        name: str = "servicenow-agent",
        instructions: Optional[str] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="servicenow",
            tools=tools,
            **kwargs,
        )


class HITLSupportWrapper(ITAgentWrapper):
    """Specialized wrapper for Human-in-the-Loop support agents.

    This is a convenience class that pre-configures the ITAgentWrapper
    for HITL escalation workflows.
    """

    def __init__(
        self,
        name: str = "hitl-support",
        instructions: Optional[str] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="hitl",
            tools=tools,
            **kwargs,
        )
