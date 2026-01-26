"""Base wrapper for LangChain/LangGraph agents with Azure AI Foundry integration.

This module provides the foundational wrapper class that enables any LangChain
or LangGraph agent to be exposed through Azure AI Foundry's enterprise features.

The wrapper is designed to be non-invasive and backward compatible:
- When Azure AI Foundry is disabled (USE_AZURE_FOUNDRY=false), agents work normally
- When enabled, requests are routed through Azure AI Foundry for enterprise features
- Original agent functionality is always preserved
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Types of agents supported by the wrapper system."""

    IT_AGENT = "it_agent"
    ENTERPRISE_AGENT = "enterprise_agent"
    DEEP_AGENT = "deep_agent"
    CUSTOM = "custom"


@dataclass
class WrapperConfig:
    """Configuration for the Azure AI Foundry wrapper.

    Attributes:
        use_azure_foundry: Whether to use Azure AI Foundry integration.
            If False, agents work with direct Azure OpenAI connection.
        project_endpoint: Azure AI Foundry project endpoint.
        enable_tracing: Whether to enable OpenTelemetry tracing.
        app_insights_connection: Application Insights connection string.
        langsmith_enabled: Whether to also use LangSmith tracing.
        feature_flags: Additional feature flags for customization.
    """

    use_azure_foundry: bool = False
    project_endpoint: Optional[str] = None
    enable_tracing: bool = True
    app_insights_connection: Optional[str] = None
    langsmith_enabled: bool = True
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "WrapperConfig":
        """Create configuration from environment variables.

        Environment Variables:
            USE_AZURE_FOUNDRY: Enable Azure AI Foundry integration (default: false)
            AZURE_AI_PROJECT_ENDPOINT: Azure AI Foundry project endpoint
            ENABLE_TRACING: Enable OpenTelemetry tracing (default: true)
            APPLICATIONINSIGHTS_CONNECTION_STRING: App Insights connection
            LANGCHAIN_TRACING_V2: Enable LangSmith tracing (default: true)
        """
        return cls(
            use_azure_foundry=os.getenv("USE_AZURE_FOUNDRY", "false").lower() == "true",
            project_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
            enable_tracing=os.getenv("ENABLE_TRACING", "true").lower() == "true",
            app_insights_connection=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
            langsmith_enabled=os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true",
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if self.use_azure_foundry and not self.project_endpoint:
            issues.append(
                "USE_AZURE_FOUNDRY=true but AZURE_AI_PROJECT_ENDPOINT not set"
            )

        if self.enable_tracing and not self.app_insights_connection:
            logger.warning(
                "Tracing enabled but APPLICATIONINSIGHTS_CONNECTION_STRING not set. "
                "Azure Monitor tracing will be disabled."
            )

        return issues


class FoundryAgentWrapper(ABC):
    """Base wrapper for integrating LangChain/LangGraph agents with Azure AI Foundry.

    This wrapper provides a thin proxy layer that:
    1. Routes requests through Azure AI Foundry when enabled
    2. Adds enterprise features (RBAC, tracing, governance)
    3. Preserves original agent functionality
    4. Supports both sync and async invocation

    The wrapper is designed to be subclassed for specific agent types:
    - ITAgentWrapper: For IT support agents (helpdesk, servicenow)
    - EnterpriseAgentWrapper: For enterprise agents (research, content, etc.)
    - DeepAgentWrapper: For deep agents with multi-agent orchestration

    Example:
        ```python
        # Create a simple wrapped agent
        from langchain_azure_ai.wrappers import FoundryAgentWrapper

        class MyAgentWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return create_react_agent(llm, tools)

        wrapper = MyAgentWrapper(
            name="my-agent",
            instructions="You are a helpful assistant.",
        )

        result = wrapper.invoke({"messages": [HumanMessage(content="Hello!")]})
        ```
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        agent_type: AgentType = AgentType.CUSTOM,
        description: Optional[str] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        config: Optional[WrapperConfig] = None,
        existing_agent: Optional[Union[CompiledStateGraph, Any]] = None,
    ):
        """Initialize the wrapper.

        Args:
            name: Name of the agent (used for Azure AI Foundry registration).
            instructions: System instructions/prompt for the agent.
            agent_type: Type of agent (IT, Enterprise, Deep, Custom).
            description: Optional description of the agent.
            tools: List of tools available to the agent.
            model: Model deployment name to use.
            temperature: Temperature for model responses.
            config: Wrapper configuration. If None, loads from environment.
            existing_agent: An existing LangChain/LangGraph agent to wrap.
                If provided, the wrapper will use this agent instead of creating a new one.
        """
        self.name = name
        self.instructions = instructions
        self.agent_type = agent_type
        self.description = description or f"{agent_type.value}: {name}"
        self.tools = list(tools) if tools else []
        self.model = model
        self.temperature = temperature
        self.config = config or WrapperConfig.from_env()

        # Validate configuration
        issues = self.config.validate()
        if issues:
            for issue in issues:
                logger.warning(f"Configuration issue: {issue}")

        # The underlying agent (created lazily or provided)
        self._existing_agent = existing_agent
        self._agent: Optional[CompiledStateGraph] = None
        self._foundry_agent_id: Optional[str] = None
        
        # Initialize telemetry (if available)
        self._telemetry = None
        try:
            from langchain_azure_ai.observability import AgentTelemetry
            self._telemetry = AgentTelemetry(
                agent_name=self.name,
                agent_type=self.agent_type.value if hasattr(self.agent_type, 'value') else str(self.agent_type),
            )
        except ImportError:
            pass

        # Initialize the agent
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the agent based on configuration."""
        if self._existing_agent is not None:
            logger.info(f"Using existing agent for {self.name}")
            self._agent = self._existing_agent
        elif self.config.use_azure_foundry and self.config.project_endpoint:
            logger.info(f"Creating Azure AI Foundry agent: {self.name}")
            self._create_foundry_agent()
        else:
            logger.info(f"Creating direct LangChain agent: {self.name}")
            self._create_direct_agent()

    def _create_foundry_agent(self) -> None:
        """Create agent through Azure AI Foundry."""
        try:
            from langchain_azure_ai.agents import AgentServiceFactory

            factory = AgentServiceFactory(
                project_endpoint=self.config.project_endpoint,
            )

            self._agent = factory.create_prompt_agent(
                name=self.name,
                model=self.model,
                description=self.description,
                instructions=self.instructions,
                tools=self.tools if self.tools else None,
                temperature=self.temperature,
                trace=self.config.enable_tracing,
            )

            # Store agent ID for cleanup
            agent_ids = factory.get_agents_id_from_graph(self._agent)
            if agent_ids:
                self._foundry_agent_id = next(iter(agent_ids))

            logger.info(f"Created Azure AI Foundry agent: {self._foundry_agent_id}")

        except Exception as e:
            logger.error(f"Failed to create Azure AI Foundry agent: {e}")
            logger.warning(
                "Falling back to direct LangChain agent after Azure AI Foundry initialization failure"
            )
            self._create_direct_agent()

    def _create_direct_agent(self) -> None:
        """Create agent with direct Azure OpenAI connection."""
        try:
            llm = self._get_llm()
            self._agent = self._create_agent_impl(llm, self.tools)
            logger.info(f"Created direct LangChain agent: {self.name}")
        except Exception as e:
            logger.error(f"Failed to create direct agent: {e}")
            raise

    def _get_llm(self) -> BaseChatModel:
        """Get the LLM instance based on configuration.

        Returns:
            BaseChatModel instance configured for Azure OpenAI.
        """
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", self.model),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
            temperature=self.temperature,
        )

    @abstractmethod
    def _create_agent_impl(
        self,
        llm: BaseChatModel,
        tools: List[Union[BaseTool, Callable]],
    ) -> CompiledStateGraph:
        """Create the underlying agent implementation.

        This method must be implemented by subclasses to define how the
        specific agent type is created.

        Args:
            llm: The language model to use.
            tools: List of tools for the agent.

        Returns:
            A compiled LangGraph StateGraph representing the agent.
        """
        pass

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Invoke the agent synchronously.

        Args:
            input: Input dictionary with 'messages' key.
            config: Optional configuration for the invocation.
            **kwargs: Additional keyword arguments.

        Returns:
            Agent output dictionary with 'messages' key.
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized")

        # Use telemetry if available
        if self._telemetry:
            with self._telemetry.track_execution("invoke") as metrics:
                result = self._agent.invoke(input, config=config, **kwargs)
                # Try to extract token usage
                if isinstance(result, dict):
                    usage = result.get("usage", {})
                    metrics.prompt_tokens = usage.get("prompt_tokens", 0)
                    metrics.completion_tokens = usage.get("completion_tokens", 0)
                return result
        else:
            return self._agent.invoke(input, config=config, **kwargs)

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Invoke the agent asynchronously.

        Args:
            input: Input dictionary with 'messages' key.
            config: Optional configuration for the invocation.
            **kwargs: Additional keyword arguments.

        Returns:
            Agent output dictionary with 'messages' key.
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized")

        return await self._agent.ainvoke(input, config=config, **kwargs)

    def stream(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Stream agent responses.

        Args:
            input: Input dictionary with 'messages' key.
            config: Optional configuration for the invocation.
            **kwargs: Additional keyword arguments.

        Yields:
            Agent output chunks.
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized")

        yield from self._agent.stream(input, config=config, **kwargs)

    def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Simple chat interface for the agent.

        This is a convenience method that wraps invoke() with a simpler interface.

        Args:
            message: The user message to send.
            thread_id: Optional thread ID for conversation continuity.
            **kwargs: Additional keyword arguments.

        Returns:
            The agent's response as a string.
        """
        import uuid
        import time

        start_time = time.perf_counter()
        
        input_dict = {"messages": [HumanMessage(content=message)]}

        # Always provide a thread_id for checkpointer (generate one if not provided)
        effective_thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": effective_thread_id}}

        # Log request if telemetry available
        if self._telemetry:
            self._telemetry.log_request(message, effective_thread_id)

        result = self.invoke(input_dict, config=config, **kwargs)

        # Extract response from messages
        messages = result.get("messages", [])
        response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                response = msg.content
                break

        # Log response if telemetry available
        if self._telemetry:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._telemetry.log_response(response, effective_thread_id, duration_ms)

        return response

    def cleanup(self) -> None:
        """Clean up resources, including Azure AI Foundry agent."""
        if self._foundry_agent_id and self.config.project_endpoint:
            try:
                from langchain_azure_ai.agents import AgentServiceFactory

                factory = AgentServiceFactory(
                    project_endpoint=self.config.project_endpoint,
                )
                factory.delete_agent(self._agent)
                logger.info(f"Deleted Azure AI Foundry agent: {self._foundry_agent_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup Azure AI Foundry agent: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False

    @property
    def agent(self) -> Optional[CompiledStateGraph]:
        """Get the underlying agent."""
        return self._agent

    @property
    def is_foundry_enabled(self) -> bool:
        """Check if Azure AI Foundry integration is enabled."""
        return self._foundry_agent_id is not None
