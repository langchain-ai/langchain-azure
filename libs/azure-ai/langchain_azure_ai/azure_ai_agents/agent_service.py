"""Azure AI Agents integration for LangChain."""

import logging
from typing import Any, Dict, List, Optional

from azure.ai.agents.models import Agent
from azure.ai.projects import AIProjectClient
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM
from langchain_core.language_models.llms import LLMResult
from langchain_core.outputs import Generation
from pydantic import Field, PrivateAttr, model_validator

logger = logging.getLogger(__name__)


class AzureAIAgentsService(BaseLLM):
    """Azure AI Agents service integration for LangChain.

    This class provides a LangChain-compatible interface to Azure AI Agents,
    enabling seamless integration of Azure AI Agents with LangChain workflows.
    It uses the Azure AI Projects SDK and Azure AI Agents SDK with a direct endpoint.

    The service automatically manages agent lifecycle, thread creation/cleanup,
    and message handling while providing the standard LangChain LLM interface
    for use in chains, prompt templates, and other LangChain components.

    Supported Configuration Parameters:
    - temperature: Controls randomness in responses (0.0 = deterministic,
      1.0 = very random)
    - top_p: Controls diversity via nucleus sampling (0.0 to 1.0)
    - response_format: Specifies output format (e.g., JSON schema)
    - tools: List of tools the agent can use (code_interpreter, file_search, functions)
    - tool_resources: Resources for tools (file IDs, vector stores, etc.)
    - metadata: Custom key-value pairs for tracking and organization

    Authentication Options:
    1. Default Credential or other Token Credential

    Examples:
        Using with direct endpoint:

        .. code-block:: python
            agent_service = AzureAIAgentsService(
                endpoint="https://your-resource.inference.ai.azure.com",
                credential=DefaultAzureCredential(),
                model="gpt-4",
                agent_name="my-agent",
                instructions="You are a helpful assistant"
            )

            # Simple text generation
            response = agent_service.invoke("What is machine learning?")

        Basic usage with endpoint:

        .. code-block:: python
            from langchain_azure_ai.azure_ai_agents import AzureAIAgentsService
            from azure.identity import DefaultAzureCredential

            agent_service = AzureAIAgentsService(
                endpoint="https://your-resource.inference.ai.azure.com",
                credential=DefaultAzureCredential(),
                model="gpt-4",
                agent_name="my-helpful-agent",
                instructions="You are a helpful assistant specialized in data "
                "analysis.",
                temperature=0.7
            )

            # Simple text generation
            response = agent_service.invoke("What is machine learning?")

        Integration with LangChain chains:

        .. code-block:: python
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = PromptTemplate(
                template="Explain {topic} in exactly 3 bullet points."
            )
            chain = prompt | agent_service | StrOutputParser()

            result = chain.invoke({"topic": "quantum computing"})

        Using with tools (code interpreter example):

        .. code-block:: python
            from azure.ai.agents.models import CodeInterpreterTool

            code_tool = CodeInterpreterTool(file_ids=["file-123"])

            agent_service = AzureAIAgentsService(
                endpoint="https://your-resource.inference.ai.azure.com",
                credential=DefaultAzureCredential(),
                model="gpt-4",
                tools=code_tool.definitions,
                tool_resources=code_tool.resources,
                instructions="You are a data analyst. Use Python to analyze data files."
            )
    """

    endpoint: str
    """The direct endpoint URI for Azure AI Agents service or from an 
    Azure AI Foundry model.
    
    Use this for direct access to Azure AI Agents.
    
    Format: https://your-resource.inference.ai.azure.com
    
    This parameter is required.
    """

    credential: Optional[TokenCredential] = None
    """Authentication credential for the Azure AI service.
    
    Supported types:
    - TokenCredential: Azure AD credential (like DefaultAzureCredential)
    
    If None, DefaultAzureCredential() is used for automatic Azure AD authentication.
    """

    model: Optional[str] = None
    """The name/ID of the model deployment to use for the agent.
    
    This should match a model deployment in your Azure AI project or resource.
    Common examples: 'gpt-4.1', 'gpt-4o-mini', 'deepseek-r1'.
    
    The model determines the agent's capabilities, cost, and performance
    characteristics.
    """

    agent_name: str = Field(default="langchain-agent")
    """A descriptive name for the agent instance.
    
    This name helps identify the agent in logs, Azure portal, and when managing
    multiple agents. Choose a name that describes the agent's purpose or role.
    """

    instructions: str = Field(default="You are a helpful AI assistant.")
    """System instructions that define the agent's behavior and personality.
    
    This is the most important configuration for controlling how your agent behaves.
    Be specific about the agent's role, capabilities, tone, and any constraints.
    
    Example: 'You are a financial analyst. Provide clear, data-driven insights
    and always cite your sources. Be conservative in your recommendations.'
    """

    agent_description: Optional[str] = None
    """Optional human-readable description of the agent's purpose.
    
    This description is used for documentation and management purposes.
    It doesn't affect the agent's behavior but helps with organization.
    """

    tools: Optional[List[Dict[str, Any]]] = None
    """List of tool definitions that the agent can use during conversations.
    
    Tools extend the agent's capabilities beyond text generation. Common tools:
    - code_interpreter: Execute Python code and analyze data
    - file_search: Search through uploaded files
    - function: Call custom functions you define
    
    Use the Azure AI Agents SDK to create proper tool definitions.
    """

    tool_resources: Optional[Any] = None
    """Resources required by the agent's tools (file IDs, vector stores, etc.).
    
    This contains the actual resources that tools need to operate:
    - File IDs for code_interpreter and file_search tools
    - Vector store configurations
    - Function schemas and implementations
    
    Must correspond to the tools defined in the 'tools' parameter.
    """

    metadata: Optional[Dict[str, Any]] = None
    """Custom key-value pairs for tracking and organizing agents.
    
    Use metadata to store information about the agent that helps with:
    - Project organization and categorization  
    - Usage tracking and analytics
    - Integration with your own systems
    
    Example: {'project': 'customer-support', 'version': '1.2', 'team': 'ai-ops'}
    """

    temperature: Optional[float] = None
    """Controls randomness in the agent's responses (0.0 to 1.0).
    
    - 0.0: Deterministic, consistent responses
    - 0.3: Focused and coherent (good for factual tasks)  
    - 0.7: Balanced creativity and coherence (default-like)
    - 1.0: Maximum creativity and randomness
    
    Lower values for factual tasks, higher values for creative tasks.
    """

    top_p: Optional[float] = None
    """Controls diversity via nucleus sampling (0.0 to 1.0).
    
    Alternative to temperature for controlling randomness. Only consider tokens
    in the top p probability mass. Lower values make responses more focused.
    
    - 0.1: Very focused, only most likely tokens
    - 0.9: More diverse token selection
    
    Don't use both temperature and top_p simultaneously.
    """

    response_format: Optional[Dict[str, Any]] = None
    """Specifies the format that the model must output.
    
    Use this to enforce structured outputs like JSON. The agent will format
    its responses according to this specification.
    
    Example: {'type': 'json_object'} for JSON responses
    Example: {'type': 'json_schema', 'json_schema': {...}} for specific schemas
    """

    client_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Additional arguments passed to the Azure AI Projects client constructor.
    
    Use this for advanced client configuration like custom timeouts, retry policies,
    or proxy settings. Most users won't need to modify this.
    """

    # Private attributes
    _client: Optional[AIProjectClient] = PrivateAttr(default=None)
    _agent: Optional[Agent] = PrivateAttr(default=None)

    class Config:
        """Configuration for this pydantic object."""

        extra = "forbid"

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = cls.model_fields.keys()
        extra = values.get("model_kwargs", {})
        values = {k: v for k, v in values.items() if k not in extra}
        extra = {k: v for k, v in extra.items() if k not in all_required_field_names}
        values["client_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> "AzureAIAgentsService":
        """Validate that endpoint is provided."""
        if not self.endpoint:
            raise ValueError("'endpoint' must be provided")
        return self

    def _create_client(self) -> AIProjectClient:
        """Create and configure the Azure AI Projects client.

        This method handles client creation with proper authentication and
        configuration for direct endpoint access.

        The client is cached after creation to avoid unnecessary recreations.

        Returns:
            AIProjectClient: Configured client instance ready for agent operations.

        Raises:
            ValueError: If endpoint is not provided.
        """
        if self._client is not None:
            return self._client

        credential = self.credential
        if credential is None:
            credential = DefaultAzureCredential()

        # Create client with endpoint
        self._client = AIProjectClient(
            endpoint=self.endpoint,
            credential=credential,
            **self.client_kwargs,  # type: ignore[arg-type]
        )

        return self._client

    def _get_or_create_agent(self) -> Agent:
        """Get the cached agent or create a new one if none exists.

        This method implements a lazy loading pattern for agent creation. It first
        checks if an agent is already cached and returns it if available. If no
        agent exists, it creates a new one using the instance configuration and
        caches it for future use. This is used for the generate and invoke methods
        in LangChain.

        The agent is created with all the configuration parameters specified during
        service initialization, including model, instructions, tools, and optional
        parameters like temperature and top_p.

        Returns:
            Agent: The cached or newly created agent instance.
        """
        if self._agent is not None:
            return self._agent

        client = self._create_client()

        # Build agent creation parameters with proper typing
        agent_params: Dict[str, Any] = {
            "model": self.model,
            "name": self.agent_name,
            "instructions": self.instructions,
        }

        # Add optional parameters
        if self.agent_description:
            agent_params["description"] = self.agent_description
        if self.tools:
            agent_params["tools"] = self.tools
        if self.tool_resources:
            agent_params["tool_resources"] = self.tool_resources
        if self.metadata:
            agent_params["metadata"] = self.metadata
        if self.temperature is not None:
            agent_params["temperature"] = self.temperature
        if self.top_p is not None:
            agent_params["top_p"] = self.top_p
        if self.response_format is not None:
            agent_params["response_format"] = self.response_format

        self._agent = client.agents.create_agent(**agent_params)
        logger.info(f"Created agent with ID: {self._agent.id}")
        return self._agent

    async def _aget_or_create_agent(self) -> Agent:
        """Asynchronously get the cached agent or create a new one.

        This is the async version of _get_or_create_agent(). Since the Azure AI
        Projects SDK doesn't have native async support for agent creation, this
        method wraps the synchronous operation using asyncio.to_thread() to avoid
        blocking the event loop.

        Returns:
            Agent: The cached or newly created agent instance.
        """
        if self._agent is not None:
            return self._agent

        # Use sync client wrapped in asyncio.to_thread to avoid async client issues
        import asyncio

        def _sync_create_agent() -> Agent:
            return self._get_or_create_agent()

        self._agent = await asyncio.to_thread(_sync_create_agent)
        return self._agent

    @property
    def _llm_type(self) -> str:
        """Return the LLM type identifier for LangChain compatibility.

        This property is used by LangChain's internal systems for logging,
        monitoring, and type identification. It helps distinguish this
        Azure AI Agents implementation from other LLM providers.

        Returns:
            str: The string "azure_ai_agents" identifying this LLM type.
        """
        return "azure_ai_agents"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for multiple prompts using the Azure AI Agent.

        This is the core LangChain LLM interface method that processes a batch of
        prompts and returns structured results. Each prompt is processed independently
        in its own conversation thread to ensure isolation.

        The method handles the complete conversation flow:
        1. Creates a dedicated thread for each prompt
        2. Sends the user message to the agent
        3. Executes the agent run and waits for completion
        4. Extracts the response from the agent's messages
        5. Cleans up the thread to prevent resource leaks

        Args:
            prompts: List of input text prompts to process.
            stop: Optional list of stop sequences (currently not implemented).
            run_manager: Optional callback manager for tracing and monitoring.
            **kwargs: Additional arguments (passed through but not used).

        Returns:
            LLMResult: Structured result containing Generation objects for each prompt.
            Each Generation contains the agent's text response.
        """
        generations = []
        for prompt in prompts:
            generation = self._generate_single(prompt, stop, run_manager, **kwargs)
            generations.append([generation])

        return LLMResult(generations=generations)

    def _generate_single(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Generation:
        """Generate a single response for one prompt using a dedicated thread.

        This method handles the complete conversation lifecycle for a single prompt:

        1. **Thread Creation**: Creates a new conversation thread for isolation
        2. **Message Sending**: Sends the user's prompt as a message
        3. **Agent Execution**: Runs the agent and waits for completion
        4. **Response Extraction**: Gets the agent's response from the message history
        5. **Cleanup**: Deletes the thread to free resources

        Each conversation is completely isolated, ensuring no cross-talk between
        different invocations and maintaining conversation context boundaries.

        Args:
            prompt: The user's input text to send to the agent.
            stop: Optional stop sequences (not currently implemented).
            run_manager: Optional callback manager for monitoring and tracing.
            **kwargs: Additional arguments (currently unused).

        Returns:
            Generation: Contains the agent's text response and metadata.

        Raises:
            Exception: Re-raises any errors that occur during the conversation flow.
            Logs detailed error information for debugging.
        """
        try:
            client = self._create_client()
            agent = self._get_or_create_agent()

            # Create a thread for this conversation
            thread = client.agents.threads.create()

            # Add a message to the thread
            client.agents.messages.create(
                thread_id=thread.id, role="user", content=prompt
            )

            # Create and process an agent run
            client.agents.runs.create_and_process(
                thread_id=thread.id, agent_id=agent.id
            )

            # Get the response messages
            messages = client.agents.messages.list(thread_id=thread.id)

            # Find the latest assistant message
            response_text = ""
            for msg in messages:
                if msg.role == "assistant":
                    # msg.content is a list of content objects, extract text from
                    # text objects
                    if hasattr(msg, "content") and msg.content:
                        for content_item in msg.content:
                            if (
                                hasattr(content_item, "type")
                                and content_item.type == "text"
                            ):
                                if hasattr(content_item, "text") and hasattr(
                                    content_item.text, "value"
                                ):
                                    response_text = content_item.text.value
                                elif hasattr(content_item, "text"):
                                    response_text = content_item.text
                                break
                    break
            # Clean up - delete the thread
            try:
                client.agents.threads.delete(thread.id)
            except Exception as e:
                logger.warning(f"Failed to clean up thread {thread.id}: {e}")

            return Generation(text=response_text)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Asynchronously generate responses for multiple prompts.

        This is the async version of _generate() that processes multiple prompts
        concurrently while maintaining the same conversation isolation guarantees.
        Since the Azure AI Projects SDK doesn't have native async support, this
        method uses asyncio.to_thread() for each prompt to avoid blocking.

        Args:
            prompts: List of input text prompts to process.
            stop: Optional list of stop sequences (not implemented).
            run_manager: Optional async callback manager for monitoring.
            **kwargs: Additional arguments (passed through).

        Returns:
            LLMResult: Structured result with Generation objects for each prompt.
        """
        generations = []
        for prompt in prompts:
            generation = await self._agenerate_single(
                prompt, stop, run_manager, **kwargs
            )
            generations.append([generation])

        return LLMResult(generations=generations)

    async def _agenerate_single(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Generation:
        """Asynchronously generate a single response using thread pool execution.

        This method wraps the synchronous conversation flow in asyncio.to_thread()
        to provide async compatibility without blocking the event loop. The actual
        conversation logic remains the same as _generate_single().

        Args:
            prompt: The user's input text to send to the agent.
            stop: Optional stop sequences (not implemented).
            run_manager: Optional async callback manager.
            **kwargs: Additional arguments.

        Returns:
            Generation: Contains the agent's text response.
        """
        try:
            # For now, use the sync client wrapped in asyncio.to_thread
            # This avoids the AsyncItemPaged issues
            import asyncio

            def _sync_generate() -> str:
                client = self._create_client()
                agent = self._get_or_create_agent()

                # Create a thread for this conversation
                thread = client.agents.threads.create()

                # Create a message
                client.agents.messages.create(
                    thread_id=thread.id, role="user", content=prompt
                )

                # Run the agent
                client.agents.runs.create_and_process(
                    thread_id=thread.id, agent_id=agent.id
                )

                # Get the response messages
                messages = client.agents.messages.list(thread_id=thread.id)

                # Find the latest assistant message
                response_text = ""
                for msg in messages:
                    if msg.role == "assistant":
                        # msg.content is a list of content objects, extract text
                        # from text objects
                        if hasattr(msg, "content") and msg.content:
                            for content_item in msg.content:
                                if (
                                    hasattr(content_item, "type")
                                    and content_item.type == "text"
                                ):
                                    if hasattr(content_item, "text") and hasattr(
                                        content_item.text, "value"
                                    ):
                                        response_text = content_item.text.value
                                    elif hasattr(content_item, "text"):
                                        response_text = content_item.text
                                    break
                        break

                # Clean up - delete the thread
                try:
                    client.agents.threads.delete(thread.id)
                except Exception as e:
                    logger.warning(f"Failed to clean up thread {thread.id}: {e}")

                return response_text

            # Run the sync operation in a thread pool
            response_text = await asyncio.to_thread(_sync_generate)
            return Generation(text=response_text)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def create_agent(self, **kwargs: Any) -> Agent:
        """Create a new agent with custom parameters, replacing cached agent.

        This method allows you to create additional agents or modify the current
        agent's configuration beyond what was set during service initialization.
        Unlike _get_or_create_agent(), this method always creates a new agent and
        replaces the cached instance.

        The method starts with the base configuration from the service instance
        and then applies any overrides provided through kwargs. This is useful
        for scenarios where you need different agent configurations during runtime.

        Args:
            **kwargs: Parameters to override or add to the agent creation.
                     Common parameters include:
                     - model: Change the model name
                     - name: Give the agent a different name
                     - instructions: Modify the agent's behavior instructions
                     - temperature: Adjust response randomness
                     - tools: Add or change available tools
                     - tool_resources: Modify tool resources

        Returns:
            Agent: The newly created agent instance (also cached internally).

        Example:
            # Create an agent with different instructions
            agent = service.create_agent(
                name="data-analyst-v2",
                instructions="You are a specialized data analyst focusing on "
                "financial data.",
                temperature=0.3
            )

            # Create an agent with tools
            agent = service.create_agent(
                tools=[{"type": "code_interpreter"}],
                instructions="You can analyze data and create visualizations."
            )
        """
        client = self._create_client()
        # Build agent creation parameters with proper typing
        agent_params: Dict[str, Any] = {
            "model": self.model,
            "name": self.agent_name,
            "instructions": self.instructions,
        }

        # Add optional parameters
        if self.agent_description:
            agent_params["description"] = self.agent_description
        if self.tools:
            agent_params["tools"] = self.tools
        if self.tool_resources:
            agent_params["tool_resources"] = self.tool_resources
        if self.metadata:
            agent_params["metadata"] = self.metadata
        if self.temperature is not None:
            agent_params["temperature"] = self.temperature
        if self.top_p is not None:
            agent_params["top_p"] = self.top_p
        if self.response_format is not None:
            agent_params["response_format"] = self.response_format

        # Override with any additional kwargs
        agent_params.update(kwargs)

        self._agent = client.agents.create_agent(**agent_params)
        logger.info(f"Created agent with ID: {self._agent.id}")
        return self._agent

    async def acreate_agent(self, **kwargs: Any) -> Agent:
        """Asynchronously create a new agent with custom parameters.

        This is the async version of create_agent() that uses thread pool execution
        to avoid blocking the event loop during agent creation.

        Args:
            **kwargs: Parameters to override or add to the agent creation.

        Returns:
            Agent: The newly created agent instance.
        """
        import asyncio

        def _sync_create() -> Agent:
            return self.create_agent(**kwargs)

        return await asyncio.to_thread(_sync_create)

    def get_client(self) -> AIProjectClient:
        """Get the underlying Azure AI Projects client for advanced operations.

        This method provides direct access to the Azure AI Projects client,
        allowing you to perform operations that aren't exposed through the
        LangChain interface. This is useful for advanced scenarios like:

        - Managing files and file uploads
        - Creating and managing conversation threads manually
        - Accessing agent run details and metadata
        - Performing bulk operations
        - Using features not yet wrapped by this service

        Returns:
            AIProjectClient: The configured client instance.

        Example:
            # Upload a file for use with tools
            client = agent_service.get_client()
            uploaded_file = client.files.upload_and_poll(
                file_path="data.csv",
                purpose=FilePurpose.AGENTS
            )

            # Create a thread manually for multi-turn conversations
            thread = client.threads.create()

        Warning:
            When using the client directly, you're responsible for proper
            resource management (e.g., cleaning up threads, deleting files).
        """
        return self._create_client()

    def get_async_client(self) -> AIProjectClient:
        """Get the underlying Azure AI Projects client (same instance as sync version).

        Note: The Azure AI Projects SDK doesn't have separate async clients,
        so this returns the same client as get_client(). For async operations,
        use the async methods of this service which handle thread pool execution.

        Returns:
            AIProjectClient: The configured client instance.
        """
        return self._create_client()

    def get_agent(self) -> Optional[Agent]:
        """Get the current cached agent instance without creating a new one.

        This method returns the agent that was created during the first
        generation call or through create_agent(). It does not trigger
        agent creation if none exists yet.

        Returns:
            Optional[Agent]: The cached agent instance if it exists, None otherwise.

        Example:
            # Check if an agent has been created
            agent = service.get_agent()
            if agent:
                print(f"Agent ID: {agent.id}, Name: {agent.name}")
            else:
                print("No agent created yet")
        """
        return self._agent

    def delete_agent(self, agent_id: Optional[str] = None) -> None:
        """Delete an agent from the Azure AI service.

        This method permanently removes an agent from your Azure AI project.
        By default, it deletes the current cached agent, but you can specify
        a different agent ID to delete any agent you have access to.

        Args:
            agent_id: The ID of the agent to delete. If None, deletes the
                     current cached agent. If no cached agent exists and
                     no ID is provided, raises ValueError.

        Raises:
            ValueError: If no agent_id is provided and no cached agent exists.

        Example:
            # Delete the current agent
            service.delete_agent()

            # Delete a specific agent by ID
            service.delete_agent("agent_abc123")

        Warning:
            Agent deletion is permanent and cannot be undone. Any conversation
            threads using this agent will no longer work. Make sure you don't
            need the agent before deleting it.
        """
        client = self._create_client()

        if agent_id is None:
            if self._agent is None:
                raise ValueError(
                    "No agent to delete. Create an agent first or provide agent_id."
                )
            agent_id = self._agent.id
            self._agent = None  # Clear the cached agent

        client.agents.delete_agent(agent_id)
        logger.info(f"Deleted agent with ID: {agent_id}")

    async def adelete_agent(self, agent_id: Optional[str] = None) -> None:
        """Asynchronously delete an agent from the Azure AI service.

        This is the async version of delete_agent() that uses thread pool
        execution to avoid blocking the event loop.

        Args:
            agent_id: The ID of the agent to delete. If None, deletes the
                     current cached agent.

        Raises:
            ValueError: If no agent_id is provided and no cached agent exists.
        """
        import asyncio

        def _sync_delete() -> None:
            return self.delete_agent(agent_id)

        await asyncio.to_thread(_sync_delete)

    def close(self) -> None:
        """Close all underlying client connections and free resources.

        This method properly closes the HTTP connections used by the Azure AI
        Projects client and any associated credentials. Call this method when
        you're done using the service to ensure proper resource cleanup.

        The method handles:
        - Closing the Azure AI Projects client's HTTP session
        - Closing credential providers that support it (like DefaultAzureCredential)
        - Graceful handling if resources are already closed

        Example:
            try:
                # Use the service
                response = service.invoke("Hello!")
            finally:
                # Always clean up
                service.close()

        Note:
            After calling close(), the service should not be used for new
            operations. Create a new instance if you need to continue working.
        """
        if hasattr(self, "_client") and self._client:
            self._client.agents.close()

        # Close the credential if it has a close method and is not None
        if self.credential is not None and hasattr(self.credential, "close"):
            self.credential.close()

    async def aclose(self) -> None:
        """Asynchronously close all underlying client connections and free resources.

        This is the async version of close() that uses thread pool execution
        to avoid blocking the event loop during cleanup operations.

        Example:
            try:
                # Use the service
                response = await service.ainvoke("Hello!")
            finally:
                # Always clean up
                await service.aclose()
        """
        import asyncio

        def _sync_close() -> None:
            return self.close()

        await asyncio.to_thread(_sync_close)

    def __del__(self) -> None:
        """Cleanup when the object is being garbage collected.

        Note: This destructor intentionally does NOT auto-delete agents from
        the Azure AI service. Agents are persistent resources that may be
        referenced by other code or used across multiple service instances.

        Users should explicitly call delete_agent() if they want to remove
        agents from the Azure AI service. This prevents accidental deletion
        of agents that might still be needed.

        For automatic resource cleanup of connections (not agents), use the
        close() method or a context manager pattern.
        """
