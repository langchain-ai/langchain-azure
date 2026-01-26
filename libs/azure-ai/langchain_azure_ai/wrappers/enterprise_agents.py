"""Enterprise Agent wrappers for Azure AI Foundry integration.

This module provides wrappers for enterprise-grade agents:
- Research Agent: Web research and analysis
- Content Agent: Content generation and editing
- Data Analyst Agent: Data analysis and visualization
- Document Agent: Document processing and analysis
- Code Assistant Agent: Code generation and review
- RAG Agent: Retrieval-Augmented Generation
- Document Intelligence Agent: Azure AI Document Intelligence

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


class EnterpriseAgentWrapper(FoundryAgentWrapper):
    """Wrapper for enterprise agents with Azure AI Foundry integration.

    This wrapper is designed for enterprise use cases:
    - Research: Web research, document analysis, fact-checking
    - Content: Content generation, editing, summarization
    - Data Analysis: Data insights, visualization, reporting
    - Document Processing: Document parsing, extraction, classification
    - Code Assistance: Code generation, review, debugging
    - RAG: Retrieval-augmented generation with vector stores
    - Document Intelligence: Azure AI Document Intelligence integration

    Example:
        ```python
        from langchain_azure_ai.wrappers import EnterpriseAgentWrapper

        # Create Research agent
        research_agent = EnterpriseAgentWrapper(
            name="research-agent",
            agent_subtype="research",
            tools=[web_search, document_loader, summarizer],
        )

        response = research_agent.chat("Research the latest AI trends")
        ```
    """

    # Default instructions for different enterprise agent subtypes
    DEFAULT_INSTRUCTIONS = {
        "research": """You are an Enterprise Research Agent. Your role is to:
1. Conduct comprehensive research on topics requested by users
2. Search the web and internal knowledge bases
3. Synthesize information from multiple sources
4. Provide well-structured, cited research summaries
5. Identify key insights and trends

Always cite your sources and indicate confidence levels in your findings.""",
        "content": """You are an Enterprise Content Agent. Your role is to:
1. Generate high-quality content for various purposes
2. Edit and improve existing content
3. Maintain brand voice and style guidelines
4. Adapt content for different audiences and formats
5. Ensure accuracy and readability

Focus on clarity, engagement, and professional quality.""",
        "data_analyst": """You are an Enterprise Data Analyst Agent. Your role is to:
1. Analyze data and identify patterns and insights
2. Create visualizations and charts
3. Generate statistical summaries and reports
4. Answer data-related questions with evidence
5. Recommend data-driven actions

Use analytical tools and explain your methodology clearly.""",
        "document": """You are an Enterprise Document Agent. Your role is to:
1. Process and analyze documents of various formats
2. Extract key information and metadata
3. Classify and organize documents
4. Summarize document contents
5. Answer questions about documents

Handle documents professionally and maintain confidentiality.""",
        "code_assistant": """You are an Enterprise Code Assistant Agent. Your role is to:
1. Generate code in various programming languages
2. Review code for bugs, security issues, and best practices
3. Explain code functionality and logic
4. Refactor and optimize code
5. Help debug and troubleshoot issues

Write clean, well-documented, and maintainable code.""",
        "rag": """You are a Retrieval-Augmented Generation Agent. Your role is to:
1. Answer questions using retrieved context
2. Search vector stores for relevant information
3. Synthesize answers from multiple sources
4. Indicate when information is not available
5. Provide citations for retrieved content

Ground your answers in retrieved context and be transparent about sources.""",
        "document_intelligence": """You are an Azure Document Intelligence Agent. Your role is to:
1. Analyze documents using Azure AI Document Intelligence
2. Extract text, tables, and key-value pairs
3. Process invoices, receipts, and forms
4. Handle various document formats (PDF, images, etc.)
5. Structure extracted data for downstream processing

Use Azure Document Intelligence tools for accurate extraction.""",
    }

    def __init__(
        self,
        name: str,
        instructions: Optional[str] = None,
        agent_subtype: str = "research",
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        config: Optional[WrapperConfig] = None,
        existing_agent: Optional[Union[CompiledStateGraph, Any]] = None,
        enable_memory: bool = True,
        enable_streaming: bool = True,
        **kwargs: Any,
    ):
        """Initialize enterprise agent wrapper.

        Args:
            name: Name of the agent.
            instructions: System instructions. If None, uses default for subtype.
            agent_subtype: Type of enterprise agent.
            tools: List of tools for the agent.
            model: Model deployment name.
            temperature: Temperature for responses.
            config: Wrapper configuration.
            existing_agent: Existing agent to wrap.
            enable_memory: Whether to enable conversation memory.
            enable_streaming: Whether to enable streaming responses.
            **kwargs: Additional arguments.
        """
        self.agent_subtype = agent_subtype.lower()
        self.enable_memory = enable_memory
        self.enable_streaming = enable_streaming
        self._memory = MemorySaver() if enable_memory else None

        # Use default instructions if not provided
        if instructions is None:
            instructions = self.DEFAULT_INSTRUCTIONS.get(
                self.agent_subtype,
                self.DEFAULT_INSTRUCTIONS["research"],
            )

        super().__init__(
            name=name,
            instructions=instructions,
            agent_type=AgentType.ENTERPRISE_AGENT,
            description=f"Enterprise Agent ({agent_subtype}): {name}",
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
        """Create enterprise agent using LangGraph's create_react_agent.

        Enterprise agents use the ReAct pattern with:
        - Specialized system prompts
        - Tools for enterprise operations
        - Memory for conversation context

        Args:
            llm: The language model to use.
            tools: List of tools for the agent.

        Returns:
            A compiled LangGraph StateGraph.
        """
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SystemMessage(content=self.instructions),
            checkpointer=self._memory,
        )

        return agent

    def analyze(
        self,
        query: str,
        context: Optional[str] = None,
        output_format: str = "text",
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform analysis using the enterprise agent.

        Args:
            query: The analysis query or question.
            context: Optional context for the analysis.
            output_format: Desired output format (text, json, markdown).
            thread_id: Optional thread ID for conversation continuity.

        Returns:
            Analysis results dictionary.
        """
        # Build the full query with context and format instructions
        full_query = query
        if context:
            full_query = f"Context:\n{context}\n\nQuery:\n{query}"
        if output_format != "text":
            full_query += f"\n\nProvide output in {output_format} format."

        response = self.chat(full_query, thread_id=thread_id)

        return {
            "agent_type": self.agent_subtype,
            "query": query,
            "output_format": output_format,
            "response": response,
            "thread_id": thread_id,
        }


class ResearchAgentWrapper(EnterpriseAgentWrapper):
    """Specialized wrapper for Research agents."""

    def __init__(
        self,
        name: str = "research-agent",
        instructions: Optional[str] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="research",
            tools=tools,
            **kwargs,
        )


class ContentAgentWrapper(EnterpriseAgentWrapper):
    """Specialized wrapper for Content generation agents."""

    def __init__(
        self,
        name: str = "content-agent",
        instructions: Optional[str] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="content",
            tools=tools,
            **kwargs,
        )


class DataAnalystWrapper(EnterpriseAgentWrapper):
    """Specialized wrapper for Data Analyst agents."""

    def __init__(
        self,
        name: str = "data-analyst-agent",
        instructions: Optional[str] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="data_analyst",
            tools=tools,
            **kwargs,
        )


class DocumentAgentWrapper(EnterpriseAgentWrapper):
    """Specialized wrapper for Document processing agents."""

    def __init__(
        self,
        name: str = "document-agent",
        instructions: Optional[str] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="document",
            tools=tools,
            **kwargs,
        )


class CodeAssistantWrapper(EnterpriseAgentWrapper):
    """Specialized wrapper for Code Assistant agents."""

    def __init__(
        self,
        name: str = "code-assistant-agent",
        instructions: Optional[str] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        temperature: float = 0.0,  # Lower temperature for code
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="code_assistant",
            tools=tools,
            temperature=temperature,
            **kwargs,
        )


class RAGAgentWrapper(EnterpriseAgentWrapper):
    """Specialized wrapper for RAG (Retrieval-Augmented Generation) agents."""

    def __init__(
        self,
        name: str = "rag-agent",
        instructions: Optional[str] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        retriever: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.retriever = retriever
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="rag",
            tools=tools,
            **kwargs,
        )

    def query_with_retrieval(
        self,
        query: str,
        top_k: int = 5,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query with retrieval from vector store.

        Args:
            query: The query string.
            top_k: Number of documents to retrieve.
            thread_id: Optional thread ID for conversation continuity.

        Returns:
            Response with retrieved context.
        """
        retrieved_docs = []
        if self.retriever:
            retrieved_docs = self.retriever.invoke(query)[:top_k]
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            full_query = f"Context:\n{context}\n\nQuestion:\n{query}"
        else:
            full_query = query

        response = self.chat(full_query, thread_id=thread_id)

        return {
            "query": query,
            "response": response,
            "sources": [doc.metadata for doc in retrieved_docs] if retrieved_docs else [],
            "thread_id": thread_id,
        }


class DocumentIntelligenceWrapper(EnterpriseAgentWrapper):
    """Specialized wrapper for Azure Document Intelligence agents."""

    def __init__(
        self,
        name: str = "document-intelligence-agent",
        instructions: Optional[str] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="document_intelligence",
            tools=tools,
            **kwargs,
        )
