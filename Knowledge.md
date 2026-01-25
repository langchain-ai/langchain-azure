# Knowledge Base: langchain-azure Repository

## Repository Overview

This is the **langchain-azure** repository - an official LangChain integration package providing first-class support for Azure AI Foundry capabilities in the LangChain and LangGraph ecosystem.

### Repository Structure

```
langchain-azure/
├── libs/
│   ├── azure-ai/              # Core Azure AI Foundry integration package
│   ├── azure-dynamic-sessions/ # Azure Dynamic Sessions integration
│   ├── azure-postgresql/      # Azure PostgreSQL vector store
│   ├── azure-storage/         # Azure Blob Storage document loaders
│   └── sqlserver/            # SQL Server vector store
├── samples/
│   ├── react-agent-docintelligence/  # Sample ReAct agent with Document Intelligence
│   ├── multi-agent-travel-planner/   # Multi-agent sample
│   └── rag-storage-document-loaders/ # RAG with Azure Storage
└── README.md
```

## Core Package: langchain-azure-ai (v1.0.4)

### Key Features

1. **Azure AI Agent Service Integration**
   - `AgentServiceFactory` for creating prompt-based agents
   - Support for declarative agents created in Azure AI Foundry
   - LangGraph integration for complex agent workflows
   - Tool calling and structured outputs

2. **Azure AI Foundry Models**
   - `AzureAIChatCompletionsModel` - Chat completions using Azure AI Inference API
   - `AzureAIEmbeddingsModel` - Embeddings generation
   - Support for Azure OpenAI and GitHub Models endpoints
   - Support for models like GPT-4o, DeepSeek-R1, etc.

3. **Vector Stores**
   - Azure AI Search (formerly Cognitive Search)
   - Azure Cosmos DB (NoSQL and MongoDB vCore)
   - Semantic caching capabilities

4. **Azure AI Services Tools**
   - `AzureAIDocumentIntelligenceTool` - Document parsing and analysis
   - `AzureAITextAnalyticsHealthTool` - Health text analytics
   - `AzureAIImageAnalysisTool` - Image analysis
   - `AIServicesToolkit` - Unified access to all tools

5. **Observability**
   - `AzureAIOpenTelemetryTracer` - OpenTelemetry tracing integration
   - Azure Application Insights support
   - Semantic conventions for GenAI

6. **Additional Features**
   - Chat message histories with Cosmos DB
   - Azure Logic Apps integration
   - Retrievers for Azure AI Search

### Dependencies

**Core Dependencies:**
- `langchain>=1.0.0,<2.0.0`
- `langchain-openai>=1.0.0,<2.0.0`
- `azure-ai-agents==1.2.0b5`
- `azure-ai-inference[opentelemetry]>=1.0.0b9,<2.0`
- `azure-ai-projects~=1.0`
- `azure-identity~=1.15`
- `azure-search-documents~=11.4`
- `azure-cosmos>=4.14.0b1,<5.0`

**Optional Dependencies:**
- `[opentelemetry]` - For tracing capabilities
- `[tools]` - For Azure AI Services tools

## Integration with LangChain/LangGraph/LangSmith

### LangChain Integration
- Full compatibility with LangChain 1.0+
- Implements standard LangChain interfaces (`BaseChatModel`, `BaseEmbeddings`, `VectorStore`, etc.)
- Works seamlessly with LangChain chains, agents, and tools

### LangGraph Integration
- Native support for building agent workflows
- `AgentServiceFactory.create_prompt_agent()` returns a `CompiledStateGraph`
- Can be deployed using LangGraph CLI (`langgraph dev`, `langgraph deploy`)
- Support for `langgraph.json` configuration files
- Integration with LangGraph Cloud/Platform

### LangSmith Integration
- Compatible with LangSmith tracing via OpenTelemetry
- `AzureAIOpenTelemetryTracer` can export traces to LangSmith
- Environment variables for LangSmith configuration work as expected

## Azure AI Foundry Agent Service

### What is Azure AI Agent Service?
Azure AI Agent Service is a managed service in Azure AI Foundry that provides:
- Declarative agent creation through the portal
- Managed agent execution and lifecycle
- Built-in conversation threading
- Tool integration and execution
- State management

### Integration Patterns

**Pattern 1: Prompt-Based Agents (Recommended)**
```python
from langchain_azure_ai.agents import AgentServiceFactory

factory = AgentServiceFactory(
    project_endpoint="https://<resource>.services.ai.azure.com/api/projects/<project>",
    credential=DefaultAzureCredential()
)

agent = factory.create_prompt_agent(
    name="my-agent",
    model="gpt-4.1",
    instructions="You are a helpful assistant...",
    tools=[tool1, tool2],
    trace=True
)

# Returns a CompiledStateGraph (LangGraph)
result = agent.invoke({"messages": [HumanMessage(content="Hello")]})
```

**Pattern 2: Declarative Agents from Azure Portal**
- Create agents through Azure AI Foundry portal
- Import and use them in LangGraph workflows
- Combines portal-managed agents with custom LangGraph logic

## Deployment Options

### Option 1: Local Development with LangGraph CLI
```bash
langgraph dev
```
- Runs agent locally with hot reload
- Interactive Studio UI
- Best for development and testing

### Option 2: Azure Container Apps (via LangGraph Platform)
- Deploy LangGraph applications to Azure
- Managed hosting with autoscaling
- Built-in observability

### Option 3: Azure AI Foundry Deployment
- Deploy agents through Azure AI Foundry portal
- Managed endpoint with API keys
- Built-in monitoring and logging

### Option 4: Custom Azure Deployment
- Deploy as Azure Container Instance
- Deploy as Azure App Service
- Deploy as Azure Kubernetes Service (AKS)

## Authentication

The package supports multiple authentication methods:
1. **DefaultAzureCredential** (Recommended for production)
   - Uses Azure Managed Identity, Azure CLI, Visual Studio, etc.
   - No secrets in code
2. **API Key Authentication**
   - Using `credential="your-api-key"`
   - Simpler for development

## Environment Variables

Key environment variables used:
- `AZURE_AI_PROJECT_ENDPOINT` - Azure AI Foundry project endpoint
- `AZURE_OPENAI_API_KEY` - API key for Azure OpenAI
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint
- `APPLICATIONINSIGHTS_CONNECTION_STRING` - For telemetry
- `LANGCHAIN_API_KEY` - For LangSmith integration
- `LANGCHAIN_TRACING_V2=true` - Enable LangSmith tracing

## Samples Included

### 1. react-agent-docintelligence
- **Purpose**: ReAct agent with Azure AI Document Intelligence tool
- **Stack**: LangGraph + Azure AI Agent Service + Document Intelligence
- **Key Files**: `src/react_agent/graph.py`
- **Deployment**: LangGraph CLI ready

### 2. multi-agent-travel-planner
- **Purpose**: Multi-agent system with nested agents
- **Pattern**: Supervisor pattern with specialized sub-agents
- **Features**: OpenTelemetry tracing, complex workflows

### 3. rag-storage-document-loaders
- **Purpose**: RAG pipeline with Azure Blob Storage
- **Stack**: Azure AI Search + Azure Storage + Embeddings
- **Pattern**: Traditional RAG (embed + query)

## Known Issues and Considerations

1. **Agent Service is in Preview**: `azure-ai-agents==1.2.0b5` is beta
2. **Breaking Changes in v1.0**: Migration from 0.1.x requires parameter changes
3. **Tracing Setup**: Requires OpenTelemetry extras for full functionality
4. **Model Availability**: Ensure models are deployed in your Azure region

## Related Azure Services Required

To use this repository effectively, you need:
1. **Azure AI Foundry Hub and Project** (required)
2. **Azure OpenAI Service** (for LLM models)
3. **Azure AI Document Intelligence** (optional, for document tools)
4. **Azure AI Search** (optional, for vector stores)
5. **Azure Cosmos DB** (optional, for vector stores/chat history)
6. **Azure Storage Account** (optional, for document loaders)
7. **Azure Application Insights** (optional, for monitoring)

---

## Updates Log

### 2026-01-24 - Initial Knowledge Base Creation
- Analyzed repository structure and capabilities
- Documented integration points with LangChain/LangGraph/LangSmith
- Identified deployment patterns for Azure AI Foundry
- Created baseline understanding for agent development

---

**Last Updated**: 2026-01-24  
**Repository Version**: langchain-azure-ai v1.0.4  
**LangChain Version**: 1.0.2  
**LangGraph CLI**: 0.4.4+
