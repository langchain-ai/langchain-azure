# Agents Guide: Building and Deploying to Azure AI Foundry

## Purpose
This document provides a practical guide for AI agents (and developers) working with this repository to build, test, and deploy LangChain agents to Azure AI Foundry.

## Quick Start Checklist

### Prerequisites
- [ ] Azure AI Foundry Hub and Project created
- [ ] Azure OpenAI model deployed (e.g., gpt-4o, gpt-4.1)
- [ ] Python 3.10+ installed
- [ ] LangChain, LangGraph, LangSmith accounts/setup (if using)
- [ ] Azure CLI installed and authenticated (`az login`)

### Development Setup
```bash
# 1. Navigate to a sample or create new project
cd samples/react-agent-docintelligence

# 2. Install dependencies (using poetry)
poetry install

# Or using pip
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your Azure endpoints

# 4. Run locally
poetry run langgraph dev
```

## Architecture Patterns

### Pattern 1: Simple Prompt-Based Agent (Recommended Starting Point)

**Use Case**: Single-agent applications with tools

**Example**: `samples/react-agent-docintelligence`

```python
from langchain_azure_ai.agents import AgentServiceFactory
from langchain_azure_ai.tools import AzureAIDocumentIntelligenceTool

# Create factory (reads AZURE_AI_PROJECT_ENDPOINT from env)
factory = AgentServiceFactory()

# Create agent with tools
agent = factory.create_prompt_agent(
    name="my-agent",
    model="gpt-4.1",
    instructions="""You are a helpful assistant that can analyze documents.
    
    Current time: {system_time}
    """,
    tools=[AzureAIDocumentIntelligenceTool()],
    trace=True  # Enable OpenTelemetry tracing
)

# Invoke (returns LangGraph CompiledStateGraph)
result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this document..."}]
})
```

**When to Use**:
- Single-purpose agents
- Straightforward tool usage
- Quick prototyping

### Pattern 2: Multi-Agent System with LangGraph

**Use Case**: Complex workflows with specialized sub-agents

**Example**: `samples/multi-agent-travel-planner`

```python
from langchain.agents import create_agent
from langgraph.graph import StateGraph

# Create multiple specialized agents
research_agent = create_agent(...)
booking_agent = create_agent(...)
supervisor_agent = create_agent(...)

# Build workflow
workflow = StateGraph(state_schema)
workflow.add_node("research", research_agent)
workflow.add_node("booking", booking_agent)
workflow.add_node("supervisor", supervisor_agent)

# Add conditional edges
workflow.add_conditional_edges("supervisor", route_to_agent)

# Compile
graph = workflow.compile()
```

**When to Use**:
- Complex multi-step workflows
- Need for specialized agents
- Dynamic routing between agents

### Pattern 3: RAG (Retrieval-Augmented Generation)

**Use Case**: Question-answering over documents

**Example**: `samples/rag-storage-document-loaders`

```python
# Step 1: Embed documents
from langchain_azure_storage import AzureBlobStorageLoader
from langchain_azure_ai.vectorstores import AzureSearchVectorStore

loader = AzureBlobStorageLoader(...)
docs = loader.load()
vectorstore = AzureSearchVectorStore.from_documents(docs, embeddings)

# Step 2: Create retrieval chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=vectorstore.as_retriever()
)
```

**When to Use**:
- Document Q&A
- Knowledge base search
- Content-based retrieval

## Development Workflow

### 1. Local Development
```bash
# Start LangGraph dev server
langgraph dev

# Opens Studio UI at http://localhost:8123
# Test your agent interactively
```

**Benefits**:
- Hot reload on code changes
- Interactive debugging
- Visual graph inspection

### 2. Testing
```bash
# Unit tests
pytest tests/unit_tests/

# Integration tests (requires Azure resources)
pytest tests/integration_tests/

# Specific test file
pytest tests/unit_tests/test_chat_models.py
```

### 3. Environment Configuration

**Minimum Required Variables**:
```bash
AZURE_AI_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project-id>"
```

**Additional Recommended Variables**:
```bash
# For authentication (if not using DefaultAzureCredential)
AZURE_OPENAI_API_KEY="your-key"

# For tracing
APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=..."

# For LangSmith integration
LANGCHAIN_API_KEY="your-langsmith-key"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="your-project-name"
```

## Deployment to Azure AI Foundry

### Option A: Deploy via LangGraph Platform (Recommended)

**Step 1: Prepare for Deployment**
```bash
# Ensure langgraph.json exists
cat langgraph.json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/react_agent/graph.py:graph"
  },
  "env": ".env"
}

# Test locally first
langgraph dev
```

**Step 2: Deploy to LangGraph Cloud**
```bash
# Login to LangGraph Cloud
langgraph login

# Deploy
langgraph deploy

# Or deploy to specific environment
langgraph deploy --env production
```

**Step 3: Configure in Azure AI Foundry**
1. Go to Azure AI Foundry portal
2. Navigate to Deployments → Add Deployment
3. Select "LangGraph Application"
4. Provide deployment URL from LangGraph Cloud
5. Configure authentication and scaling

### Option B: Deploy Directly to Azure AI Foundry

**Step 1: Create Agent in Portal**
1. Navigate to Azure AI Foundry
2. Go to "Agents" → "Create Agent"
3. Configure:
   - Model: gpt-4.1
   - Instructions: Your system prompt
   - Tools: Select from available tools

**Step 2: Integrate with LangGraph**
```python
from langchain_azure_ai.agents.prebuilt import PromptBasedAgentNode

# Use agent created in portal
agent_node = PromptBasedAgentNode(
    agent_id="your-agent-id",
    project_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT")
)

# Add to LangGraph workflow
workflow.add_node("azure_agent", agent_node)
```

**Step 3: Deploy**
- Agent runs in Azure AI Foundry
- Access via REST API or SDK

### Option C: Container-Based Deployment

**Step 1: Create Dockerfile**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Deploy to Azure Container Apps**
```bash
# Build and push image
az acr build --registry <registry-name> --image agent:v1 .

# Deploy to Container Apps
az containerapp create \
  --name agent-app \
  --resource-group <rg-name> \
  --image <registry>.azurecr.io/agent:v1 \
  --environment <env-name> \
  --env-vars AZURE_AI_PROJECT_ENDPOINT=<endpoint>
```

## Best Practices

### 1. Agent Design
- **Single Responsibility**: Each agent should have a clear, focused purpose
- **Clear Instructions**: Write detailed system prompts with examples
- **Tool Selection**: Only include necessary tools to reduce confusion
- **Error Handling**: Implement graceful degradation

### 2. Tool Development
```python
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """Search for information about a query.
    
    Args:
        query: The search query string
        
    Returns:
        Search results as a formatted string
    """
    # Implementation
    return results
```

**Tool Guidelines**:
- Clear docstrings (agents read these!)
- Type hints for all parameters
- Descriptive names and descriptions
- Handle errors gracefully

### 3. Tracing and Observability
```python
# Enable comprehensive tracing
from langchain_azure_ai.callbacks import AzureAIOpenTelemetryTracer

tracer = AzureAIOpenTelemetryTracer(
    connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
)

# Use with LangChain
result = agent.invoke(
    {"messages": messages},
    config={"callbacks": [tracer]}
)
```

### 4. Security
- **Never commit secrets**: Use `.env` files (add to `.gitignore`)
- **Use Managed Identity**: Prefer `DefaultAzureCredential` over API keys
- **Role-Based Access**: Assign minimal required permissions
- **Input Validation**: Sanitize user inputs before processing

### 5. Cost Optimization
- **Model Selection**: Use appropriate model tier (gpt-4-turbo vs gpt-4)
- **Streaming**: Enable streaming for better UX and cost visibility
- **Caching**: Use semantic caching for repeated queries
- **Token Limits**: Set max_tokens to control costs

## Common Patterns and Code Templates

### Pattern: Agent with Multiple Tools
```python
from langchain_azure_ai.agents import AgentServiceFactory
from langchain_azure_ai.tools import (
    AzureAIDocumentIntelligenceTool,
    AzureAIImageAnalysisTool
)
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search internal database."""
    return db.search(query)

factory = AgentServiceFactory()
agent = factory.create_prompt_agent(
    name="multi-tool-agent",
    model="gpt-4.1",
    instructions="You are a versatile assistant...",
    tools=[
        AzureAIDocumentIntelligenceTool(),
        AzureAIImageAnalysisTool(),
        search_database
    ],
    trace=True
)
```

### Pattern: Streaming Responses
```python
# For better UX and incremental responses
for chunk in agent.stream({"messages": messages}):
    if "messages" in chunk:
        for message in chunk["messages"]:
            if hasattr(message, "content"):
                print(message.content, end="", flush=True)
```

### Pattern: Structured Outputs
```python
from pydantic import BaseModel

class DocumentSummary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

# Use with structured output
agent_with_structure = agent.with_structured_output(DocumentSummary)
result = agent_with_structure.invoke({"messages": messages})
# result is a DocumentSummary instance
```

## Troubleshooting

### Issue: Agent not finding tools
**Solution**: Ensure tools are properly registered and have clear docstrings

### Issue: Tracing not working
**Solution**: 
1. Check OpenTelemetry extras installed: `pip install langchain-azure-ai[opentelemetry]`
2. Verify `APPLICATIONINSIGHTS_CONNECTION_STRING` is set
3. Enable tracing: `trace=True` in agent creation

### Issue: Authentication failures
**Solution**:
1. Run `az login` to authenticate Azure CLI
2. Verify project endpoint format
3. Check RBAC permissions on Azure resources

### Issue: Model not available
**Solution**:
1. Ensure model is deployed in your Azure OpenAI resource
2. Check deployment name matches model parameter
3. Verify region supports the model

## File Structure for New Agents

```
my-agent/
├── .env.example              # Environment template
├── .gitignore               # Ignore .env and secrets
├── langgraph.json           # LangGraph configuration
├── pyproject.toml           # Dependencies
├── README.md                # Agent documentation
├── src/
│   └── my_agent/
│       ├── __init__.py
│       ├── graph.py         # Agent definition
│       ├── prompts.py       # System prompts
│       ├── tools.py         # Custom tools (optional)
│       └── state.py         # State schema (if complex)
└── tests/
    ├── test_agent.py
    └── test_tools.py
```

## Next Steps

1. **Start with Samples**: Clone and run existing samples to understand patterns
2. **Customize**: Modify instructions and tools for your use case
3. **Test Locally**: Use `langgraph dev` for rapid iteration
4. **Deploy**: Choose deployment option based on requirements
5. **Monitor**: Set up tracing and monitoring for production
6. **Iterate**: Collect feedback and improve agent behavior

## Resources

- **Azure AI Foundry LangChain Docs**: https://aka.ms/azureai/langchain
- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **LangSmith**: https://smith.langchain.com/
- **Azure AI Foundry Portal**: https://ai.azure.com/

---

**Last Updated**: 2026-01-24  
**Target Audience**: Developers and AI Agents building with this repository
