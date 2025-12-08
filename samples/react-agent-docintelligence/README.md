# React Agent with Azure AI Document Intelligence

This sample demonstrates how to build a **ReAct (Reasoning and Action) agent** using **LangGraph** and **Azure AI Foundry Agent Service** with tool calling capabilities. The agent uses Azure AI Document Intelligence to analyze documents, images, PDFs, and other file types with high accuracy.

## Overview

This example showcases:
- **Azure AI Foundry Agent Service integration** for creating prompt-based agents
- **LangGraph** framework for building the agent workflow
- **Tool calling** with Azure AI Document Intelligence for document analysis
- **OpenTelemetry tracing** for observability

The agent follows the ReAct pattern, which combines reasoning and action by:
1. Receiving a user request
2. Reasoning about which tool to use
3. Taking action by calling the appropriate tool
4. Observing the result
5. Repeating until the task is complete

## Features

- **Document Analysis**: Parse and extract information from various document formats (PDF, images, forms, etc.)
- **Intelligent Reasoning**: Uses GPT-4 to reason about document analysis tasks
- **Tool Integration**: Seamless integration with Azure AI Document Intelligence service
- **Tracing Support**: Built-in OpenTelemetry tracing for debugging and monitoring

## Prerequisites

- Python 3.10 or higher
- Azure subscription with access to:
  - Azure AI Foundry project
  - Azure OpenAI Service (with GPT-4 deployment)
  - Azure AI Document Intelligence service
- Poetry for dependency management

## Setup

### 1. Install Dependencies

```bash
poetry install
```

### 2. Configure Environment Variables

Copy the `.env.example` file to `.env` and configure your Azure AI project endpoint:

```bash
cp .env.example .env
```

Edit the `.env` file:

```bash
AZURE_AI_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<my-project>"
```

### 3. Deploy Required Azure Resources

Ensure you have the following resources deployed in your Azure AI Foundry project:
- **Azure OpenAI**: Deploy a GPT-4 model (the code references `gpt-4.1`)
- **Azure AI Document Intelligence**: Enable the service in your project

## Project Structure

```
react-agent-docintelligence/
├── src/
│   └── react_agent/
│       ├── __init__.py       # Package initialization
│       ├── graph.py          # Agent graph definition
│       └── prompts.py        # System prompts
├── langgraph.json            # LangGraph configuration
├── pyproject.toml            # Project dependencies
├── .env.example              # Environment variable template
├── Makefile                  # Development commands
└── readme.md                 # This file
```

## Usage

### Running with LangGraph CLI

The agent is configured to run with the LangGraph CLI:

```bash
poetry run langgraph dev
```

This will start a local development server where you can interact with the agent.

### Programmatic Usage

You can also import and use the agent in your own Python code:

```python
from react_agent import graph

# Invoke the agent
result = graph.invoke({
    "messages": [
        {"role": "user", "content": "Analyze this document and extract key information..."}
    ]
})
```

## How It Works

### 1. Agent Creation

The agent is created using the `AgentServiceFactory` from `langchain-azure-ai`:

```python
service = AgentServiceFactory()
graph = service.create_prompt_agent(
    name="react-agent",
    description="A simple agent that can parse documents using Azure AI Document Intelligence.",
    model="gpt-4.1",
    instructions=SYSTEM_PROMPT,
    tools=[AzureAIDocumentIntelligenceTool()],
    trace=True
)
```

### 2. Tool Integration

The agent has access to the `AzureAIDocumentIntelligenceTool`, which enables it to:
- Analyze document structure and layout
- Extract text, tables, and key-value pairs
- Process various file formats (PDF, images, forms, receipts, invoices, etc.)

### 3. System Prompt

The agent uses a custom system prompt that defines its capabilities:

```python
SYSTEM_PROMPT = """You are a helpful AI assistant. You have the capability of analyzing
documents, images, PDFs, and other file types using the Azure AI Document Intelligence 
service tool with high level of accuracy.

System time: {system_time}"""
```

## Configuration

### LangGraph Configuration

The `langgraph.json` file defines the graph configuration:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/react_agent/graph.py:graph"
  },
  "env": ".env"
}
```

### Dependencies

Key dependencies include:
- `langchain`: Core LangChain framework
- `langchain-azure-ai[opentelemetry,tools]`: Azure AI integrations with tracing and tools
- `langgraph-cli[inmem]`: LangGraph CLI for development

## Example Interactions

Here are some example prompts you can use with the agent:

1. **Document Analysis**:
   ```
   "Analyze this invoice and extract the total amount, date, and vendor information."
   ```

2. **Form Processing**:
   ```
   "Extract all form fields from this application document."
   ```

3. **Table Extraction**:
   ```
   "Find all tables in this PDF and extract their contents."
   ```

4. **Multi-page Documents**:
   ```
   "Summarize the key points from this multi-page contract."
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
