# Document Intake & Analysis Pipeline

This sample demonstrates how to build a **multi-agent document processing pipeline** using **LangGraph** and **Azure AI Foundry Agent Service**. Two specialized agents are composed in a custom `StateGraph` to parse, classify, and analyze documents end-to-end.

## Overview

Instead of a single monolithic agent, this example splits document processing into two focused agents wired together in a pipeline:

```
START → Parser Agent → (tool loop) → Prepare Analysis → Analyst Agent → END
```

| Agent | Role | Tools |
|-------|------|-------|
| **Document Parser** | Extracts raw content from documents (PDFs, images, forms, invoices, etc.) | Azure AI Document Intelligence |
| **Document Analyst** | Classifies the document, extracts key entities, summarizes content, and flags action items or risks | None (pure LLM reasoning) |

A lightweight **bridging node** (`prepare_analysis`) converts the parser's output into a prompt for the analyst, demonstrating how to hand off context between agents in a LangGraph workflow.

## What This Demonstrates

- **Custom `StateGraph`** — Building a multi-node graph instead of using the one-liner `create_prompt_agent`
- **`create_prompt_agent_node`** — Creating individual agent nodes and composing them manually
- **Agent-to-agent handoff** — Bridging output from one agent to input for the next
- **Conditional routing** — Tool-calling loop for the parser, then forwarding to analysis
- **OpenTelemetry tracing** — End-to-end observability across the full pipeline

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
│       ├── graph.py          # Multi-agent pipeline graph
│       └── prompts.py        # Specialized prompts for each agent
├── langgraph.json            # LangGraph configuration
├── pyproject.toml            # Project dependencies
├── .env.example              # Environment variable template
├── Makefile                  # Development commands
└── README.md                 # This file
```

## Usage

### Running with LangGraph CLI

The agent is configured to run with the LangGraph CLI:

```bash
poetry run langgraph dev
```

This will start a local development server where you can interact with the pipeline.

### Programmatic Usage

You can also import and use the pipeline in your own Python code:

```python
from react_agent import graph

result = graph.invoke({
    "messages": [
        {"role": "user", "content": "Analyze this invoice: https://example.com/invoice.pdf"}
    ]
})

# The last message contains the analyst's structured report
print(result["messages"][-1].content)
```

## How It Works

### 1. Agent Nodes

Two specialized agents are created using `create_prompt_agent_node` from `AgentServiceFactory`:

```python
service = AgentServiceFactory()

# Agent 1: Extracts content from documents
parser_node = service.create_prompt_agent_node(
    name="document-parser",
    model="gpt-4.1",
    instructions=PARSER_PROMPT,
    tools=[AzureAIDocumentIntelligenceTool()],
)

# Agent 2: Analyzes the extracted content
analyst_node = service.create_prompt_agent_node(
    name="document-analyst",
    model="gpt-4.1",
    instructions=ANALYST_PROMPT,
)
```

### 2. Bridging Node

Since each agent operates on its own Azure AI Foundry thread, a bridging function converts the parser's AI response into a `HumanMessage` for the analyst:

```python
def prepare_analysis(state):
    parser_output = state["messages"][-1].content
    return {"messages": [HumanMessage(content=f"Analyze this:\n\n{parser_output}")]}
```

### 3. Graph Composition

The nodes are wired together in a `StateGraph` with conditional routing for the parser's tool-calling loop:

```python
builder = StateGraph(AgentState)

builder.add_node("parser", parser_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("prepare_analysis", prepare_analysis)
builder.add_node("analyst", analyst_node)

builder.add_edge(START, "parser")
builder.add_conditional_edges("parser", route_parser_output)
builder.add_edge("tools", "parser")
builder.add_edge("prepare_analysis", "analyst")
builder.add_edge("analyst", END)
```

### 4. System Prompts

Each agent has a focused system prompt:

- **Parser**: Instructed to extract content faithfully without summarizing
- **Analyst**: Instructed to classify, extract entities, summarize, list action items, and flag risks

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

Here are some example prompts you can use with the pipeline:

1. **Invoice Processing**:
   ```
   "Parse this invoice and give me a full analysis: https://example.com/invoice.pdf"
   ```

2. **Contract Review**:
   ```
   "Review this contract and flag any risks or unusual clauses: https://example.com/contract.pdf"
   ```

3. **Form Data Extraction**:
   ```
   "Extract and analyze all fields from this application form: https://example.com/form.png"
   ```

4. **Report Summarization**:
   ```
   "Process this quarterly report and summarize the key findings: https://example.com/report.pdf"
   ```

The pipeline will first extract the raw content (parser agent), then produce a structured analysis with document type, key entities, summary, action items, and risk flags (analyst agent).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
