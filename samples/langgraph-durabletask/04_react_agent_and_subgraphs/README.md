# 04 - Prebuilt ReAct Agent & Compiled Subgraphs

This sample demonstrates how to compose a prebuilt `langgraph.prebuilt.create_react_agent` as a **compiled subgraph node** inside a parent `StateGraph`, while still isolating its `@durable_tool` calls as independent **Azure Durable Task Scheduler (DTS)** tool activities.

## Key Concepts Demonstrated

- **Prebuilt `create_react_agent` Compatibility**: Using LangGraph's `create_react_agent(model, tools)` factory without modifying how the agent is constructed.
- **Compiled Subgraphs as Nodes**: Adding a compiled subgraph (`react_subgraph`) as a node (`calculator_agent`) inside a parent `StateGraph` (`prepare -> calculator_agent -> finalize`).
- **Recursive Subgraph Tool Interception**: When `attach_durability` wraps the parent graph, it recursively inspects compiled subgraphs and intercepts `@durable_tool` calls (`multiply`) inside the nested `ToolNode` so they execute as top-level `langgraph.<name>.tool` DTS activities with independent retry policies.
- **Deterministic Offline Chat Model**: Using a local `BaseChatModel` subclass (`OfflineCalcModel`) so the sample runs deterministically offline without external LLM API keys.

## Workflow Architecture

```text
Parent Graph (react_subgraph_demo):

START ──► prepare ──► calculator_agent (compiled create_react_agent subgraph) ──► finalize ──► END
                            │
                            ├─► agent (OfflineCalcModel emits tool_call: multiply(a=6, b=7))
                            │
                            ├─► tools ──► [DTS Activity: langgraph.react_subgraph_demo.tool (multiply)]
                            │
                            └─► agent (OfflineCalcModel reads ToolMessage("42") -> final AIMessage)
```

## Prerequisites & Environment Setup

### 1. Python & Dependencies

Requires **Python 3.11+**.

From the repository root:

```bash
pip install -e ".[test]"
```

Or from this sample folder:

```bash
pip install -r requirements.txt
```

### 2. Configure Azure Durable Task Scheduler

#### Option A: Local DTS Emulator (Docker)

```bash
docker run -d --name dts-emulator -p 8080:8080 -p 8082:8082 mcr.microsoft.com/dts/dts-emulator:latest
```

```bash
# Bash (Linux / macOS / WSL)
export DTS_EMULATOR=127.0.0.1:8080
```

```powershell
# PowerShell (Windows)
$env:DTS_EMULATOR = "127.0.0.1:8080"
```

> **DTS Dashboard:** Open `http://localhost:8082` to inspect the parent graph steps and the isolated `langgraph.react_subgraph_demo.tool` execution for `multiply`.

#### Option B: Hosted Azure Durable Task Scheduler (Entra ID)

```bash
# Bash (Linux / macOS / WSL)
az login
export DURABLE_TASK_SCHEDULER_CONNECTION_STRING="Endpoint=https://<scheduler>.<region>.durabletask.io;TaskHub=<hub>;Authentication=AzureCli"
```

```powershell
# PowerShell (Windows)
az login
$env:DURABLE_TASK_SCHEDULER_CONNECTION_STRING = "Endpoint=https://<scheduler>.<region>.durabletask.io;TaskHub=<hub>;Authentication=AzureCli"
```

## Running the Sample

### From This Sample Directory (`samples/langgraph-durabletask/04_react_agent_and_subgraphs/`)

**Bash:**
```bash
cd samples/langgraph-durabletask/04_react_agent_and_subgraphs
export DTS_EMULATOR=127.0.0.1:8080
python sample.py
```

**PowerShell:**
```powershell
cd examples\04_react_agent_and_subgraphs
$env:DTS_EMULATOR = "127.0.0.1:8080"
python sample.py
```

### From the Repository Root

**Bash:**
```bash
export DTS_EMULATOR=127.0.0.1:8080
python samples/langgraph-durabletask/04_react_agent_and_subgraphs/sample.py
```

**PowerShell:**
```powershell
$env:DTS_EMULATOR = "127.0.0.1:8080"
python examples\04_react_agent_and_subgraphs\sample.py
```

## Code Walkthrough

### 1. Define a `@durable_tool`

The `multiply` tool is decorated with `@durable_tool(retries=3)` so that even when invoked inside a nested `create_react_agent` subgraph, DTS schedules it as an isolated tool activity:

```python
@durable_tool(retries=3)
@tool
def multiply(a: int, b: int) -> str:
    """Multiply two integers inside a durable tool activity."""
    return str(a * b)
```

### 2. Embed `create_react_agent` Inside a Parent `StateGraph`

```python
class ParentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    trace: Annotated[list[str], add]


def build_parent_with_subgraph():
    react_subgraph = create_react_agent(OfflineCalcModel(), [multiply])

    builder = StateGraph(ParentState)
    builder.add_node("prepare", lambda s: {"trace": ["prepare"]})
    builder.add_node("calculator_agent", react_subgraph)
    builder.add_node("finalize", lambda s: {"trace": ["finalize"]})
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "calculator_agent")
    builder.add_edge("calculator_agent", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()
```

### 3. Attach Durability and Run

```python
reg = attach_durability(build_parent_with_subgraph(), name="react_subgraph_demo")
worker = DurableTaskSchedulerWorker(**sdk_options(endpoint))
sdk = DurableTaskSchedulerClient(**sdk_options(endpoint))
reg.register(worker)
worker.start()

client = reg.client(sdk)
run = client.start({"messages": [HumanMessage(content="What is 6 * 7?")], "trace": []})
out = run.wait_result(timeout=60)
print("Trace:", out["trace"])
print("Final AI reply:", out["messages"][-1].content)
```

## Expected Output

```text
Trace: ['prepare', 'finalize']
Final AI reply: The product is 42.
```
