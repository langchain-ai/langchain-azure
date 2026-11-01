# 01 - Hello Graph (Zero-Config Durable Workflow)

This sample introduces the core developer workflow for turning any standard LangGraph `StateGraph` into a crash-proof, distributed workflow on **Azure Durable Task Scheduler (DTS)** with a single call to `attach_durability(graph, name="hello_graph")`.

It runs in **Zero-Config mode** (`builder.compile()` with no external checkpointer argument required), demonstrating synchronous and asynchronous graph nodes, conditional edge routing, and `input_schema` / `output_schema` state projection.

## Key Concepts Demonstrated

- **Zero-Config Durability (`builder.compile()`)**: Compiling a `StateGraph` without a database checkpointer and letting `langchain-azure-compute[durabletask]` (`langchain_azure_compute.graph`) automatically persist and hydrate `InMemorySaver` state across worker turns via DTS's `checkpoint_bundle`.
- **`attach_durability(graph, name="hello_graph")`**: Wrapping a compiled LangGraph graph into a `Registration` that binds the DTS orchestration and activity entities.
- **`sdk_options(endpoint)`**: Auto-detecting whether `endpoint` is a local DTS emulator (`host:port`), a full connection string (`Endpoint=...;TaskHub=...;Authentication=...`), or a managed Azure DTS endpoint with `DefaultAzureCredential`.
- **Worker & Client Lifecycle**: Registering the durable graph on `DurableTaskSchedulerWorker` via `reg.register(worker)` and starting runs from `DurableTaskSchedulerClient` via `reg.client(sdk).start(...)` and `run.wait_result()`.
- **State Schema Projection**: Defining separate `HelloInput`, internal `HelloState` (with an `Annotated[list[str], operator.add]` reducer), and `HelloOutput` TypedDict schemas.

## Workflow Architecture

```text
LangGraph Topology:
                    ┌── intent == "greeting" ──► greeting ──┐
START ──► normalize ──► classify                            ├──► END
                    └── intent == "farewell" ──► farewell ──┘

Durable Task Scheduler (DTS) Entities Registered:
  • Orchestration : langgraph.hello_graph.run
  • Step Activity : langgraph.hello_graph.step   (executes Pregel supersteps)
  • Tool Activity : langgraph.hello_graph.tool   (executes @durable_tool calls)
  • Node Activity : langgraph.hello_graph.node   (executes @durable_node branches)
```

## Prerequisites & Environment Setup

### 1. Python & Dependencies

Requires **Python 3.11+**.

From the repository root (editable install with test extras):

```bash
pip install -e ".[test]"
```

Or directly from this sample folder:

```bash
pip install -r requirements.txt
```

### 2. Configure Azure Durable Task Scheduler

Choose **Option A (Local Docker Emulator)** for local development or **Option B (Hosted Azure DTS)** for cloud execution.

#### Option A: Local DTS Emulator (Docker)

Start the official DTS emulator container (gRPC port `8080`, Dashboard port `8082`):

```bash
docker run -d --name dts-emulator -p 8080:8080 -p 8082:8082 mcr.microsoft.com/dts/dts-emulator:latest
```

Set `DTS_EMULATOR` in your shell:

```bash
# Bash (Linux / macOS / WSL)
export DTS_EMULATOR=127.0.0.1:8080
```

```powershell
# PowerShell (Windows)
$env:DTS_EMULATOR = "127.0.0.1:8080"
```

> **DTS Dashboard:** Open `http://localhost:8082` in your browser to inspect orchestration histories, step inputs/outputs, and execution timelines in real time.

#### Option B: Hosted Azure Durable Task Scheduler (Entra ID)

Authenticate with Azure CLI and pass your scheduler's connection string:

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

### From This Sample Directory (`samples/langgraph-durabletask/01_hello_graph/`)

**Bash:**
```bash
cd samples/langgraph-durabletask/01_hello_graph
export DTS_EMULATOR=127.0.0.1:8080
python sample.py
```

**PowerShell:**
```powershell
cd examples\01_hello_graph
$env:DTS_EMULATOR = "127.0.0.1:8080"
python sample.py
```

### From the Repository Root

**Bash:**
```bash
export DTS_EMULATOR=127.0.0.1:8080
python samples/langgraph-durabletask/01_hello_graph/sample.py
```

**PowerShell:**
```powershell
$env:DTS_EMULATOR = "127.0.0.1:8080"
python examples\01_hello_graph\sample.py
```

## Code Walkthrough

### 1. Define Input, Internal State, and Output Schemas

`HelloState` tracks an internal `intent` field that is used for conditional routing but omitted from `HelloOutput`:

```python
class HelloInput(TypedDict):
    text: str


class HelloState(TypedDict):
    text: str
    intent: NotRequired[str]
    reply: NotRequired[str]
    visited: NotRequired[Annotated[list[str], operator.add]]


class HelloOutput(TypedDict):
    text: str
    reply: str
    visited: list[str]
```

### 2. Build and Compile the Graph in Zero-Config Mode

Notice that `classify` is an `async def` coroutine while `normalize`, `greeting`, and `farewell` are synchronous functions - `langchain-azure-compute[durabletask]` (`langchain_azure_compute.graph`) bridges both seamlessly inside the worker's step executor:

```python
def build_graph():
    builder = StateGraph(HelloState, input_schema=HelloInput, output_schema=HelloOutput)
    builder.add_node("normalize", normalize)
    builder.add_node("classify", classify)
    builder.add_node("greeting", greeting)
    builder.add_node("farewell", farewell)
    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "classify")
    builder.add_conditional_edges(
        "classify",
        lambda s: s["intent"],
        {"greeting": "greeting", "farewell": "farewell"},
    )
    builder.add_edge("greeting", END)
    builder.add_edge("farewell", END)
    # Zero-Config mode: no checkpointer argument required
    return builder.compile()
```

### 3. Attach Durability, Start the Worker, and Run Workflows

```python
reg = attach_durability(build_graph(), name="hello_graph")
worker = DurableTaskSchedulerWorker(**sdk_options(endpoint))
sdk = DurableTaskSchedulerClient(**sdk_options(endpoint))
reg.register(worker)
worker.start()

client = reg.client(sdk)
for sample_text in (" Hello World ", "See you later"):
    run = client.start({"text": sample_text})
    result = run.wait_result(timeout=30)
    print(f"Input: {sample_text!r} -> Output: {result}")
```

## Expected Output

```text
Input: ' Hello World ' -> Output: {'text': 'hello world', 'reply': 'Hello from a durable LangGraph workflow!', 'visited': ['normalize', 'classify', 'greeting']}
Input: 'See you later' -> Output: {'text': 'see you later', 'reply': 'Goodbye from a durable LangGraph workflow!', 'visited': ['normalize', 'classify', 'farewell']}
```
