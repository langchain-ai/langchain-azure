# 02 - Vacation Booking (Per-Tool Durable Activities & Idempotent Retries)

This sample builds a multi-tool vacation booking workflow (`plan_trip -> ToolNode([book_flight, book_hotel, book_car]) -> review -> confirm`) to demonstrate per-tool retry isolation and **at-least-once activity execution with external idempotency keys**.

## Key Concepts Demonstrated

- **Per-Tool Activity Isolation (`@durable_tool(retries=3)`)**: Wrapping LangChain `@tool` functions so each tool invocation inside a `ToolNode` batch runs as its own isolated DTS activity (`langgraph.<name>.tool`) with its own retry policy.
- **Parallel Tool Fan-Out (`when_all`)**: When `plan_trip` emits 3 tool calls (`book_flight`, `book_hotel`, `book_car`) in one `AIMessage`, the DTS orchestrator schedules all 3 tool activities concurrently in a single batch.
- **Isolated Retry, Not Exactly-Once Execution**: `book_car` raises a transient error on attempt 1. Completed siblings are recovered from DTS history rather than ordinarily repeated with the failed tool. A worker crash can still redeliver any activity: the external booking API must deduplicate by the durable call ID.
- **Idempotent Client Retries (`client.start(..., instance_id=...)`)**: Calling `client.start` a second time with the same `instance_id` attaches to the already-completed orchestration instead of starting a duplicate workflow.

## Workflow Architecture

```text
LangGraph Topology & DTS Execution Boundary:

START
  │
  ▼
plan_trip                ──► [langgraph.vacation_planner.step]
  │                          Emits 3 tool_calls in one AIMessage
  ▼
ToolNode ("tools")       ──► Orchestrator fans out via ctx.when_all([...]):
  ├── book_flight            • [langgraph.vacation_planner.tool] (FLIGHT-1001 ✓)
  ├── book_hotel             • [langgraph.vacation_planner.tool] (HOTEL-1002  ✓)
  └── book_car               • [langgraph.vacation_planner.tool] (Attempt 1 ✗ -> Retry -> CAR-1003 ✓)
  │
  ▼
review ──► confirm ──► END   [langgraph.vacation_planner.step]
```

## Prerequisites & Environment Setup

### 1. Python & Dependencies

Requires **Python 3.11+**.

From the repository root:

```bash
pip install -e "libs/azure-compute[durabletask]"
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

> **DTS Dashboard:** Open `http://localhost:8082` to see the 3 parallel `langgraph.vacation_planner.tool` activities and the retry on `book_car`.

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

### From This Sample Directory (`samples/langgraph-durabletask/02_vacation_booking/`)

**Bash:**
```bash
cd samples/langgraph-durabletask/02_vacation_booking
export DTS_EMULATOR=127.0.0.1:8080
python sample.py
```

**PowerShell:**
```powershell
cd samples\langgraph-durabletask\02_vacation_booking
$env:DTS_EMULATOR = "127.0.0.1:8080"
python sample.py
```

### From the Repository Root

**Bash:**
```bash
export DTS_EMULATOR=127.0.0.1:8080
python samples/langgraph-durabletask/02_vacation_booking/sample.py
```

**PowerShell:**
```powershell
$env:DTS_EMULATOR = "127.0.0.1:8080"
python samples\langgraph-durabletask\02_vacation_booking\sample.py
```

## Code Walkthrough

### 1. Decorate Side-Effecting Tools with `@durable_tool`

Stack `@durable_tool(retries=3)` above (or below) LangChain's `@tool`. Tools receive the server-derived durable call ID through `ToolRuntime.config`; it is stable across activity retries and distinct across invocations. The sample JSON-lines ledger looks up this key before appending a booking:

```python
from langgraph.prebuilt.tool_node import ToolRuntime

@durable_tool(retries=3)
@tool
def book_flight(destination: str, runtime: ToolRuntime) -> str:
    """Book a flight to the destination."""
    key = runtime.config["metadata"]["durable_call_id"]
    return record_booking("FLIGHT", destination, key)


@durable_tool(retries=3)
@tool
def book_hotel(destination: str, runtime: ToolRuntime) -> str:
    """Book a hotel at the destination."""
    key = runtime.config["metadata"]["durable_call_id"]
    return record_booking("HOTEL", destination, key)


@durable_tool(retries=3)
@tool
def book_car(destination: str, runtime: ToolRuntime) -> str:
    """Book a rental car at the destination (simulates a transient failure on attempt 1)."""
    with log_lock:
        seq["car_attempts"] += 1
        attempt = seq["car_attempts"]
    if fail_car_once and attempt == 1:
        raise RuntimeError("Transient rental car API outage on attempt 1")
    key = runtime.config["metadata"]["durable_call_id"]
    return record_booking("CAR", destination, key)
```

The ledger is a single-process teaching example, not distributed transactional
storage. In production pass the same key to the booking provider's idempotency
API or atomically enforce a database uniqueness constraint.

### 2. Wire Up a Standard LangGraph `ToolNode`

No changes are required to your LangGraph graph construction - use `langgraph.prebuilt.ToolNode` as usual:

```python
builder = StateGraph(VacationState)
builder.add_node("plan_trip", plan_trip)
builder.add_node("tools", ToolNode([book_flight, book_hotel, book_car]))
builder.add_node("review", review)
builder.add_node("confirm", confirm)
builder.add_edge(START, "plan_trip")
builder.add_edge("plan_trip", "tools")
builder.add_edge("tools", "review")
builder.add_edge("review", "confirm")
builder.add_edge("confirm", END)
```

### 3. Run and Verify Idempotent Client Retries via `instance_id`

Passing the same `instance_id` on a client retry returns the existing run handle without re-invoking the graph or re-booking:

```python
run_id = f"vacation-kyoto-{uuid.uuid4().hex[:6]}"
inp = {"messages": [HumanMessage(content="Book trip to Kyoto")], "destination": "Kyoto"}

run1 = client.start(inp, instance_id=run_id)
result1 = run1.wait_result(timeout=60)

# Retrying with the same instance_id attaches to the completed run without re-booking
run2 = client.start(inp, instance_id=run_id)
result2 = run2.wait_result(timeout=60)
```

## Expected Output

```text
Confirmations: ['FLIGHT-1001', 'HOTEL-1002', 'CAR-1003']
Idempotent retry matched: True

Idempotent booking ledger (3 unique call IDs in this demonstration):
{"call_id": "...", "kind": "FLIGHT", "destination": "Kyoto", "confirmation": "FLIGHT-1001"}
{"call_id": "...", "kind": "HOTEL", "destination": "Kyoto", "confirmation": "HOTEL-1002"}
{"call_id": "...", "kind": "CAR", "destination": "Kyoto", "confirmation": "CAR-1003"}
```
