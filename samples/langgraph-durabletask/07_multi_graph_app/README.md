# 07 - Multi-Graph Application (`triage`, `vacation`, `refund`)

This sample answers a common architectural question when adopting `langchain-azure-compute[durabletask]` (`langchain_azure_compute.graph`):

> *"What happens if I have multiple durable graphs in my application - can they share a single worker, client, and checkpointer, and can one durable graph delegate work to another?"*

It registers **three distinct durable graphs** (`vacation`, `refund`, and `triage`) on a **single `DurableTaskSchedulerWorker`**, shares a **single `DurableTaskSchedulerClient`** and **`SqliteSaver`**, runs all three concurrently, and demonstrates cross-graph delegation where `triage` starts and awaits a child `refund` or `vacation` durable run with a deterministic `instance_id`.

## Key Concepts Demonstrated

- **Entity Namespacing per Graph Name**: Calling `attach_durability(graph, name="<graph_name>")` namespaces all DTS entities under `langgraph.<graph_name>.{run,step,tool,node}` so multiple graphs never collide on the same worker.
- **Single Worker & Shared Checkpointer**: Registering `reg_vacation.register(worker)`, `reg_refund.register(worker)`, and `reg_triage.register(worker)` on one `DurableTaskSchedulerWorker` sharing one `SqliteSaver`.
- **Concurrent Multi-Graph Execution**: Starting `vacation`, `refund`, and `triage` runs in parallel on the same worker pool.
- **Idempotent Cross-Graph Delegation**: Inside `triage`'s `@durable_node` (`dispatch_specialist`), starting a specialist graph with a deterministic `instance_id=f"delegated-refund-{req_id}"` so activity retries attach to the same specialist orchestration.

## Workflow Architecture

```text
Single DurableTaskSchedulerWorker + Shared SqliteSaver:
  ├── langgraph.vacation.{run,step,tool,node}   (plan -> ToolNode([reserve_package]) -> finalize)
  ├── langgraph.refund.{run,step,tool,node}     (issue_refund [@durable_node])
  └── langgraph.triage.{run,step,tool,node}     (classify -> dispatch_specialist [@durable_node])
                                                                   │
                                       ┌───────────────────────────┴───────────────────────────┐
                                       ▼ (category == "vacation")                              ▼ (category == "refund")
                         vacation_client.start(                                  refund_client.start(
                           instance_id="delegated-vacation-<req_id>"               instance_id="delegated-refund-<req_id>"
                         )                                                       )
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

> **DTS Dashboard:** Open `http://localhost:8082` to see `langgraph.vacation.run`, `langgraph.refund.run`, `langgraph.triage.run`, and the delegated `delegated-refund-REQ-77-...` orchestration.

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

### From This Sample Directory (`samples/langgraph-durabletask/07_multi_graph_app/`)

**Bash:**
```bash
cd samples/langgraph-durabletask/07_multi_graph_app
export DTS_EMULATOR=127.0.0.1:8080
python sample.py
```

**PowerShell:**
```powershell
cd samples\langgraph-durabletask\07_multi_graph_app
$env:DTS_EMULATOR = "127.0.0.1:8080"
python sample.py
```

### From the Repository Root

**Bash:**
```bash
export DTS_EMULATOR=127.0.0.1:8080
python samples/langgraph-durabletask/07_multi_graph_app/sample.py
```

**PowerShell:**
```powershell
$env:DTS_EMULATOR = "127.0.0.1:8080"
python samples\langgraph-durabletask\07_multi_graph_app\sample.py
```

## Code Walkthrough

### 1. Attach Durability with Distinct Names and Register on One Worker

```python
reg_vacation = attach_durability(build_vacation_graph(shared_saver), name="vacation")
reg_refund = attach_durability(build_refund_graph(shared_saver), name="refund")

vacation_client = reg_vacation.client(sdk)
refund_client = reg_refund.client(sdk)

reg_triage = attach_durability(
    build_triage_graph(lambda: vacation_client, lambda: refund_client, shared_saver),
    name="triage",
)
triage_client = reg_triage.client(sdk)

reg_vacation.register(worker)
reg_refund.register(worker)
reg_triage.register(worker)
worker.start()
```

### 2. Delegate from `triage` to Specialist Graphs with Deterministic `instance_id`

```python
@durable_node(retries=2)
def dispatch_specialist(state: TriageState) -> dict:
    req_id = state["request_id"]
    if state["category"] == "refund":
        child_run = refund_client_fn().start(
            {"order_id": req_id, "amount": 199.00},
            instance_id=f"delegated-refund-{req_id}",
        )
        out = child_run.wait_result(timeout=30)
        return {"resolution": f"Routed to refund graph -> {out['refund_receipt']}"}
```

### 3. Run All 3 Graphs Concurrently

```python
run_vac = vacation_client.start({"messages": [HumanMessage(content="Trip to Maui")], "destination": "Maui"})
run_ref = refund_client.start({"order_id": "ORD-500", "amount": 85.50})
run_tri = triage_client.start(
    {"request_id": f"REQ-77-{os.urandom(3).hex()}", "query": "Please refund my broken headphones"}
)
```

## Expected Output

```text
1. Direct vacation graph (`langgraph.vacation.*`): Confirmed Maui: PKG-MAUI-9001
2. Direct refund graph   (`langgraph.refund.*`)  : REFUND-ORD-500-$85.50
3. Delegating triage graph (`langgraph.triage.*` -> `langgraph.refund.*`): Routed to refund graph -> REFUND-REQ-77-...-$199.00
```
