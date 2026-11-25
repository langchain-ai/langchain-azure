# 09 - Two-Process Worker Crash Recovery (`InMemorySaver` Handoff)

This sample proves that **Zero-Config mode** (`builder.compile()` with no external database checkpointer) is **fully crash-proof across separate OS worker processes**.

It spawns two independent OS worker processes (`pod-A` and `pod-B`), starts 3 concurrent order fulfillment workflows (`ORD-9001`, `ORD-9002`, `ORD-9003`) on `pod-A`, **hard-kills `pod-A` (`proc_a.kill()`)** after it executes `prepare_order` and `@durable_tool` `reserve_inventory`, sends human approval events while **zero workers are alive**, and then spawns a brand-new `pod-B` process (with a completely empty `InMemorySaver`) that seamlessly hydrates all 3 workflows from DTS's `checkpoint_bundle` and finishes shipping every order.

## Key Concepts Demonstrated

- **Zero-Config `checkpoint_bundle` Hydration**: After every superstep and durable tool activity in Zero-Config mode (`inline_checkpoints=True`), `langchain-azure-compute[durabletask]` (`langchain_azure_compute.graph`) exports the active thread's checkpoint slice into the DTS orchestration state. Any new worker process (`pod-B`) hydrates its fresh `InMemorySaver` before running the next step.
- **Multi-Instance Isolation**: Running 3 concurrent workflow instances (`ORD-9001`, `ORD-9002`, `ORD-9003`) across a hard process kill (`SIGKILL` / `TerminateProcess`) with zero cross-talk or lost state.
- **Zero-Worker Durable Event Queuing**: Calling `run.resume("approved-by-vp")` while **zero worker processes are running** - DTS durably queues the external events and delivers them as soon as `pod-B` comes online.
- **`Command(update=...)` inside `@durable_tool`**: Updating both graph state (`items_reserved=True`, `pid_trace`) and `ToolMessage` history directly from a durable tool activity using `ToolRuntime`.

## Workflow Architecture

```text
Process 1: pod-A (OS PID 1 - fresh InMemorySaver)
  START ──► prepare_order (PID 1) ──► @durable_tool reserve_inventory (PID 1) ──► manager_approval [pauses at interrupt()]
                                                                                          │
  💥 proc_a.kill() - pod-A RAM and InMemorySaver are completely destroyed!                │
  📩 Client calls run.resume("approved-by-vp") for ORD-9001, ORD-9002, ORD-9003           │
     while ZERO worker processes are alive (queued durably in DTS)                        │
                                                                                          ▼
Process 2: pod-B (OS PID 2 - brand-new empty InMemorySaver)
  Hydrates InMemorySaver from DTS checkpoint_bundle ──► manager_approval (PID 2) ──► ship_order (PID 2) ──► END
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

> **DTS Dashboard:** Open `http://localhost:8082` to inspect all 3 concurrent `langgraph.order_crash_recovery.run` instances and see their pre-crash and post-crash step histories.

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

### From This Sample Directory (`samples/langgraph-durabletask/09_two_process_crash_recovery/`)

**Bash:**
```bash
cd samples/langgraph-durabletask/09_two_process_crash_recovery
export DTS_EMULATOR=127.0.0.1:8080
python sample.py
```

**PowerShell:**
```powershell
cd samples\langgraph-durabletask\09_two_process_crash_recovery
$env:DTS_EMULATOR = "127.0.0.1:8080"
python sample.py
```

### From the Repository Root

**Bash:**
```bash
export DTS_EMULATOR=127.0.0.1:8080
python samples/langgraph-durabletask/09_two_process_crash_recovery/sample.py
```

**PowerShell:**
```powershell
$env:DTS_EMULATOR = "127.0.0.1:8080"
python samples\langgraph-durabletask\09_two_process_crash_recovery\sample.py
```

## Code Walkthrough

### 1. Record OS PID and Pod Name in Every Step & Tool

Each node and `@durable_tool` appends `f"<step>:pid={os.getpid()}:pod={pod_name}"` to `pid_trace`, and `ship_order` asserts that `items_reserved` (set on `pod-A` before the crash) is still `True` when `pod-B` executes:

```python
@durable_tool(retries=3)
@tool
def reserve_inventory(order_id: str, runtime: ToolRuntime[Any, Any]) -> Command:
    step_entry = f"reserve_inventory:pid={os.getpid()}:pod={pod_name}"
    return Command(
        update={
            "items_reserved": True,
            "pid_trace": [step_entry],
            "messages": [
                ToolMessage(
                    content=f"Inventory reserved for {order_id} on {pod_name} (pid={os.getpid()})",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
```

### 2. Hard-Kill `pod-A`, Resume with Zero Workers Alive, and Finish on `pod-B`

```python
# 1. Spawn Process 1 (pod-A)
proc_a, pid_a = _spawn_worker_subprocess("pod-A", graph_name=graph_name, endpoint=endpoint)

# 2. Start 3 concurrent orders and wait until all pause at manager_approval on pod-A
runs = {oid: client.start({"order_id": oid}) for oid in order_ids}
receipts = {oid: run.wait_pending(timeout=60) for oid, run in runs.items()}

# 3. HARD KILL Process 1 (pod-A) - its RAM and InMemorySaver are completely wiped
proc_a.kill()
proc_a.wait(timeout=10)

# 4. While ZERO workers are alive, send human-in-the-loop resume events to DTS
for oid, run in runs.items():
    run.resume("approved-by-vp")

# 5. Spawn Process 2 (pod-B) - brand-new OS process with an empty InMemorySaver
proc_b, pid_b = _spawn_worker_subprocess("pod-B", graph_name=graph_name, endpoint=endpoint)

# 6. Wait for all 3 instances to hydrate from DTS checkpoint_bundle and complete on pod-B
results = {oid: run.wait_result(timeout=60) for oid, run in runs.items()}
```

## Expected Output

```text
[1/6] Spawned Worker Process 1 (pod-A) -> OS PID=21480 (fresh InMemorySaver)
[2/6] All 3 concurrent instances (ORD-9001, ORD-9002, ORD-9003) executed prepare_order + @durable_tool reserve_inventory on pod-A (PID=21480) and paused at manager_approval:
      - ORD-9001: interrupt payload = {'order_id': 'ORD-9001', 'items_reserved': True, 'question': 'Approve shipment?'}
      - ORD-9002: interrupt payload = {'order_id': 'ORD-9002', 'items_reserved': True, 'question': 'Approve shipment?'}
      - ORD-9003: interrupt payload = {'order_id': 'ORD-9003', 'items_reserved': True, 'question': 'Approve shipment?'}
[3/6] HARD KILLED Worker Process 1 (pod-A, PID=21480) -> InMemorySaver RAM destroyed!
[4/6] Sent run.resume('approved-by-vp') for all 3 instances while ZERO workers are alive
[5/6] Spawned Worker Process 2 (pod-B) -> OS PID=29844 (brand-new empty InMemorySaver)
[6/6] All 3 instances hydrated InMemorySaver from DTS checkpoint_bundle and completed on pod-B!
      - ORD-9001: items_reserved=True, approved_by='approved-by-vp', shipped=True
        pid_trace = ['prepare_order:pid=21480:pod=pod-A', 'reserve_inventory:pid=21480:pod=pod-A', 'manager_approval:pid=29844:pod=pod-B', 'ship_order:pid=29844:pod=pod-B']
      - ORD-9002: items_reserved=True, approved_by='approved-by-vp', shipped=True
        pid_trace = ['prepare_order:pid=21480:pod=pod-A', 'reserve_inventory:pid=21480:pod=pod-A', 'manager_approval:pid=29844:pod=pod-B', 'ship_order:pid=29844:pod=pod-B']
      - ORD-9003: items_reserved=True, approved_by='approved-by-vp', shipped=True
        pid_trace = ['prepare_order:pid=21480:pod=pod-A', 'reserve_inventory:pid=21480:pod=pod-A', 'manager_approval:pid=29844:pod=pod-B', 'ship_order:pid=29844:pod=pod-B']

Verified cross-process InMemorySaver handoff: pod-A (PID 21480) -> [CRASH] -> pod-B (PID 29844) across 3 concurrent instances with 0 state loss.
```
