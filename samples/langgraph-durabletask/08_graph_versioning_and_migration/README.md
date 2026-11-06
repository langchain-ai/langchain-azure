# 08 - Graph Versioning & Live Checkpoint Migration (`v1` → `v2`)

Long-running durable workflows often outlive a single code deployment. When you evolve a graph's topology - renaming nodes, changing state schemas, or inserting new compliance steps - existing in-flight runs on `"v1"` need a safe path forward without corrupting state or breaking replay determinism.

This sample demonstrates `langchain-azure-compute[durabletask]` (`langchain_azure_compute.graph`)'s **version registry and live checkpoint migration system** (`version`, `previous_versions`, `migrations`, `upgrade_on_continue_as_new=True`, and `run.upgrade("v2", ...)`).

## Key Concepts Demonstrated

- **Side-by-Side Version Dispatch (`version` & `previous_versions`)**: Registering `version="v2"` alongside `previous_versions={"v1": graph_v1}` so the worker can execute both existing `"v1"` runs and new `"v2"` runs.
- **State & Node Remapping (`migrations={("v1", "v2"): fn}`)**: Defining a migration function that transforms the checkpointed state - and optionally remaps a renamed pause node via `{"as_node": "new_approval", "values": ...}` - before the first `"v2"` step executes.
- **Automatic Rollover Upgrade (`upgrade_on_continue_as_new=True`)**: Automatically migrating a running `"v1"` workflow to `"v2"` when the orchestrator rolls over across a `continue_as_new` generation boundary (`Limits(steps_per_generation=1)`).
- **Explicit Mid-Run Upgrade (`run.upgrade("v2", resume=..., receipt=...)`)**: Upgrading a `"v1"` workflow while it is paused at `interrupt()` inside `old_approval` so it resumes as `new_approval` on `"v2"` and executes the newly added downstream `audit_v2` node.

## Workflow Architecture

```text
Part 1 - Auto-Upgrade Across continue_as_new (order_auto_upgrade):
  [Generation 1 on v1]
    START ──► step1 (+1, log="v1:step1")
                │
                ▼ (steps_per_generation=1 triggers continue_as_new + upgrade_on_continue_as_new=True)
  [Migration ("v1", "v2")]
    total = total + 100, log += ["migrated:v1->v2"]
                │
                ▼
  [Generation 2 on v2]
    step2 (*10, log="v2:step2") ──► step3_v2 (log="v2:step3_audit") ──► END

Part 2 - Explicit Upgrade While Paused at interrupt() (order_interrupt_upgrade):
  [Starts on v1]
    START ──► prep (+1, log="v1:prep") ──► old_approval [pauses at interrupt('Approve order on v1?')]
                                                │
                                                ▼ Caller invokes: run.upgrade("v2", resume="approved-by-manager")
  [Migration ("v1", "v2") remaps as_node="new_approval"]
                                                │
                                                ▼
  [Resumes on v2]
    finish (*10, log="v2:finish") ──► audit_v2 (log="v2:audit") ──► END
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

> **DTS Dashboard:** Open `http://localhost:8082` to inspect the `continue_as_new` transition and the `lg.upgrade` external event in the orchestration history.

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

### From This Sample Directory (`samples/langgraph-durabletask/08_graph_versioning_and_migration/`)

**Bash:**
```bash
cd samples/langgraph-durabletask/08_graph_versioning_and_migration
export DTS_EMULATOR=127.0.0.1:8080
python sample.py
```

**PowerShell:**
```powershell
cd samples\langgraph-durabletask\08_graph_versioning_and_migration
$env:DTS_EMULATOR = "127.0.0.1:8080"
python sample.py
```

### From the Repository Root

**Bash:**
```bash
export DTS_EMULATOR=127.0.0.1:8080
python samples/langgraph-durabletask/08_graph_versioning_and_migration/sample.py
```

**PowerShell:**
```powershell
$env:DTS_EMULATOR = "127.0.0.1:8080"
python samples\langgraph-durabletask\08_graph_versioning_and_migration\sample.py
```

## Code Walkthrough

### 1. Automatic Upgrade Across `continue_as_new`

```python
reg_auto = attach_durability(
    g_can_v2,
    name="order_auto_upgrade",
    limits=Limits(steps_per_generation=1),
    version="v2",
    previous_versions={"v1": g_can_v1},
    migrations={
        ("v1", "v2"): lambda state: {
            "total": state["total"] + 100,
            "log": state["log"] + ["migrated:v1->v2"],
        }
    },
    upgrade_on_continue_as_new=True,
)
```

Starting the workflow with `version="v1"` runs `step1` on `g_can_v1`, rolls over at the generation boundary, runs the `("v1", "v2")` migration, and finishes `step2` and `step3_v2` on `g_can_v2`:

```python
run_auto = reg_auto.client(sdk).start({"total": 5, "log": []}, version="v1")
out_auto = run_auto.wait_result(timeout=60)
```

### 2. Explicit Live Upgrade at an `interrupt()` Pause

When `"v2"` renames `old_approval` to `new_approval` and adds `audit_v2`, the 3-argument migration `(state, snap, resume_val)` returns `{"as_node": "new_approval", "values": ...}`:

```python
reg_explicit = attach_durability(
    g_int_v2,
    name="order_interrupt_upgrade",
    version="v2",
    previous_versions={"v1": g_int_v1},
    migrations={
        ("v1", "v2"): lambda state, snap, resume_val: {
            "as_node": "new_approval",
            "values": {"log": state["log"] + [f"v2:upgraded_approval:{resume_val}"]},
        }
    },
)

run_exp = reg_explicit.client(sdk).start({"total": 4, "log": []}, version="v1")
receipt = run_exp.wait_pending(timeout=30)
run_exp.upgrade("v2", resume="approved-by-manager", receipt=receipt)
out_exp = run_exp.wait_result(timeout=60)
```

## Expected Output

```text
1. Auto-upgrade on continue_as_new (final version='v2'): {'total': 1060, 'log': ['v1:step1', 'migrated:v1->v2', 'v2:step2', 'v2:step3_audit']}
2. Paused on v1 interrupt: InterruptReceipt(id='...', value='Approve order on v1?', ckpt='...')
   Explicit upgrade on resume (final version='v2'): {'total': 50, 'log': ['v1:prep', 'v2:upgraded_approval:approved-by-manager', 'v2:finish', 'v2:audit']}
```
