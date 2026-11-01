# 06 - Durable Nodes (`@durable_node` & `@durable_node`) & Map-Reduce Fan-Out

This sample demonstrates how to turn **parallel graph branches** and **dynamic `Send()` map-reduce workers** into independent **Azure Durable Task Scheduler (DTS)** node activities (`langgraph.<name>.node`) using `@durable_node` and `@durable_node`.

When branches are marked as durable nodes, the DTS orchestrator fans them out across your worker pool (`ctx.when_all([...])`), retries only the failing branch on transient errors without re-running completed sibling branches, and supports bounded batch concurrency via `node_max_concurrency`.

## Key Concepts Demonstrated

- **Static Parallel Fan-Out (`START -> [branch_a, branch_b, branch_c] -> join`)**: Decorating parallel nodes with `@durable_node`, `@durable_node(retries=3)`, or `durable_node(branch_c, durable=True, retries=2)` so each branch executes as its own `langgraph.<name>.node` activity.
- **Dynamic Map-Reduce (`Send("research_worker", item)`)**: Returning `[Send("research_worker", {"topic": t}) for t in state["topics"]]` from a conditional edge and executing each `Send` branch as an isolated DTS node activity.
- **Per-Branch Retry Isolation**: In Part 1, `branch_b` fails once on attempt 1 (`branch_a` and `branch_c` run only once). In Part 2, the `"security"` topic fails once on attempt 1 (`"architecture"` and `"performance"` run only once).
- **Concurrency Throttling (`node_max_concurrency=2`)**: Bounding how many `langgraph.<name>.node` activities the orchestrator schedules in parallel per batch.

## Workflow Architecture

```text
Part 1 - Static Parallel Fan-Out (static_parallel_nodes):
          ┌──► branch_a (@durable_node)            ──► [langgraph.static_parallel_nodes.node] (1 attempt)  ──┐
  START ──┼──► branch_b (@durable_node(retries=3)) ──► [langgraph.static_parallel_nodes.node] (2 attempts) ──┼──► join ──► END
          └──► branch_c (durable_node(..., retries=2))──► [langgraph.static_parallel_nodes.node] (1 attempt)  ──┘

Part 2 - Dynamic Send() Map-Reduce (research_map_reduce, node_max_concurrency=2):
                        ┌── Send("research_worker", {"topic": "architecture"}) ──► [node activity] (1 attempt)  ──┐
  START ──► planner ────┼── Send("research_worker", {"topic": "security"})     ──► [node activity] (2 attempts) ──┼──► synthesize ──► END
                        └── Send("research_worker", {"topic": "performance"})  ──► [node activity] (1 attempt)  ──┘
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

> **DTS Dashboard:** Open `http://localhost:8082` to inspect the parallel `langgraph.*.node` activities and per-branch retries.

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

### From This Sample Directory (`samples/langgraph-durabletask/06_durable_nodes_map_reduce/`)

**Bash:**
```bash
cd samples/langgraph-durabletask/06_durable_nodes_map_reduce
export DTS_EMULATOR=127.0.0.1:8080
python sample.py
```

**PowerShell:**
```powershell
cd samples\langgraph-durabletask\06_durable_nodes_map_reduce
$env:DTS_EMULATOR = "127.0.0.1:8080"
python sample.py
```

### From the Repository Root

**Bash:**
```bash
export DTS_EMULATOR=127.0.0.1:8080
python samples/langgraph-durabletask/06_durable_nodes_map_reduce/sample.py
```

**PowerShell:**
```powershell
$env:DTS_EMULATOR = "127.0.0.1:8080"
python samples\langgraph-durabletask\06_durable_nodes_map_reduce\sample.py
```

## Code Walkthrough

### 1. Static Parallel Branches with `@durable_node` and `@durable_node`

```python
@durable_node
def branch_a(state: StaticParallelState) -> dict:
    ...
    return {"results": [f"A={state['seed'] + 1}"]}


@durable_node(retries=3)
def branch_b(state: StaticParallelState) -> dict:
    ...
    if attempt == 1:
        raise RuntimeError("Transient failure in branch_b (attempt 1)")
    return {"results": [f"B={state['seed'] + 2}"]}


def branch_c(state: StaticParallelState) -> dict:
    ...
    return {"results": [f"C={state['seed'] + 3}"]}


durable_node(branch_c, durable=True, retries=2)
```

### 2. Dynamic `Send()` Map-Reduce with `@durable_node` and `node_max_concurrency`

`dispatch_workers` emits a `Send("research_worker", {"topic": topic})` per topic, and `research_worker` is decorated with `@durable_node(retries=3)`:

```python
def dispatch_workers(state: ResearchState) -> list[Send]:
    return [Send("research_worker", {"topic": topic}) for topic in state["topics"]]


@durable_node(retries=3)
def research_worker(arg: dict) -> dict:
    topic = arg["topic"]
    ...
    if topic == "security" and attempt == 1:
        raise RuntimeError("Simulated transient timeout in security worker (attempt 1)")
    return {"findings": [f"REPORT[{topic}]"]}
```

When attaching durability (or starting a run), `node_max_concurrency=2` caps concurrent branch activities to 2 at a time:

```python
reg_mr = attach_durability(
    build_map_reduce_graph(call_counts),
    name="research_map_reduce",
    node_max_concurrency=2,
)
```

## Expected Output

```text
1. Static parallel summary: A=11 + B=12 + C=13
   Static branch attempts (only branch_b retried): {'branch_a': 1, 'branch_b': 2, 'branch_c': 1}
2. Map-reduce synthesis: PLANNED(3 topics) | REPORT[architecture] | REPORT[performance] | REPORT[security]
   Map-reduce attempt counts (only 'security' retried): {'architecture': 1, 'security': 2, 'performance': 1}
```
