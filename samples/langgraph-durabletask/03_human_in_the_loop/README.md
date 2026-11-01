# 03 - Human-in-the-Loop (`interrupt()`, Parallel Gates & `run.resume()`)

This sample demonstrates how LangGraph's native `interrupt()` primitive maps onto **Azure Durable Task Scheduler (DTS)** external events (`wait_for_external_event`), allowing a workflow to pause durably for seconds, hours, or weeks with **zero worker compute consumed while waiting**.

It covers both a **single-approval workflow** (`expense_approval`) and a **parallel multi-approver workflow** (`parallel_approval`) where Finance and Legal branches pause concurrently and are resumed in a single batch call.

## Key Concepts Demonstrated

- **Native `langgraph.types.interrupt()`**: Calling `interrupt(payload)` inside any node pauses the graph and surfaces a typed `InterruptReceipt` to the caller.
- **`run.wait_pending(timeout=...)`**: Waiting for the workflow to reach an `interrupt()` pause and returning the first pending `InterruptReceipt`.
- **`run.pending_interrupts()`**: Inspecting all simultaneously open `InterruptReceipt`s when parallel graph branches (`finance_gate` and `legal_gate`) pause in the same superstep.
- **Single & Batch `run.resume(...)`**: Resuming a single interrupt with `run.resume(receipt, value)` or resuming multiple parallel interrupts at once with `run.resume({receipt: value, ...})`.
- **Pause-Scoped Checkpoint Tokens (`ckpt`)**: Under the hood, each pause receives a unique composite token (`<checkpoint_id>@<interrupt_id_digest>`) so repeated pauses across loops or sequential interrupts inside a node never collide.

## Workflow Architecture

```text
Part 1 - Single Interrupt (expense_approval):
  START ──► review_expense [calls interrupt({...})] ──(wait_for_external_event)──► END

Part 2 - Parallel Branch Interrupts (parallel_approval):
            ┌──► finance_gate [interrupt({"gate": "finance"})] ──┐
  START ────┤                                                    ├──► finalize ──► END
            └──► legal_gate   [interrupt({"gate": "legal"})]   ──┘
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

> **DTS Dashboard:** Open `http://localhost:8082` to inspect the orchestration custom status while paused and the `EventRaised` entries (`resume:<interrupt_id>`) when resumed.

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

### From This Sample Directory (`samples/langgraph-durabletask/03_human_in_the_loop/`)

**Bash:**
```bash
cd samples/langgraph-durabletask/03_human_in_the_loop
export DTS_EMULATOR=127.0.0.1:8080
python sample.py
```

**PowerShell:**
```powershell
cd samples\langgraph-durabletask\03_human_in_the_loop
$env:DTS_EMULATOR = "127.0.0.1:8080"
python sample.py
```

### From the Repository Root

**Bash:**
```bash
export DTS_EMULATOR=127.0.0.1:8080
python samples/langgraph-durabletask/03_human_in_the_loop/sample.py
```

**PowerShell:**
```powershell
$env:DTS_EMULATOR = "127.0.0.1:8080"
python samples\langgraph-durabletask\03_human_in_the_loop\sample.py
```

## Code Walkthrough

### 1. Single Interrupt Node & Client Resume

Inside `review_expense`, `interrupt(...)` suspends execution and returns the human decision once resumed:

```python
def review_expense(state: ExpenseState) -> dict:
    decision = interrupt(
        {
            "question": "Approve expense reimbursement?",
            "employee": state["employee"],
            "amount": state["amount"],
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    return {
        "approved": approved,
        "status": f"Approved (${state['amount']:.2f})" if approved else "Rejected",
    }
```

From the client, `wait_pending()` blocks until the orchestrator publishes the pending interrupt in its custom status, and `run.resume(receipt, ...)` delivers the external event:

```python
run1 = client_single.start({"employee": "Ada Lovelace", "amount": 420.00})
receipt = run1.wait_pending(timeout=30)
run1.resume(receipt, {"approved": True})
print("Final state:", run1.wait_result(timeout=30))
```

### 2. Parallel Interrupt Gates & Batch Resume

When `START` fans out to both `finance_gate` and `legal_gate`, both call `interrupt(...)` in the same superstep. `run.pending_interrupts()` returns a tuple of both `InterruptReceipt` objects, and passing a dictionary to `run.resume({...})` resolves both in one turn:

```python
run2 = client_parallel.start({"proposal": "Enterprise Vendor Contract", "decisions": []})
run2.wait_pending(timeout=30)
receipts = run2.pending_interrupts()
run2.resume({r: f"approved-by-{r.value['gate']}" for r in receipts})
print("Parallel final state:", run2.wait_result(timeout=30))
```

## Expected Output

```text
1. Single interrupt receipt: InterruptReceipt(id='...', value={'question': 'Approve expense reimbursement?', 'employee': 'Ada Lovelace', 'amount': 420.0}, ckpt='...')
   Final state: {'employee': 'Ada Lovelace', 'amount': 420.0, 'approved': True, 'status': 'Approved ($420.00)'}
2. Parallel pending interrupts: (InterruptReceipt(id='...', value={'gate': 'finance', 'proposal': 'Enterprise Vendor Contract'}, ckpt='...'), InterruptReceipt(id='...', value={'gate': 'legal', 'proposal': 'Enterprise Vendor Contract'}, ckpt='...'))
   Parallel final state: {'proposal': 'Enterprise Vendor Contract', 'decisions': ['finance:approved-by-finance', 'legal:approved-by-legal'], 'final_status': 'finance:approved-by-finance & legal:approved-by-legal'}
```
