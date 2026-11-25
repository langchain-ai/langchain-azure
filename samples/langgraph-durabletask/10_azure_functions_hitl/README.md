# 10 - Azure Functions Human-in-the-Loop (`expense_approval`) Across Two Workers

This sample demonstrates hosting a stateful LangGraph workflow with `@durable_tool` and human-in-the-loop `interrupt()` pauses on **Azure Functions Python v2** and **Azure Durable Task Scheduler (DTS)** using `create_function_app(...)`.

It includes:
- `function_app.py`, `host.json`, `local.settings.json.example`, and `requirements.txt` - a ready-to-run Azure Functions Python v2 project exposing REST endpoints to start, inspect, and resume paused LangGraph workflows over HTTP.
- `sample.py` - an automated end-to-end verification script that starts the HTTP Function routes, runs pre-interrupt steps on **`worker-1`**, **hard-kills `worker-1`** while the workflow is waiting for human approval, posts the human decision to `POST /api/graphs/expense_approval/respond/{instanceId}/{interruptId}`, and verifies that **`worker-2`** (a separate OS process) hydrates the checkpoint and completes the payout.

## Key Concepts Demonstrated

- **One-Call Azure Functions Hosting (`create_function_app(reg)`)**: Turning one or more durable `Registration` objects into an `azure.functions.FunctionApp` (or attaching routes to an existing `FunctionApp` via `reg.register_function_routes(app)`).
- **Turnkey HTTP Lifecycle Endpoints**:
  - `POST /api/graphs/expense_approval/start` - starts a durable workflow run (`202 Accepted`) and returns `instanceId`, `threadId`, `statusQueryGetUri`, and `respondUri`.
  - `GET /api/graphs/expense_approval/status/{instance_id}` - returns `runtimeStatus`, `pendingInterrupts` (each with `interruptId`, `value`, and `respondUrl`), and completed `output`.
  - `POST /api/graphs/expense_approval/respond/{instance_id}[/{interrupt_id}]` - delivers a human JSON decision (e.g., `{"approved": true, "notes": "..."}`) to resume a paused `interrupt()`. The server stamps the verified approver identity, never trusting a body-supplied reviewer.
  - `GET /api/health` - returns health status and registered graph names.
- **Cross-Worker HTTP Interrupt Resume**: Proving that the HTTP host and DTS workers are decoupled - `worker-1` can be terminated while waiting for human input, and `worker-2` seamlessly resumes the paused graph when the HTTP `/respond` request arrives.

## Workflow Architecture

### Supported Production Topologies

- **Topology A (recommended for standalone `DurableTaskSchedulerWorker`):**
  Azure Functions, ACA, or App Service serves thin HTTP control routes with
  `start_worker=False`. Dedicated worker replicas run on ACA or AKS, scaled with
  the `azure-durabletask-scheduler` KEDA scaler, or on a Dedicated App Service
  plan with Always On. Use the same scheduler connection and task hub for every
  HTTP host and worker. The automated demo clears conflicting child scheduler
  variables and explicitly forwards the parent's resolved endpoint and hub.
- **Topology B (local development / Dedicated Always-On only):** explicitly
  enable `start_worker=True` (or `START_EMBEDDED_WORKER=1`) for a single-process
  `func start` session or a Dedicated plan with Always On enabled.

Every helper and this sample default to **no embedded worker**. Consumption and
Flex plans must use topology A: their scale controller cannot see an HTTP-only
app's DTS thread, may terminate it mid-activity, and can scale to zero after
`/respond` returns 200. That response confirms event delivery, not worker liveness.
Run a dedicated worker before using the default `local.settings.json.example`.

```text
HTTP Client / Approver
  │
  ├─ 1. POST /api/graphs/expense_approval/start
  │       │
  │       ▼
  │     [worker-1 (PID 1)]
  │       START ──► submit_expense ──► @durable_tool policy_check ──► manager_review [calls interrupt({...})]
  │
  ├─ 2. GET /api/graphs/expense_approval/status/{instanceId}
  │       └── Returns pendingInterrupts[0] with interruptId + respondUrl
  │
  │     💥 worker-1 (PID 1) is hard-killed while workflow is paused at interrupt()!
  │
  ├─ 3. POST /api/graphs/expense_approval/respond/{instanceId}/{interruptId}
  │       Body: {"approved": true, "notes": "Approved for Q4 travel"}
  │       Verified caller: vp-finance (distinct from the expense starter)
  │       │
  │       ▼
  │     [worker-2 (PID 2)]
  │       manager_review (resumes) ──┬── approved == True  ──► process_payout ──► END
  │                                  └── approved == False ──► reject_expense ──► END
  │
  └─ 4. GET /api/graphs/expense_approval/status/{instanceId}
          └── runtimeStatus == "COMPLETED", worker_trace shows worker-1 -> worker-2 handoff
```

## Prerequisites & Environment Setup

### 1. Python & Dependencies

Requires **Python 3.11+**.

From the repository root:

```bash
pip install -e "libs/azure-compute[durabletask]" -r samples/langgraph-durabletask/10_azure_functions_hitl/requirements.txt
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

> **DTS Dashboard:** Open `http://localhost:8082` to inspect the `expense_approval` orchestrations (`EXP-501` and `EXP-502`), tool activities, and `resume:<interrupt_id>` external events.

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

### Identity and Ownership Contract

Function keys authorize access to the Function host, not individual users.
By default these routes require verified per-instance identity: deploy behind
Azure App Service Authentication (EasyAuth, `WEBSITE_AUTH_ENABLED=true`), with
unauthenticated requests rejected and no direct path bypassing the platform,
or pass an `identity_provider` callback that validates bearer tokens (signature,
issuer, audience, expiry) and returns a stable principal ID. Never set the
EasyAuth environment flag to trust raw headers on a directly exposed/local server.
Production callers send their bearer token (and a Function key when FUNCTION
authorization is retained), not self-asserted `x-ms-client-principal-id` headers.
Ownership is preserved across continue-as-new generations; unknown ownership or
a failed lookup is denied.

`AuthLevel.ANONYMOUS` with EasyAuth or an `identity_provider` still enforces
ownership. Without either identity source, ANONYMOUS requires explicit
`require_instance_owner=False`; omission raises `ValueError`.
The sample's `respond_authorizer` requires a caller distinct from the starter
and present in the comma-separated `APPROVER_PRINCIPAL_IDS` environment setting
(stable, token-verified principal IDs, not display names). In production configure
this allowlist; an empty allowlist denies every response. The approver can respond,
but cannot read the owner's status route. The owner can share the returned
interrupt response URL with the approver. General-purpose apps without a callback
retain owner-only responses.

For the **loopback-only local demo**, explicitly opt in to the sample's insecure
identity stub before launching `sample.py`, `func start`, or `function_app.py`:

```bash
export LOCAL_DEMO_IDENTITY=1
```

```powershell
$env:LOCAL_DEMO_IDENTITY = "1"
```

The stub accepts `x-demo-user: demo-user` (starter/status reader) and
`x-demo-user: vp-finance` (approver). The starter cannot approve their own expense.
These headers are not authentication and
must never be enabled in a deployed app or exposed beyond loopback. The automated
demo sends this header; the manual commands below do too. To deliberately use
shared Function-key-only access instead of user isolation, set
`require_instance_owner=False`; Azure must validate the key, and all key holders
can access every instance. The standalone Python server does not validate keys.

### Mode 1: Automated Two-Worker Verification (`sample.py`)

Runs the complete two-worker crash/resume verification and both approval (`EXP-501`) and rejection (`EXP-502`) paths automatically:

#### From This Sample Directory (`samples/langgraph-durabletask/10_azure_functions_hitl/`)

**Bash:**
```bash
cd samples/langgraph-durabletask/10_azure_functions_hitl
export DTS_EMULATOR=127.0.0.1:8080
python sample.py
```

**PowerShell:**
```powershell
cd samples\langgraph-durabletask\10_azure_functions_hitl
$env:DTS_EMULATOR = "127.0.0.1:8080"
python sample.py
```

#### From the Repository Root

**Bash:**
```bash
export DTS_EMULATOR=127.0.0.1:8080
python samples/langgraph-durabletask/10_azure_functions_hitl/sample.py
```

**PowerShell:**
```powershell
$env:DTS_EMULATOR = "127.0.0.1:8080"
python samples\langgraph-durabletask\10_azure_functions_hitl\sample.py
```

---

### Mode 2: Interactive Multi-Terminal Run (`func start` or `function_app.py`)

You can also run the HTTP Function host and two separate worker processes interactively across terminals:

1. **Terminal 1 - Start `worker-1`:**
   ```bash
   export DTS_EMULATOR=127.0.0.1:8080
   python samples/langgraph-durabletask/10_azure_functions_hitl/function_app.py --worker worker-1
   ```
2. **Terminal 2 - Start `worker-2`:**
   ```bash
   export DTS_EMULATOR=127.0.0.1:8080
   python samples/langgraph-durabletask/10_azure_functions_hitl/function_app.py --worker worker-2
   ```
3. **Terminal 3 - Start the Azure Functions HTTP Host:**
   ```bash
   cd samples/langgraph-durabletask/10_azure_functions_hitl
   cp local.settings.json.example local.settings.json
   # Using Azure Functions Core Tools:
   func start
   # Or directly with Python (no func CLI required):
   START_EMBEDDED_WORKER=0 DTS_EMULATOR=127.0.0.1:8080 python function_app.py --no-embedded-worker --port 7071
   ```
4. **Terminal 4 - Start, Inspect, and Approve an Expense via HTTP:**

   **Bash (`curl`):**
   ```bash
   # Start workflow
   curl -s -X POST http://localhost:7071/api/graphs/expense_approval/start \
     -H "x-demo-user: demo-user" \
     -H "Content-Type: application/json" \
     -d '{"expense_id": "EXP-501", "employee": "Alice", "amount": 1850.0, "category": "conference"}'

   # Check status & pendingInterrupts
   curl -s -H "x-demo-user: demo-user" http://localhost:7071/api/graphs/expense_approval/status/<instanceId> | python -m json.tool

   # Submit human approval
   curl -s -X POST http://localhost:7071/api/graphs/expense_approval/respond/<instanceId>/<interruptId> \
     -H "x-demo-user: vp-finance" \
     -H "Content-Type: application/json" \
     -d '{"approved": true, "notes": "Approved for Q4 travel"}'
   ```

   **PowerShell (`Invoke-RestMethod`):**
   ```powershell
   $identity = @{"x-demo-user" = "demo-user"}
   $start = Invoke-RestMethod -Method Post -Uri "http://localhost:7071/api/graphs/expense_approval/start" `
     -Headers $identity `
     -ContentType "application/json" `
     -Body '{"expense_id": "EXP-501", "employee": "Alice", "amount": 1850.0, "category": "conference"}'

   $status = Invoke-RestMethod -Method Get -Uri $start.statusQueryGetUri -Headers $identity
   $interruptId = $status.pendingInterrupts[0].interruptId

   Invoke-RestMethod -Method Post `
     -Uri "http://localhost:7071/api/graphs/expense_approval/respond/$($start.instanceId)/$interruptId" `
     -Headers @{"x-demo-user" = "vp-finance"} `
     -ContentType "application/json" `
     -Body '{"approved": true, "notes": "Approved for Q4 travel"}'
   ```

## Code Walkthrough

For **single-process local development only**, set `START_EMBEDDED_WORKER=1`
before `func start`, or run `python function_app.py --embedded-worker`.
Do not enable this setting on Consumption/Flex. `sample.py` always disables the
embedded worker because it intentionally demonstrates two dedicated processes.

### 1. Expose Any Durable Graph as an Azure Functions v2 App (`function_app.py`)

```python
reg = attach_durability(
    build_expense_graph(_default_worker_id), name="expense_approval"
)
app = create_function_app(
    reg,
    start_worker=False,
    respond_authorizer=_authorize_expense_response,
    **_identity_options(),
)
```

`create_function_app(reg)` registers the HTTP routes (`/api/graphs/expense_approval/start`, `/status/{instance_id}`, `/respond/{instance_id}`, `/respond/{instance_id}/{interrupt_id}`, and `/api/health`) and optionally manages the embedded DTS worker lifecycle alongside the Function host.

### 2. Pause at `interrupt()` Inside `manager_review` and Route on the Human Response

```python
def manager_review(state: ExpenseState) -> dict[str, Any]:
    human_decision = interrupt(
        {
            "expense_id": state["expense_id"],
            "employee": state["employee"],
            "amount": state["amount"],
            "policy_status": state.get("policy_status", "UNKNOWN"),
            "prompt": "Approve or reject this expense?",
        }
    )
    approved = bool(human_decision.get("approved", False))
    reviewer = str(human_decision["responder"])  # Stamped by the HTTP handler.
    notes = str(human_decision.get("notes") or "")
    trace_entry = f"manager_review:worker={worker_id}:pid={os.getpid()}"
    return {
        "approved": approved,
        "reviewer": reviewer,
        "review_notes": notes,
        "worker_trace": [trace_entry],
    }
```

## Expected Output

```text
[1/7] Azure Functions HTTP host listening at http://127.0.0.1:.../api (health=healthy)
[2/7] Spawned Worker 1 (worker-1) -> OS PID=18420
[3/7] EXP-501 (...) paused at interrupt ... on worker-1 (PID=18420): payload={'expense_id': 'EXP-501', 'employee': 'Alice', 'amount': 1850.0, 'policy_status': 'REQUIRES_VP_APPROVAL', 'prompt': 'Approve or reject this expense?'}
[4/7] Spawned Worker 2 (worker-2, PID=24912) and HARD-KILLED Worker 1 (PID=18420) while EXP-501 is waiting for human input!
[5/7] POSTed human approval to Azure Function http://127.0.0.1:.../api/graphs/expense_approval/respond/.../... -> {'instanceId': '...', 'graph': 'expense_approval', 'resumed': True, 'interruptId': '...', 'pendingInterrupts': []}
[6/7] EXP-501 completed on Worker 2 (PID=24912)! final_status='APPROVED_AND_PAID', payout_reference='PAY-EXP-501-$1850.00'
      worker_trace = ['submit_expense:worker=worker-1:pid=18420', 'policy_check:worker=worker-1:pid=18420', 'manager_review:worker=worker-2:pid=24912', 'process_payout:worker=worker-2:pid=24912']
[7/7] EXP-502 rejection path verified via POST /respond/...: final_status='REJECTED', review_notes='Missing itemized receipt'

Verified Azure Functions HITL across 2 workers: worker-1 (PID 18420) -> [HTTP /respond resume] -> worker-2 (PID 24912).
```
