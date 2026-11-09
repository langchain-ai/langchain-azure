# 05 - Multi-Turn Stateful Conversation (`SqliteSaver`, `SqliteStore` & `Runtime[Context]`)

This sample demonstrates how to run **multi-turn stateful conversations** across separate durable runs that share a `thread_id`, combining:
1. **`SqliteSaver`** for pointer-only checkpoint persistence across turns (`inline_checkpoints=False`),
2. **`SqliteStore`** (`BaseStore`) for cross-thread / per-tenant semantic memory (`runtime.store`), and
3. **`RunnableConfig`** + **`Runtime[AppContext]`** propagation from `client.start(..., config=..., context=...)` into graph nodes.

## Key Concepts Demonstrated

Memory facts are keyed by each input `HumanMessage.id`, which is preserved in
the checkpoint. Retrying a step overwrites the same fact instead of inflating
the fact count or `Turn #`. Supply a stable, distinct message ID for every
logical turn, including turns with identical text.

- **Shared `thread_id` Across Separate Runs**: Starting Turn 1 and Turn 2 as two distinct DTS orchestrations that share `thread_id="conversation-user-42"`, automatically accumulating conversation history (`messages`) in `SqliteSaver`.
- **Pointer-Only Checkpoint Mode**: When a graph is compiled with a persistent checkpointer (`builder.compile(checkpointer=saver, store=store)`), the DTS orchestration history carries only lightweight checkpoint pointers while full state lives in SQLite.
- **`BaseStore` (`SqliteStore`) Long-Term Memory**: Reading and writing structured user facts (`runtime.store.get` / `runtime.store.put`) namespaced by `("memories", tenant_id, user_id)`.
- **`Runtime[AppContext]` & `RunnableConfig` Forwarding**: Passing a dataclass `context=AppContext(tenant_id="acme-corp")` and `config={"configurable": {"channel": "mobile"}, "tags": ["vip"]}` to `client.start(...)` and reading both inside the node.

## Workflow Architecture

```text
Turn 1 (client.start(..., thread_id="conversation-user-42", config=cfg, context=ctx)):
  START ──► assistant_node ──► END
              ├── Reads/writes SqliteSaver checkpoint for thread "conversation-user-42" (2 messages)
              └── Appends Fact #1 ("I prefer window seats.") to SqliteStore ("memories", "acme-corp", "user-42")

Turn 2 (client.start(..., thread_id="conversation-user-42", config=cfg, context=ctx)):
  START ──► assistant_node ──► END
              ├── Resumes from Turn 1 SqliteSaver checkpoint (now 4 messages: Human1, AI1, Human2, AI2)
              └── Appends Fact #2 ("I also like boutique hotels.") to SqliteStore
```

## Prerequisites & Environment Setup

### 1. Python & Dependencies

Requires **Python 3.11+**.

From the repository root:

```bash
pip install -e "libs/azure-compute[durabletask]" -r samples/langgraph-durabletask/05_multi_turn_conversation/requirements.txt
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

> **DTS Dashboard:** Open `http://localhost:8082` to see both `langgraph.multi_turn_chat.run` orchestrations execute against the shared thread.

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

### From This Sample Directory (`samples/langgraph-durabletask/05_multi_turn_conversation/`)

**Bash:**
```bash
cd samples/langgraph-durabletask/05_multi_turn_conversation
export DTS_EMULATOR=127.0.0.1:8080
python sample.py
```

**PowerShell:**
```powershell
cd samples\langgraph-durabletask\05_multi_turn_conversation
$env:DTS_EMULATOR = "127.0.0.1:8080"
python sample.py
```

### From the Repository Root

**Bash:**
```bash
export DTS_EMULATOR=127.0.0.1:8080
python samples/langgraph-durabletask/05_multi_turn_conversation/sample.py
```

**PowerShell:**
```powershell
$env:DTS_EMULATOR = "127.0.0.1:8080"
python samples\langgraph-durabletask\05_multi_turn_conversation\sample.py
```

## Code Walkthrough

### 1. Define `AppContext` and the `assistant_node` Receiving `RunnableConfig` and `Runtime[AppContext]`

```python
@dataclass
class AppContext:
    tenant_id: str


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str


def assistant_node(state: ChatState, config: RunnableConfig, runtime: Runtime[AppContext]) -> dict:
    uid = state["user_id"]
    tenant = runtime.context.tenant_id if runtime.context else "default"
    channel = (config.get("configurable") or {}).get("channel", "web")
    last_user_msg = state["messages"][-1].content
    ns = ("memories", tenant, uid)

    existing = runtime.store.get(ns, "history")
    facts = list(existing.value.get("facts", [])) if existing and existing.value else []
    facts.append(str(last_user_msg))
    runtime.store.put(ns, "history", {"facts": facts, "channel": channel})

    reply = f"[{tenant}/{channel}] Turn #{len(facts)} recorded for {uid}: remembered {len(facts)} fact(s)."
    return {"messages": [AIMessage(content=reply)]}
```

### 2. Compile with `SqliteSaver`, `SqliteStore`, and `context_schema=AppContext`

```python
saver = SqliteSaver(cp_conn)
store = SqliteStore(store_conn)
store.setup()

builder = StateGraph(ChatState, context_schema=AppContext)
builder.add_node("assistant", assistant_node)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)
graph = builder.compile(checkpointer=saver, store=store)
```

### 3. Run Turn 1 and Turn 2 on the Same `thread_id`

```python
shared_thread = "conversation-user-42"
ctx = AppContext(tenant_id="acme-corp")
cfg = {"configurable": {"channel": "mobile"}, "tags": ["vip"]}

# Turn 1
run1 = client.start(
    {"messages": [HumanMessage(content="I prefer window seats.")], "user_id": "user-42"},
    thread_id=shared_thread,
    config=cfg,
    context=ctx,
)
out1 = run1.wait_result(timeout=30)

# Turn 2 on the same shared_thread
run2 = client.start(
    {"messages": [HumanMessage(content="I also like boutique hotels.")], "user_id": "user-42"},
    thread_id=shared_thread,
    config=cfg,
    context=ctx,
)
out2 = run2.wait_result(timeout=30)
```

## Expected Output

```text
Turn 1 reply: [acme-corp/mobile] Turn #1 recorded for user-42: remembered 1 fact(s).
Turn 2 reply: [acme-corp/mobile] Turn #2 recorded for user-42: remembered 2 fact(s).
Total messages in thread: 4
SqliteStore facts: {'facts': ['I prefer window seats.', 'I also like boutique hotels.'], 'channel': 'mobile'}
```
