# `langchain-azure-compute[durabletask]` (`langchain_azure_compute.graph`) Runnable Examples

All examples in this directory are self-contained and use deterministic, offline models so they run without external LLM API keys.

## Prerequisites

1. Install the package:
   ```bash
   pip install -e "libs/azure-compute[durabletask]"
   ```

2. Start the local Azure Durable Task Scheduler (DTS) emulator:
   ```bash
   docker run -d --name dts-emulator -p 8080:8080 -p 8082:8082 mcr.microsoft.com/dts/dts-emulator:latest
   ```

3. Set `DTS_EMULATOR`:
   ```bash
   export DTS_EMULATOR=127.0.0.1:8080
   # PowerShell: $env:DTS_EMULATOR = "127.0.0.1:8080"
   ```

> If `DTS_EMULATOR` is not set, each script prints instructions for starting the emulator and exits cleanly.

## Examples Catalog

Each sample is organized in a **self-contained sample folder** (`samples/langgraph-durabletask/<sample>/` with step-by-step `README.md`, `requirements.txt`, and `sample.py`):

| Sample Guide & Folder | Entry Point | What It Demonstrates |
| --- | --- | --- |
| [`01_hello_graph/`](01_hello_graph/README.md) | [`01_hello_graph/sample.py`](01_hello_graph/sample.py) | Minimal linear & conditional `StateGraph` compiled in Zero-Config mode (`builder.compile()`) with `input_schema` and `output_schema` projection. |
| [`02_vacation_booking/`](02_vacation_booking/README.md) | [`02_vacation_booking/sample.py`](02_vacation_booking/sample.py) | Vacation Planning graph (`plan_trip -> ToolNode([book_flight, book_hotel, book_car]) -> review -> confirm`) showing Scenario 1 (transient tool failure on `book_car` during `ToolNode` proving per-tool retry isolation vs. `SqliteSaver` superstep replay) and Scenario 2 (idempotent client retry via `client.start(..., instance_id=...)`). |
| [`03_human_in_the_loop/`](03_human_in_the_loop/README.md) | [`03_human_in_the_loop/sample.py`](03_human_in_the_loop/sample.py) | Human-in-the-loop approval using `interrupt()`, `run.wait_pending()`, `run.pending_interrupts()`, and `run.resume()`. |
| [`04_react_agent_and_subgraphs/`](04_react_agent_and_subgraphs/README.md) | [`04_react_agent_and_subgraphs/sample.py`](04_react_agent_and_subgraphs/sample.py) | Running `langgraph.prebuilt.create_react_agent` and nested compiled subgraphs with `@durable_tool` activities. |
| [`05_multi_turn_conversation/`](05_multi_turn_conversation/README.md) | [`05_multi_turn_conversation/sample.py`](05_multi_turn_conversation/sample.py) | Multi-turn stateful conversation across separate runs sharing a `thread_id` with `SqliteSaver` checkpointer, `SqliteStore` memory, and `config` / `Runtime[Context]` propagation. |
| [`06_durable_nodes_map_reduce/`](06_durable_nodes_map_reduce/README.md) | [`06_durable_nodes_map_reduce/sample.py`](06_durable_nodes_map_reduce/sample.py) | Static parallel fan-out (`START -> [branch_a, branch_b, branch_c] -> join`) and dynamic `Send("research_worker", item)` map-reduce fan-out using `@durable_node` and `@durable_node` with per-branch retry isolation and `node_max_concurrency`. |
| [`07_multi_graph_app/`](07_multi_graph_app/README.md) | [`07_multi_graph_app/sample.py`](07_multi_graph_app/sample.py) | Multiple durable graphs (`name="triage"`, `name="vacation"`, `name="refund"`) registered on a single `DurableTaskSchedulerWorker` and `DurableTaskSchedulerClient` sharing one `SqliteSaver`, running concurrently AND delegating from one graph to another. |
| [`08_graph_versioning_and_migration/`](08_graph_versioning_and_migration/README.md) | [`08_graph_versioning_and_migration/sample.py`](08_graph_versioning_and_migration/sample.py) | Graph versioning & live checkpoint migration (`version="v2"`, `previous_versions={"v1": graph_v1}`, `migrations={("v1", "v2"): ...}`, `upgrade_on_continue_as_new=True`, and explicit `run.upgrade("v2", ...)`). |
| [`09_two_process_crash_recovery/`](09_two_process_crash_recovery/README.md) | [`09_two_process_crash_recovery/sample.py`](09_two_process_crash_recovery/sample.py) | Two separate OS worker processes (`pod-A` and `pod-B`) running a Zero-Config `InMemorySaver` graph (`builder.compile()`) across 3 concurrent instances (`ORD-9001`, `ORD-9002`, `ORD-9003`), where `pod-A` is hard-killed (`proc_a.kill()`) mid-run and `pod-B` seamlessly continues by hydrating its `InMemorySaver` from the DTS `checkpoint_bundle`. |
| [`10_azure_functions_hitl/`](10_azure_functions_hitl/README.md) | [`10_azure_functions_hitl/sample.py`](10_azure_functions_hitl/sample.py) & [`10_azure_functions_hitl/function_app.py`](10_azure_functions_hitl/function_app.py) | Complete Azure Functions Python v2 project (`function_app.py`, `host.json`) exposing HTTP routes (`POST /api/graphs/expense_approval/start`, `GET /api/graphs/expense_approval/status/{instance_id}`, `POST /api/graphs/expense_approval/respond/{instance_id}/{interrupt_id}`) for an Expense Approval graph with `@durable_tool` and `interrupt()`, verified across **two separate worker processes (`worker-1` and `worker-2`)** where `worker-1` is killed at the interrupt and `worker-2` resumes after the human HTTP response. |
