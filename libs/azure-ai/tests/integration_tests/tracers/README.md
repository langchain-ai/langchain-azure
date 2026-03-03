## Tracer Integration Test Matrix

These tests exercise the `AzureAIOpenTelemetryTracer` against real LangGraph graphs to ensure the most important telemetry flows keep working as the tracer gains new capabilities.

| Test | Scenario | Key Coverage |
| --- | --- | --- |
| `test_basic_agent_tracing_records_spans` | Single-node agent driven by `FakeMessagesListChatModel`. | Verifies agent span creation, conversation IDs, and assistant messages from synthetic invocations. |
| `test_agent_with_tool_records_tool_span` | Agent that leverages a LangGraph `ToolNode`. | Checks tool span creation and attachment of tool call metadata. |
| `test_langgraph_agent_loop_records_spans` | Multi-node graph with MemorySaver checkpointing. | Ensures span parentage survives loops/middleware nodes and conversation metadata propagates. |
| `test_custom_graph_tracer_compatibility` | Dataclass state + command-like returns (LangGraph `goto`). | Validates per-node message key overrides, command duck-typing, and `metadata.langgraph.goto` recording. |
| `test_negative_agent_tracer_records` | Real `ChatOpenAI` call (VCR-captured). | Exercises the true network flow to ensure span shapes stay compatible with OpenAI responses (requires `OPENAI_API_KEY`). |
| `test_trace_all_nodes_records_unlabeled_graph` | Graph without per-node metadata while tracer runs in `trace_all_langgraph_nodes` mode. | Confirms every LangGraph node gets its own span and that global `message_paths` are honoured. |
| `test_metadata_message_path_records_wrapped_state` | Metadata uses `otel_messages_path` to drill into nested dataclass state. | Shows that node-level overrides work even when messages are wrapped deep in custom state. |
| `test_static_retriever_records_results` | Direct invocation of a `BaseRetriever` subclass with tracer callbacks. | Validates retriever spans (`execute_tool`) capture queries and serialized documents. |
| `test_tool_error_span_records_status` | ToolNode executes a tool that raises an exception. | Ensures `execute_tool` spans record `error.type` and emit error statuses when tools fail. |

### Recording Guidance

* The only test that talks to a live model is `test_negative_agent_tracer_records`. Re-record it (and any similar additions) with `OPENAI_API_KEY` exported and `pytest --record-mode=all`.
* All other tests rely on local fakes, so they run offline and do not need credentials.

### Adding New Use-Cases

When you add new tracer behaviours (e.g., streaming deltas, retriever spans, Azure Monitor plumbing), pair them with:

1. **Unit tests** in `test_inference_tracing.py` to cover the pure-Python helpers/branching.
2. **An e2e test** in this directory that shows the intended LangGraph usage and guards against regressions in callback ordering.
3. **Cassette updates** (only if the scenario truly requires calling an external service).

Keeping both levels of coverage aligned is how we maintain the ~85 % unit coverage target for the tracer module while still validating the full LangGraph surface.
