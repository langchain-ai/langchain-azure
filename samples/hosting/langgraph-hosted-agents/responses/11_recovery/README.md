# Server-side recovery Responses sample

This sample demonstrates the server-side recovery shape for a long-running
background response. `resilient_background=True` lets AgentServer recover the
in-progress task after a container restart without requiring the client to keep
polling or resubmit the request.

The graph writes a tiny external watermark before the optional demo crash. On
re-entry it observes the watermark and completes instead of repeating the first
phase. In production, use a durable LangGraph checkpointer or your own durable
store for real workflow state.

## Run locally

```bash
pip install -r requirements.txt
python main.py
```

Submit a background response:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Recover this job if the server restarts.","background":true,"store":true}'
```

To exercise the crash path in a hosted/container run, set:

```bash
DEMO_CRASH_ONCE=1
```

The first process exits after writing the watermark. The recovered process
re-enters the task, sees the watermark, and completes the response.
