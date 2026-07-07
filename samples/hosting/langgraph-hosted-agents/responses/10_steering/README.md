# Steering Responses sample

This sample hosts a checkpointed LangGraph workflow with
`steerable_conversations=True`. A newer background turn for the same
conversation is queued behind the active turn, and the active graph run receives
a cooperative cancellation signal through
`config["configurable"]["foundry_cancellation_signal"]`.

## Run locally

```bash
pip install -r requirements.txt
python main.py
```

Start a long turn:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"conversation":{"id":"steering-demo"},"input":"Research topic A.","background":true,"store":true}'
```

Before it completes, steer the conversation:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"conversation":{"id":"steering-demo"},"input":"Actually switch to topic B.","background":true,"store":true}'
```
