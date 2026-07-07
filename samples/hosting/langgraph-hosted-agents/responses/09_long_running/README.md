# Long-running Responses sample

This sample hosts the same slogan workflow as the regular workflow sample: a
writer drafts a slogan, a legal reviewer checks it, and a formatter produces the
final response. `ResponsesHostServer(..., resilient_background=True)` enables
background Responses requests to run inside the AgentServer Task API envelope, so
callers can poll while the model work is still running.

Use it to validate the basic long-running path:

- `POST /responses` with `background=true` returns quickly.
- `GET /responses/{response_id}` can poll until completion.
- While the response is active, the Task API lease keeps the hosted container
  warm.

## Run locally

```bash
pip install -r requirements.txt
python main.py
```

Submit a background response:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Create a launch slogan for a solar-powered backpack.","background":true,"store":true}'
```

Poll the returned response id:

```bash
curl http://127.0.0.1:8088/responses/RESPONSE_ID
```
