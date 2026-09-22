# What this sample demonstrates

A LangGraph agent that reads image and file attachments sent in a Responses
request. The host preserves `input_image` and `input_file` parts, and the
`ChatOpenAI` model forwards them using `use_responses_api=True`.

Send attachments with `role: "user"` as in this sample. The host also preserves
attachments in assistant-role graph input, but `langchain-openai` 1.4.1 drops
assistant `input_image` / `input_file` blocks when constructing its downstream
Responses request. That provider limitation requires an upstream serialization
fix; changing the message role would change conversation semantics.

The model reads the attachment directly. No server-side filesystem tools,
`DATA_DIR`, or bundled server files are required.

## Running the agent

Follow the [local setup instructions](../../README.md#running-the-agent-host-locally).
Use a model deployment that supports the attachment type and the Responses API.
For development before an SDK release containing this fix is available, install
the shared [editable requirements](../../requirements.txt) from this checkout; an
older published hosting package may discard attachments.

From this sample directory:

```bash
python main.py
```

## Send the example PDF

In another terminal, from the same sample directory:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  --data-binary @request.json
```

PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8088/responses -Method Post `
  -ContentType application/json -InFile request.json
```

[`request.json`](request.json) contains a small, one-page inline PDF. Its text
includes the verification code `ORCHID-4827`; the prompt asks the model to read
the code without providing it. The answer should contain that code. This tests
the attachment path, rather than asking a filesystem tool to read a server file.
The [manual E2E runner](../../tests/run_samples_e2e.py) checks the same request
against a configured Foundry model. Add `"stream": true` to the JSON body to
receive streaming events.

## Use your own attachment

Replace the file part in `request.json` with an appropriate Responses content
block, keeping it alongside the text prompt:

```json
{
  "type": "input_image",
  "image_url": "https://example.com/your-image.png",
  "detail": "auto"
}
```

The URL is a placeholder: use an image accessible to your model service, or a
data URI. Files can use `file_url`, `file_id`, or inline `file_data` with a
filename. Choose formats and sizes supported by the deployed model.

A file ID must be accessible to the downstream model service and identity.
Uploading a file to a hosted-agent session does not automatically make its ID
a valid model file ID. This sample does not upload, download, or grant access to
files. Request attachments are also not automatically mounted as local paths.

## Deploying the agent

Follow the [deployment instructions](../../README.md#deploying-the-agent-to-foundry)
after installing a published SDK version containing attachment preservation.
The same request content is used locally and when invoking the deployed agent.
