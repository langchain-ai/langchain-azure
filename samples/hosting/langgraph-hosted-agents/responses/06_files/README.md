# What this sample demonstrates

A LangGraph agent that reads image and file attachments sent in a Responses
request. The host preserves `input_image` and `input_file` parts, and the
`ChatOpenAI` model forwards them using `use_responses_api=True`.

Send your question and attachments in a `user` message, as shown in `request.json`.

The model reads the request attachment directly.

## How it works

### Model integration

`DefaultAzureCredential` supplies Azure credentials, `AIProjectClient` provides
an OpenAI-compatible endpoint for the configured project. `ChatOpenAI` calls the
deployed model using the Responses API. Choose a multimodal deployment supporting the Responses API
and PDF input for `request.json`.

### Agent hosting

`ResponsesHostServer` exposes the LangGraph agent at `/responses` and converts
request content into LangChain messages while preserving the attachment parts.

## Running the agent

Follow the [local setup instructions](../../README.md#running-the-agent-host-locally).
Use a multimodal deployment supporting the Responses API and PDF input.
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

[`request.json`](request.json) contains an actual one-page PDF encoded as Base64
in `file_data` of an `input_file` block. Pasting the Base64 string into an ordinary
text-only Playground field does not test this attachment path. The PDF text
includes the verification code `ORCHID-4827`; the prompt asks the model to read
the code without providing it. The answer should contain that code.
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
