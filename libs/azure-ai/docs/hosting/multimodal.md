# Rich content in the Responses host

`ResponsesHostServer` preserves the supported content below rather than extracting
only its text. This applies to local JSON and SSE responses, Responses transcript
history, and LangGraph checkpoint state. It does not change the Invocations
request/response format.

## Supported content

| Path | Supported representation |
|---|---|
| User/system/developer input and tool-result history | Strings; `input_text`; `input_image` with `image_url` (external URL or data URI) or `file_id`; `input_file` with `file_url`, `file_id`, or `file_data` |
| Assistant history | Text, native citation annotations, log probabilities, and `refusal` parts |
| Tool output | Strings; lists of text and the native image/file parts above; LangChain v1 `image`/`file` blocks with `url`, `base64` plus `mime_type`, or `file_id`; OpenAI `image_url` and nested `file` blocks |
| Assistant output | Strings; `text`/`output_text` parts, native Responses citation annotations and log probabilities, and `refusal` parts |

Part order, data URI contents (including MIME type), filenames, references, and
image/file detail are retained where present in the protocol. The host does not
fetch URLs, upload files, decode inline documents into text, or invent filenames.
Each image/file block must specify exactly one source. Unsupported content types,
unrepresentable fields, and malformed references raise `ValueError` rather than
returning an empty or partial successful answer.

Native annotations include `url_citation`, `file_citation`,
`container_file_citation`, and `file_path`. A LangChain `citation` containing only
the representable `url`, `title`, `start_index`, and `end_index` fields is converted
to `url_citation`; other citation shapes are rejected, not silently truncated.

Rich request parts become LangChain content lists: text becomes `type="text"`,
while image/file parts retain their Responses-native `input_image`/`input_file`
shape. Text-only request lists still become strings for compatibility. Use an
OpenAI-compatible LangChain model configured with `use_responses_api=True` to
forward these native parts. Other providers need their own adapter; preservation
in graph state does not imply every model accepts the same block format. File IDs
must be accessible to the downstream model's service and identity.

## Example input and tool output

Send this as the JSON body of `POST /responses`:

```json
{
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Compare this image and document."},
        {
          "type": "input_image",
          "image_url": "https://example.com/chart.png",
          "detail": "high"
        },
        {
          "type": "input_file",
          "file_id": "file-accessible-to-your-model",
          "filename": "report.pdf"
        }
      ]
    }
  ],
  "stream": true
}
```

Previously, the graph received only `"Compare this image and document."`.
It now receives a `HumanMessage` with all three parts in order.

A tool can return public rich content explicitly:

```python
from langchain_core.messages import ToolMessage

result = ToolMessage(
    tool_call_id="call-from-the-assistant",
    content=[
        {"type": "text", "text": "Generated report"},
        {
            "type": "file",
            "file_id": "file-accessible-to-your-model",
            "filename": "report.pdf",
        },
    ],
)
```

The Responses `function_call_output.output` is a list containing `input_text`
and `input_file`, not `"Generated report"` alone. Replaying that output through
Responses history restores a rich `ToolMessage` with the same call ID.

## Streaming and output middleware

Plain string assistant chunks keep the existing text-delta behavior. Structured
content lists are buffered to the message boundary so indexed fragments merge
once, in order, without emitting incomplete or duplicate parts. Plain typed text
and refusal parts then emit their usual content events.

The `azure-ai-agentserver-responses` 2.1.0b2 text/message builders discard
annotations and log probabilities in their `done` payloads. To preserve those
fields, annotated/log-probability messages use the SDK's public generic item
builder and emit complete `response.output_item.added`/`done` events. They do not
produce per-text-part deltas. Clients must consume `output_item.done` or
`response.completed`, not rely exclusively on `output_text.delta`. Full
annotation-preserving token streaming requires upstream SDK builders to retain
metadata in both content-part and output-item completion events.

Buffering structured content to message boundaries is not an output-middleware
approval boundary. This change retains the host's existing publication policy:
model content can be published before output middleware finishes.
The separate [middleware output policy proposal](https://github.com/langchain-ai/langchain-azure/pull/1026)
introduces final-only publication and is not included here. Integrating that
policy must pass the approved final `AIMessage` through the rich-content emitter
rather than flatten it to text. This version does not claim final-only guardrail
protection. Content conversion does not sanitize image/file bytes, checkpoints,
logging, or tracing; middleware must be appropriate for the modalities used.

## Deliberate first-version limits

`ToolMessage.artifact` is separate application data, not model content. Responses
has no independent arbitrary-artifact field on a function-call output. Exporting
a tool message with a non-`None` artifact therefore fails explicitly. Keep private
artifacts in application storage and consciously put only public, model-visible
text/image/file references in `ToolMessage.content`. A true separate
artifact channel requires an agreed protocol/schema extension and SDK support;
the host does not insert invented wire fields or stringify arbitrary objects.

Assistant image/audio/video outputs, image-generation tool items, arbitrary
provider-specific content, and LangChain v0 data-block formats are not implemented
by this adapter. An assistant image is not relabeled as a user input or a fictitious
tool result. Existing reasoning-summary and explicit `AIMessage.tool_calls`
handling remain separate from assistant content conversion.

Verification uses real SDK serialization, local Starlette JSON/SSE requests,
LangChain Responses request construction, and both history mechanisms. It does
not establish deployed client-to-Foundry interoperability, downstream model
multimodal support, file authorization, or live-service size limits.
