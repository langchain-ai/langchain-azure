# What this sample demonstrates

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent that
**reads files at runtime** through local Python tools, hosted using the
**Responses protocol**. The agent has two tools:

- `list_files(subpath="")` — list the contents of a directory below the
  configured data root.
- `read_text_file(file_path)` — read a UTF-8 text file from the data
  root (capped at 64 KiB).

A small starter file at [`data/notes.txt`](data/notes.txt) ships with
the sample so you can exercise the agent without any extra setup.

## Relationship to the Agent Framework `06_files` sample

The upstream Agent Framework sample uploads files into a hosted-agent
session and references them by `file_id` in the Responses input. Today
the langchain Responses host only forwards **text** content blocks (see
the hosting converter at
[`libs/azure-ai/langchain_azure_ai/agents/hosting/_converters/_request.py`](../../../../libs/azure-ai/langchain_azure_ai/agents/hosting/_converters/_request.py))
— `input_file` / `input_image` items are dropped. Until the hosting
layer learns to passthrough those items, this sample exposes files to
the agent via the filesystem inside the container rather than as
request attachments. The user-facing pattern is the same: "the agent
can read files at runtime"; the wiring is different.

## How It Works

### Model Integration

The agent uses `langchain_azure_ai.chat_models.AzureAIOpenAIApiChatModel`
with the Foundry project endpoint and `DefaultAzureCredential`.

See [main.py](main.py) for the full implementation.

### Filesystem Tools

Both tools resolve paths under `DATA_DIR` (defaults to `./data` next to
`main.py`) and reject any path that escapes the root. `read_text_file`
caps the response at 64 KiB so the agent never accidentally pulls a
multi-megabyte file into a single tool message.

To make new files available to the agent, drop them into `DATA_DIR`
before starting the host. For container deployments, ship them in the
image (or mount them via a Foundry-managed volume) and point `DATA_DIR`
at that location.

### Agent Hosting

The agent is hosted using
[`langchain_azure_ai.agents.hosting.ResponsesHostServer`](../../../../libs/azure-ai/langchain_azure_ai/agents/hosting),
which adapts the compiled LangGraph runnable into a REST endpoint
compatible with the OpenAI Responses protocol.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the
README in the parent directory.

## Interacting with the agent

> Depending on how you run the agent host, you can invoke the agent
> using `curl` (`Invoke-WebRequest` in PowerShell) or `azd`. Please
> refer to the [parent README](../../README.md) for more details.

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "List the files available to you, then summarize notes.txt."}'
```

```bash
curl -N -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Read notes.txt and tell me what action items it mentions.", "stream": true}'
```

Intermediate `function_call` / `function_call_output` items are
surfaced for every file the agent inspects — same shape as the
local-tools sample, but the tool reads are scoped to the data root.

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory. The default `Dockerfile` copies the
sample's `data/` directory into the image so `notes.txt` is reachable
without extra wiring. Replace the contents of `data/` with your own
files before `azd deploy`, or set `DATA_DIR` to a mounted volume path.
