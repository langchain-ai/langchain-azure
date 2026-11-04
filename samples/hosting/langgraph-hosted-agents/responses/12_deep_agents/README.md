# Deep Research on Foundry

The official LangChain [Deep Research example](https://github.com/langchain-ai/deepagents/tree/b6504f6654ee6f25e1cd923087291de9e51572cc/examples/deep_research),
adapted to Microsoft Foundry's **Responses protocol**.

## Research, revise, download

1. Plan and delegate web research to `research-agent` using Foundry Toolbox
   `web_search`, then save the cited report as `/report-v1.md`.
2. Read the saved report, create `/report-v2.md`, and edit that copy to add
   recovery limitations while keeping the original findings and citations.
3. Publish each named version with `publish_report(report_name="report-v1.md")`
   or `publish_report(report_name="report-v2.md")` and download both snapshots.
   The original virtual file and its earlier download remain unchanged.

`StateBackend` keeps working files in conversation state;
`FoundryCheckpointSaver` persists them locally or in Foundry State Store.
`publish_report` reads only the selected saved file, not model-supplied upload
text. Names are simple ASCII Markdown filenames, not paths. Blob authorization
and network failures return handled tool errors: the named file remains editable,
but no download link is returned. Unexpected programming errors still surface.

> [!NOTE]
> Deep Agents' `DeltaChannel` can make checkpoint restoration **significantly slower**
> with Foundry State Store, especially as the number of conversation turns grows.
> [agent.py](agent.py) stores full messages/files to avoid delta-history reads,
> at the cost of larger checkpoints.
> This performance workaround uses a private reducer in the pinned Deep Agents version.
> Another performance workaround is [Deep Agents 0.5.9](https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.5.9),
> which predates `DeltaChannel`. For that older API, remove `FullCheckpointState`,
> its supporting imports, and the `state_schema` argument.
> `CosmosDBSaver` bulk-loads delta history and does not need this performance workaround.
> If you change the schema or Deep Agents version, test restoring existing checkpoints.

## Setup and run

Complete [SETUP.md](SETUP.md) first. **Local execution also requires a cloud
Toolbox exposing `web_search` and separate Blob permissions** for the local
credential. Deployed agents need their own Blob grant; the developer's access
does not transfer to the agent.

From this directory, install Python 3.11+ dependencies, copy
[.env.example](.env.example) to `.env`, and fill in the values from setup:

```bash
python -m pip install -r requirements.txt
python main.py
```

In another terminal, run the directly executable [client.py](client.py):

```bash
python client.py
```

For an existing deployed agent configured by [SETUP.md](SETUP.md#deploying-the-agent-to-foundry):

```bash
python client.py --agent-name <deployed-agent-name>
```

Expect full cited reports, a recovery-limitations revision, and downloaded
`report-v1.md` and `report-v2.md` under `~/Downloads/deep-research` (override with
`--output-dir`). The client chains the latest `response.id`, binds deployed
continuations using `agent_session_id`, verifies download digests, and stops the
returned test sessions on exit. If publication fails, it prints the explanation
and response ID, exits without claiming a download, and leaves the saved working
report available for a later turn.

Download links are read-only HTTPS bearer secrets valid for one hour; expiry
**does not delete** the Blob. Keep transcripts and URLs private. Both telemetry
content flags default to `false`; see [privacy and tracing](SETUP.md#privacy-and-tracing).
Search still sends queries to the search service, and checkpoints retain content.

## Validation

```bash
python -m unittest discover -s tests
```

These offline checks simulate model/Blob boundaries; they do not establish Azure
RBAC or real download success. See [real JSON/SSE acceptance and cleanup](SETUP.md#acceptance-and-cleanup)
for the full model/search/export workflow. Source code remains split between
[agent.py](agent.py), [research prompts/tools](research_agent/), the small
[publication tool](report_export.py), and [host startup](main.py).
