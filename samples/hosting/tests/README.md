# Hosted sample tracing checks

From the repository root, in a Python 3.11–3.14 virtual environment:

```bash
python -m pip install -r samples/hosting/tests/requirements.txt
python -m pytest samples/hosting/tests/test_tracing.py -q
```

The test discovers hosted samples throughout `samples/`, including directories
outside `samples/hosting/`. It detects imports of the SDK's host servers and
configuration-runner commands in Dockerfiles and YAML files. Every discovered
sample must have a scenario in `tracing_scenarios.py`; missing or stale scenarios
fail the coverage check. Tests never skip an unregistered sample.

Each sample runs in a fresh process using its own startup code, graph, tracing
setup, and HTTP server. Configuration-driven samples use the SDK runner. Each
scenario checks three configurations:

- Azure Monitor export to a local ingestion endpoint.
- OTLP/HTTP export to a local collector.
- `OTEL_SDK_DISABLED=true`, with no exported GenAI spans.

Assertions cover model spans, tool names and spans for tool-using scenarios,
incoming W3C trace correlation, and parent IDs. Approval scenarios also execute
the approved tool and final model call after resuming. No-tool samples check
model and agent spans. Full sampling is enabled only in the test process.

Remote chat models and MCP/Toolbox services are substituted. Cloud checkpoint
stores use an in-memory saver; resilient samples retain their local SQLite
checkpointer. This tests tracing through real sample graphs, host servers, and
exporters. It does not validate remote services, cloud persistence, Azure
ingestion, Foundry project connections, or portal visibility. The shared test
environment uses the SDK checkout and the middleware/Deep Agents runtime pins;
it does not install every sample's isolated dependency set.

The `hosted-sample-tracing` workflow runs these checks on Python 3.11–3.14 for
sample and SDK changes. It requires no Azure credentials.

## Adding a hosted sample

Add its repository path relative to `samples/`, protocol, and representative
scenario to `CASES` in `tracing_scenarios.py`. Reuse a scenario when the model
decisions and tool arguments match; otherwise add deterministic model responses.
For new external services or startup conventions, extend the test launcher at
the external boundary. Keep the sample's graph and tracing setup unchanged.

Run the coverage check and the new sample before the full suite:

```bash
python -m pytest samples/hosting/tests/test_tracing.py -q -k 'scenario or discovery'
python -m pytest samples/hosting/tests/test_tracing.py -q -k '<sample-directory>'
```

Samples hosted through a new SDK or a different entrypoint convention need a
discovery rule as well as a scenario. The discovery self-test uses future sample
locations to protect the current rules.
