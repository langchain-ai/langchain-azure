# LangGraph Time Travel in Hosting

Reviewed 2026-09-24 against the current working tree using primary sources,
18 focused host tests, and deterministic local HTTP and graph probes.

Design revised 2026-09-28 to use the OpenAI Responses API as the public
contract, especially for `previous_response_id`. Implementation resumed against
that contract on the same date.

The latest full hosting verification passed 719 tests, including 231 branching
cases. Read [Implementation Handoff](#implementation-handoff) for the tested
runtime combinations, SDK compatibility gaps, and remaining release gates.

## Conclusion

A compatible checkpointed LangGraph can time travel behind either host, but
neither default HTTP adapter exposes checkpoint history, arbitrary checkpoint
selection, state editing, or an explicit replay/fork operation. Hosting an
arbitrary `Runnable` with Invocations does not itself add LangGraph capabilities.
Both hosts expose the original graph: [Responses](../../langchain_azure_ai/agents/hosting/_responses_host.py#L400) and [Invocations](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L531).

The narrower proposal to select a parent through `previous_response_id` is
feasible without a new HTTP field or an upstream SDK change in non-steerable
mode. See [Responses Parent Selection Feasibility](#responses-parent-selection-feasibility)
for the contract, local proof, and remaining production requirements.
The [proposed branching design](#proposed-responses-branching-design) records the
subsequently discussed interface and compatibility decisions. The opt-in
implementation now supports exact completed-response selection; passing local
tests does not establish full OpenAI conformance or production readiness.

## Implementation Handoff

### Current Status (2026-09-30)

2026-09-24: the user requested a progress checkpoint before committing, pushing,
and continuing on another machine. Implementation work stopped at that request;
the atomic admission/approval changes discussed immediately beforehand were
not implemented at that time. Work resumed on 2026-09-28 and added those changes
and the public-contract corrections below. The historical failing-test record
is preserved separately; it is not the current result.

At the pause, all eight changed Python source/test files were already staged by
the user. This documentation update is a subsequent working-tree change; the
assistant did not stage, commit, push, or create a branch.

| Work item | Current state |
| --- | --- |
| OpenAI response-linkage contract | Invalid/mixed linkage and body `response_id` are rejected before SDK execution. Request-local instructions, immediate-parent identity, and safe `server_error` mapping are covered through JSON, SSE, and retrieval. |
| Instructions with summarization | Default message mode now tracks explicit instruction deletions, allowing continuation after standard summarization or trimming without relaxing identity checks. Opt-in `instructions_mode="context"` plus model integration keeps temporary instructions out of persistent messages and summarizer input. |
| Opt-in interface and exact completed-parent selection | Implemented, default off. Missing/non-saver values and inherited unimplemented async read/write methods fail during construction, without storage I/O. |
| Strict checkpoint reads and parent-linked recovery | Exact reads, origin/progress validation, failed-root termination, and mode preservation are covered. SQLite saver and SDK local-store recreation preserves branches; actual process-crash windows remain unverified. |
| Ordinary HITL and historical approvals | Normal/waiting, sequential, and parallel partial approvals work, including user-scoped metadata. Atomic pause ownership rejects a second historical answer and its waiting aliases. Distributed competing-approval tests remain a gate. |
| Storage and execution ownership | Explicit platform response IDs have pre-SDK atomic admission after provider lookup. Verified pre-admission SDK rejections release only the matching owner and ETag; accepted or uncertain attempts retain ownership. Generated IDs are claimed before graph execution. Foreground `store=false` does not publish a reusable boundary. |
| Quality and compatibility | Latest full run: 719 hosting tests passed, including 231 branching cases. All 231 branching cases also passed on Python 3.11 with minimum direct hosting dependencies. Earlier Python 3.12/3.13 runs covered 172 cases, before the default-mode removal fix. Python 3.14 is the full-suite environment. This is not every supported dependency combination or a cloud matrix. |

### API Alignment Decision (2026-09-28)

The public request/response format and behavior must follow the OpenAI
[create reference][openai-create]. Retain headers required by the Foundry
platform; OpenAI-specific authentication, organization, project, and similar
platform headers are not compatibility requirements. This is not permission
to remove Foundry authentication or to expose new LangGraph control headers.

The revised [compatibility scope](#compatibility-scope) supersedes the earlier
decision to preserve mixed `conversation` / `previous_response_id` requests
and exclude `instructions` from this work. Exact parent selection remains the
goal; preserving an SDK behavior that contradicts the public contract is not
evidence of OpenAI compatibility. The default-off graph feature remains a
rollout constraint, with its legacy limitations stated explicitly below.

These corrections are implemented and locally verified. Existing callers must
choose one non-null linkage field, resend persistent top-level instructions each
turn, and stop sending a body `response_id`. Foundry platform identity headers
remain supported. The sections below distinguish current results from the
historical pause record.

### Request Instruction Isolation (2026-09-30)

Real `SummarizationMiddleware` and persistent `trim_messages` reproduced a
compatibility failure: legitimate history replacement removes a host instruction
message while its checkpoint source marker remains. The next request then fails
identity verification. This affects both branching modes; preserving reducer
metadata alone does not prevent intentional message removal.

The default-mode follow-up fixes that continuation failure at the saver boundary.
The request-local adapter observes `RemoveMessage` writes by thread, namespace,
checkpoint, and task. It clears `langchain_response_instructions_source_v1` only
when the previous instruction identity was verified, explicit deletion explains
its absence, and the updates do not retain or reintroduce that identity. Later
checkpoints carry the verified absence. Restored `pending_writes` provide deletion
evidence after adapter recreation. Parent checkpoints and message contents are
not rewritten; missing IDs or tags without valid deletion evidence still fail.

Both branching settings use the same `ResponseCheckpointSaver`. Its internal
`branching` argument follows the request's existing mode: default reads preserve
the wrapped saver's behavior, while branching validates exact checkpoints. Both
modes propagate backend errors and track instruction writes. No new public host
option is exposed. Shallow graph copies preserve instance overrides and leave
the original graph-owned saver unchanged. Stateless graphs are not wrapped.

This does not isolate instructions from the graph's message processing. A
summarizer may consume them before the main model call or include their text in a
summary. Use context mode when instructions must stay outside summarizer input;
the default-mode continuation fix does not require enabling context mode.

`instructions_mode="messages"` remains the default, with unchanged graph inputs
and strict provenance validation. Opt-in `instructions_mode="context"` passes
instructions in transient runnable configuration instead. Applications must add
`ResponsesInstructionsMiddleware` to `create_agent`, or use
`get_response_instructions(config)` in custom model nodes. The middleware combines
the current instructions with the application system message only for the model
call, preserving system-message fields and content blocks. It does not replace
the application's runtime context or rewrite compiled graph nodes.

Context mode keeps the raw instructions out of graph state, checkpoint metadata,
and summarizer input. Model outputs may still reflect them; this is request
isolation, not semantic erasure. Omitting, nulling, or emptying instructions on a
new turn does not inherit the previous value. A new HITL approval uses its own
instructions; recovery of the same admitted task uses that task's instructions.

The host stamps the instruction mode into a server-owned persisted header and
removes spoofed client values. Recovery uses that admitted mode even if the host
default changes; absent headers on old tasks mean message mode. Both modes use
the existing checkpoint provenance format. Completed checkpoints can continue in
either mode, removing only verified old host instructions in the child's state.
Old checkpoints whose instruction identity was already lost before deletion
tracking, reformatted messages, and otherwise unverifiable legacy instructions
still fail closed and require a new root; this fix does not backfill old records.

Deploy the graph integration to all workers before enabling context mode. Keep
that integration while context-mode tasks may still recover, even if the default
is changed back to message mode. Drain context-mode tasks before rolling back to
binaries that predate this feature;
arbitrary old/new worker mixtures are not supported by the new option. See the
[sample instructions](../../../../samples/hosting/langgraph-hosted-agents/responses/01_basic/README.md#request-instructions-with-summarization).

### Uniform Read Failures (2026-09-24)

The user questioned the added `strict` argument and asked for uniformly strict
handling. The implemented decision is to remove that argument entirely:

- `detect_pending_interrupts(graph, config)` now propagates state-read exceptions unchanged for all callers, including legacy Responses and Invocations.
- No `aget_state` method, an explicitly absent/disabled saver, and a valid new thread without saved state remain normal empty-interrupt cases.
- There is no optional permissive error mode. Do not reintroduce the `strict=False` parameter.
- This is a deliberate exception to the earlier blanket "old behavior unchanged" requirement. It does not add time travel to Invocations or apply the new response-boundary storage protocol to legacy requests.
- The request-scoped `ResponseCheckpointSaver` rejects a missing or mismatched explicit checkpoint when the request uses branching. Its default-mode reads preserve the original saver contract; both modes propagate actual read exceptions.

### Implemented Files

| File | Implemented changes |
| --- | --- |
| [Responses host](../../langchain_azure_ai/agents/hosting/_responses_host.py) | Adds default-off exact parent selection with constructor validation, explicit public provider injection, request-local instruction provenance, strict saver copies, safe error mapping, ownership checks, conditional boundary publication, and failed-root termination. Recovery follows confirmed origin/progress and the admitted mode, not the current flag. Steering and injected `app` remain unsupported for the opt-in path. |
| [Request instructions](../../langchain_azure_ai/agents/hosting/_response_instructions.py) | Adds a transient custom-node accessor and sync/async model middleware for opt-in context mode, plus checkpoint-scoped tracking of explicit host-instruction removals. Application prompts and message contents are not mutated. |
| [Branching helpers](../../langchain_azure_ai/agents/hosting/_responses/branching.py) | Adds public linkage validation, trusted admission headers, atomic response/pause ownership, confirmed origins, compact completed-boundary metadata, and strict actual saver reads. Parent response status/metadata and full boundary index must agree. A single `ResponseCheckpointSaver` tracks instruction writes and gates exact-read validation on the existing branching mode, without owning saver lifecycle or retention. |
| [HITL converter](../../langchain_azure_ai/agents/hosting/_converters/_hitl.py) | Removes broad state-read exception swallowing and the temporary `strict` argument, while explicitly accepting no-saver graphs. |
| [Invocations host](../../langchain_azure_ai/agents/hosting/_invoke_host.py) | Moves the post-stream interrupt lookup inside the existing SSE exception handler, so read failure emits `error`, not a broken stream or a false `done`. Pre-stream failures already use the SDK's safe 500 response. |
| [Branching tests](../../tests/unit_tests/agents/hosting/test_response_branching.py) | 231 cases cover the existing contracts plus pre-admission retries, rollback uncertainty and ETag races, default-mode summarization/trimming, explicit deletion versus identity loss, saver recreation and evidence scoping, interrupted-save recovery, instruction-mode migration, concurrent isolation, sync/async read policies and middleware, HITL, and history-only instruction handling. |
| [Responses tests](../../tests/unit_tests/agents/hosting/test_responses_host.py) | Adds early saver-validation tests and a legacy state-read failure test proving graph execution stops. |
| [Hosting fixtures](../../tests/unit_tests/agents/hosting/conftest.py) | Adds atomic create/conditional-delete semantics, changing ETags, and snapshot fields required for instruction provenance; dedicated tests restore the real local Foundry state-store implementation. |
| [Invocations tests](../../tests/unit_tests/agents/hosting/test_invoke_host.py) | Adds pre-stream and post-stream read-failure tests. Uses `responses.store._memory.InMemoryResponseProvider`, which exists in the tested SDK version. |
| [HITL tests](../../tests/unit_tests/agents/hosting/hitl/test_converters.py) | Covers normal stateless/new-thread cases and unchanged propagation of timeout, permission, and invalid-data exceptions. |

No dependency, lockfile, public endpoint, saver backend, or chain-store protocol
change has been made. An opt-in usage section was added to the
[basic Responses sample](../../../../samples/hosting/langgraph-hosted-agents/responses/01_basic/README.md#opt-in-checkpoint-branches).
Constructor checks reject inherited unimplemented async methods; they cannot
certify an arbitrary custom saver's history retention. Exact runtime reads remain
mandatory, and production applications must select a history-preserving saver.

### Persisted State in the WIP

The implementation currently uses these internal names:

| Name | Value / meaning |
| --- | --- |
| Server-owned mode header | `x-client-langchain-response-branching: checkpoint-v1` |
| Server-owned execution token header | `x-client-langchain-response-owner`; new opaque token per admitted create request |
| Server-owned instruction-mode header | `x-client-langchain-instructions-mode: messages` or `context`; retained for same-task recovery |
| Response internal mode metadata | `langgraph_response_branching` |
| Confirmed origin key | `langgraph_branch_origin_v1` |
| Completed boundary key | `langgraph_response_boundary_v1` |
| Full record fields | `version="1"`, `checkpoint_ns=""`, `thread_id`, `checkpoint_id`, `paused="true"/"false"`, `pause_id`; origins additionally contain `mode` and `parent_response_id`. |
| Instruction provenance | `langchain_response_instructions_v1` in checkpoint metadata and host-injected message `additional_kwargs`; only proven host instructions are removed from a child's state. |
| Instruction source | `langchain_response_instructions_source_v1`; the active response ID, or an empty string after verified explicit deletion or when no host instructions were supplied. |

Origin/index identities use the existing user-scoped response-ID helper, not a
mutable conversation-head key. SDK `2.1.0b2` forwards only `x-client-*` headers to
`ResponseContext.client_headers`; the original `x-langchain-*` marker was dropped
and has been replaced. Middleware removes incoming values of all current
internal headers even when the feature is disabled, and stamps trusted values.
These are internal transport markers, not OpenAI fields or client controls.
SDK durable-task recovery preserves them; invalid branching modes and owner
mismatches fail closed in the tested recovery path. An absent instruction-mode
header means legacy message mode; unknown instruction modes fail closed.

Atomic ownership uses exported `FoundryStateStore.create_item` with conflict
handling, independently of the chain store. Its namespace hashes the project,
agent ID/name, and agent version; response keys also include the caller's user
partition. Before explicit platform-ID admission, the middleware establishes
the public Foundry request context for state access and passes `PlatformContext`
to the response provider. No private provider accessor is used on the new path.

The provider existence check now precedes the atomic claim, so a lookup failure
does not consume a new response ID. The claim still precedes SDK dispatch to
arbitrate concurrent creates. A completed SDK `400 invalid_request_error` or
`404 not_found_error` can release that claim only when the registered handler
has not started and a fresh provider lookup confirms that no response exists.
The request-local handler marker is shared with child tasks. Rollback checks the
owner and uses `delete_item(..., if_match=etag)`; a changed owner or version is
never deleted. The rejection is sent after the rollback attempt finishes.

Accepted/background response records, handler entry, unknown or incomplete
errors, HTTP 500, application exceptions, and uncertain provider/storage results
retain ownership. Cleanup errors are logged without exposing backend details.
This is narrowly scoped rollback of unadmitted claims, not permission to replay
failed roots or delete SDK task, response, or graph checkpoint records. It does
not repair old leaked claims or reclaim claims after a process crash.

Ownership records are non-expiring tombstones (`item_ttl_seconds=-1`) containing
only a schema version and owner token. They can outlive failed or unstored
responses; they neither retain graph state nor authorize failed-root replay.
This is additional internal storage, separate from existing response, chain, and
saver retention. Apart from the pre-admission rollback above, no automatic
cleanup or compaction is implemented; operational
retention and deployment/isolation behavior require review before release.

Pause ownership includes the exact checkpoint and a fixed-size pause epoch.
Waiting responses share their origin's epoch. A new pause after an actual
partial/sequential approval gets a new epoch even if LangGraph reuses the
checkpoint ID, so subsequent ordinary approvals are not blocked by the prior
claim. Historical waiting aliases retain the consumed epoch and fail.

The response's completed envelope is the boundary authority; the independent
index alone is insufficient. The SDK limits serialized internal response
metadata to 512 characters. The boundary marker therefore reuses the exact
`thread_id` and `checkpoint_id` already captured in that same envelope rather
than duplicating them. Publication verifies the reference matches run progress;
lookup reconstructs the full record and compares it with the index. User-scoped
normal and paused responses are covered. Atomic execution claims protect writes;
the chain store's `get`/`set` operations are not themselves compare-and-set.

### Current Verification

| Check | Result |
| --- | --- |
| Latest full `tests/unit_tests/agents/hosting` run on Python 3.14 with frozen dependencies | 719 passed, including 231 branching cases; includes admission retry/rollback, saver consolidation, the default-mode instruction-removal fix, and unchanged legacy hosting tests. |
| Latest branching suite on Python 3.11.16 / LangChain 1.2.12 / LangGraph 1.1.1 / prebuilt 1.0.8 / Agent Server Core and Responses 2.1.0b2 / Invocations 1.1.0b1 | 231 passed, including owner/ETag races against the real SDK local store. LangChain Core 1.6.6 and SQLite saver 3.1.1 were pinned; transitive dependencies are not all at their minimum versions. |
| Earlier branching suite before saver consolidation on Python 3.11 / LangChain 1.2.12 / LangGraph 1.1.1 / prebuilt 1.0.8 / Agent Server Core and Responses 2.1.0b2 | 202 passed. LangChain Core 1.6.6 and SQLite saver 3.1.1 were pinned; transitive dependencies are not all at their minimum versions. Not rerun for the class consolidation. |
| Earlier full hosting run before default-mode instruction-removal tracking | 660 passed, including 172 branching cases. |
| Earlier branching suite on Python 3.12.12 and 3.13.2 with frozen dependencies | 172 passed on each runtime; not rerun for the latest removal-tracking changes. |
| Earlier full frozen-environment hosting run | 528 passed, including 134 branching cases. |
| Earlier Invocations module run after the SSE error-redaction fix | 68 passed. |
| Earlier full hosting run on Python 3.14 / LangGraph 1.2.11 | 458 passed, including 64 branching cases; no skipped branching tests. |
| Earlier branching run on Python 3.11 / LangGraph 1.1.1 / prebuilt 1.0.8 | 64 passed, including SQLite/provider recreation. |
| Ruff lint and formatting of tests and changed runtime files | Passed. |
| Mypy of all tests and the two runtime modules changed for admission rollback | Passed, 98 source files. The earlier removal-tracking check covered 99 files. |

The frozen Python 3.12-3.14 runs used LangChain 1.3.15, LangGraph 1.2.12,
Agent Server Core 2.2.0, and Responses 2.2.0b2. The minimum-direct-dependency
run used langchain-core 1.6.6 and SQLite checkpointer 3.1.1. The full hosting
suite ran on Python 3.14; the other runtime checks targeted the branching suite.

The 30 new cases cover unchanged-default HTTP continuation, instruction clearing
and replacement, real summarization/trimming, ID-targeted deletion, retained
instructions and lossy reducers, both saver APIs, pending-write evidence scoping,
and interrupted-save recovery. A native LangGraph control confirmed that recovery
from an explicitly selected pre-commit checkpoint may rerun its node; these tests
do not establish exactly-once side effects. Failed branching roots remain terminal.

Saver consolidation adds four sync/async read-policy cases: new threads remain
valid in both modes, while an explicitly missing checkpoint is rejected only for
branching. Existing default-mode recovery and branch-isolation tests are unchanged.

The two earlier runtime combinations used Agent Server Core/Responses `2.1.0b2`
and Invocations `1.1.0b1`. Their SQLite checks used `langgraph-checkpoint-sqlite` `3.1.1`.
The persistence test closes the saver connection and reconstructs the graph,
host, response provider, and local state stores; it proves A,C and A,B,D after
recreation, not crash recovery during execution or distributed failover.

An unpinned lower-version overlay resolved prebuilt `1.0.13` with LangGraph
`1.1.1`; test collection then failed because prebuilt imports `ExecutionInfo`,
which that LangGraph version does not export. Pinning prebuilt `1.0.8` fixes the
verification environment. Repository dependency constraints were not changed;
the declared transitive combinations still need compatibility review.

Editor diagnostics continue to report unresolved imports in the test module,
while uv runtime imports and the all-tests mypy check succeed. The editor's
selected interpreter was not changed; do not describe its Problems panel as
clean or substitute runtime tests for a full static-check matrix.

### Historical Verification at the 2026-09-24 Pause

These were separate focused runs at successive edit points, not a final combined
suite run. Their failures were fixed in the resumed implementation. Do not add
these counts together as a current passing-suite total.

| Last observed check | Result |
| --- | --- |
| `test_constructor_rejects_branching_without_saver` | 3 passed. |
| Initial strict-saver and foreground branch cases | 5 passed: deletion after preflight fails before nodes; graph copy preserves the original saver; JSON/SSE support A -> B, A -> C, branch continuation and regeneration. |
| HITL converter module plus the then-current 5 branching cases, after removing `strict` | 63 passed. |
| `checkpoint_read_failure` selection in Responses/Invocations host modules | 4 passed. |
| `recovery or interrupted_root` in branching tests | 10 passed, including enabled/disabled current flags, missing/malformed progress, no parent-provider reread, and no root replay/defer loop. |
| `approval or duplicate_response` in branching tests | 3 failed at the pause; fixed by atomic response and pause ownership. |
| Last command before the pause, selecting only `duplicate_response` | 1 failed then: actual node executions were `['A', 'B']`, expected `['A']`. |

Previously failing tests, now passing:

1. `test_normal_approval_and_waiting_preserved_but_second_answer_rejected[False]`.
2. `test_normal_approval_and_waiting_preserved_but_second_answer_rejected[True]`.
3. `test_duplicate_response_identity_never_runs_graph_twice`.

The first two successfully create a pause, re-emit it on ordinary input, and
approve it as Alice. At the pause, a later Bob answer against the earlier paused
response was incorrectly reported as completed. The current implementation logs
internal reason `unsupported_approval_branch` and returns terminal `server_error`.
The third sends `x-agent-response-id` equal to an existing response ID; the SDK
previously accepted it and executed B again. A 200 status alone was not treated as a failure:
the refined test permits idempotent replay but proves actual duplicate execution.

The original tests predate the API-alignment revision; their safety assertions
remain, while terminal errors now follow the published schema. Repeated use of
a **parent** ID is a valid fork, not a duplicate-response failure.

### Remaining Release Gates

1. Exercise real process termination/restart and SDK durable-task admission, including the first durable mode record, same-task instructions, interrupted roots, and parent-linked progress. Mocked recovery and replacement of local host objects do not cover all crash windows.
2. Inject terminal-response persistence failures and crashes between index and terminal writes. Confirm that failed publication cannot authorize a parent, overwrite a completed boundary, or rerun completed graph work merely to repair an index.
3. Validate overlapping branches and competing approvals across workers with the intended persistent saver and deployed Foundry stores. Cover full user/deployment isolation, permissions, existing namespace properties, and ownership tombstone retention/maintenance.
4. Resolve or explicitly ship with the upstream SDK storage limitation below. Do not silently rewrite `store`, introduce a temporary-retention layer, or claim OpenAI forbids the combination.
5. Extend the supported dependency/Python matrix, whole-package static checks, and background cancellation/reconnection cases. Python 3.12/3.13 and all allowed upstream combinations were not exercised here.

SDK `2.1.0b2` rejects `background=true, store=false` in request validation, with
`background=true requires store=true`. Current OpenAI documentation permits
temporary retention for this combination. The host does not override the SDK
or silently set `store=true`; this remains an explicit compatibility gap.

Useful verified SDK facts for continuation:

- In SDK `2.1.0b2`, `response_acceptor` is a steering-queue hook, not a general pre-admission hook. It cannot establish this feature's durable admission mode.
- [SDK identity resolution][sdk-request-parsing] accepts `x-agent-response-id`, then a body `response_id`, otherwise generates an ID. This is observed SDK behavior, not the OpenAI create contract. Preserve Foundry platform header handling; a body `response_id` is not part of the public OpenAI-facing API. Any retained platform-only identity route must be isolated and protected. The private helper is not an approved integration point.
- `ResponseProviderProtocol.get_response(id, context=...)` can raise `KeyError`; `FoundryStorageProvider` raises the exported `FoundryResourceNotFoundError` instead. Admission handles both. The new path receives the public provider explicitly and always passes platform context; the pre-existing legacy ancestry helper still uses its old private accessor.
- SDK terminal persistence may treat `ResponseAlreadyExistsError` as recovery and switch to update. Its intermediate checkpoint persistence logs errors without acknowledging success to the handler. Neither supplies the missing admission guard by itself.
- [FoundryStateStore](https://learn.microsoft.com/python/api/azure-ai-agentserver-core/azure.ai.agentserver.core.storage.foundrystatestore?view=azure-python) exposes `create_item(key, value)` with duplicate-key failure, `set_item(..., if_match=etag)`, and `delete_item(..., if_match=etag)`. `FoundryStorageConflictError` and `FoundryStoragePreconditionError` are exported. Atomic claims and conditional rollback are tested against the real SDK local backend on Core 2.1.0b2 and 2.2.0, not a live Foundry service.
- LangGraph `1.1.1` and `1.2.11` support the tested `graph.copy({"checkpointer": adapter})` integration without mutation of the shared graph.

The additional ownership namespace is an explicit implementation change to the
earlier estimate. Existing saver/response/index retention is unchanged; only the
new ownership records use non-expiring tombstones. Ordinary `get`/`set`,
in-process locks, or a preflight-only check cannot prove cross-process ownership.

### Reproduce on Another Machine

Run from the repository root in a uv-managed environment. The observed runtime
was Python 3.14, LangGraph `1.2.11`, Agent Server Core/Responses `2.1.0b2`, and
Invocations `1.1.0b1`. Do not run tests with global Python.

On a fresh machine, prepare the package environment using its project metadata
before using `--no-sync`, for example:

```powershell
uv sync --project libs/azure-ai --python 3.14 --extra hosting
```

The previous machine already had a package environment; its successful commands
used temporary `uv run --no-sync --with ...` overlays. A fresh dependency
resolution has not been verified in this session. A pinned overlay for resuming
the focused tests is:

```powershell
$testEnvironment = @(
"run", "--project", "libs/azure-ai", "--no-sync",
"--with", "langgraph==1.2.11",
"--with", "langgraph-checkpoint-sqlite==3.1.1",
"--with", "azure-ai-agentserver-core==2.1.0b2",
"--with", "azure-ai-agentserver-responses==2.1.0b2",
"--with", "azure-ai-agentserver-invocations==1.1.0b1",
"--with", "pytest", "--with", "pytest-asyncio",
"--with", "pytest-socket", "--with", "pytest-mock"
)
uv @testEnvironment python -m pytest `
libs/azure-ai/tests/unit_tests/agents/hosting/test_response_branching.py `
-q --show-capture=no --disable-warnings
```

The historical failures above are now fixed. To narrow the run, append
`-k 'approval or duplicate_response'` or `-k 'recovery or interrupted_root'`.
The branching test module already isolates `AGENTSERVER_STATE_ROOT` with
`tmp_path` and mocks SDK tracing setup. Other host-module runs used a temporary
state root set **before SDK imports** and patched
`azure.ai.agentserver.core._tracing._configure_tracing` in the verification
process. Preserve this isolation; do not disable production tracing or use live
Foundry/model services for these unit tests.

The full hosting run used the same overlay and the following isolated wrapper:

```powershell
& {
	$previousRoot = $env:AGENTSERVER_STATE_ROOT
	$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString("N"))
	$env:AGENTSERVER_STATE_ROOT = $testRoot
	try {
		uv @testEnvironment python -c "import pytest; from unittest.mock import patch; patch('azure.ai.agentserver.core._tracing._configure_tracing', lambda *args, **kwargs: None).start(); raise SystemExit(pytest.main(['libs/azure-ai/tests/unit_tests/agents/hosting', '-q', '--show-capture=no', '--disable-warnings', '--tb=short']))"
		$testExitCode = $LASTEXITCODE
	}
	finally {
		$env:AGENTSERVER_STATE_ROOT = $previousRoot
		if (Test-Path -LiteralPath $testRoot) {
			Remove-Item -LiteralPath $testRoot -Recurse -Force
		}
	}
	if ($testExitCode -ne 0) { throw "Hosting tests exited with code $testExitCode" }
}
```

For the minimum-runtime check, use a separate environment without changing the
project environment or lockfile:

```powershell
uv run --isolated --no-project --python 3.11 --with-editable ./libs/azure-ai --with langgraph==1.1.1 --with langgraph-prebuilt==1.0.8 --with langgraph-checkpoint-sqlite==3.1.1 --with azure-ai-agentserver-core==2.1.0b2 --with azure-ai-agentserver-responses==2.1.0b2 --with azure-ai-agentserver-invocations==1.1.0b1 --with pytest --with pytest-asyncio --with pytest-socket --with pytest-mock python -m pytest libs/azure-ai/tests/unit_tests/agents/hosting/test_response_branching.py -q --show-capture=no --disable-warnings --tb=short
```

These are local verification commands, not a claim that every accepted upstream
dependency resolution is compatible. Release still requires the remaining
gates above; no cloud service or model was called in these checks.

## LangGraph Contract

The official [time-travel guide](https://docs.langchain.com/oss/python/langgraph/use-time-travel), [persistence overview](https://docs.langchain.com/oss/python/langgraph/persistence), and [checkpointer contract](https://docs.langchain.com/oss/python/langgraph/checkpointers) establish:

- Compile with a history-preserving checkpointer; identify the correct `thread_id` and `checkpoint_ns`.
- `get_state_history(config)` / `aget_state_history(config)` enumerate snapshots newest-first. A snapshot's config identifies its exact `checkpoint_id`.
- Replay with `invoke(None, snapshot.config)` / `ainvoke(None, snapshot.config)`. Nodes after that checkpoint execute again; a final checkpoint with no next nodes is a no-op.
- Fork with `update_state(snapshot.config, values, as_node=...)` / `aupdate_state(...)`, then invoke with `None` and the returned config. This creates a new checkpoint, preserves the original history, and applies normal reducers; `as_node` controls successor selection when specified.
- LLM calls, tools, and other external calls after the checkpoint run again and may differ. Re-executed interrupts pause again and need a new `Command(resume=...)`.

## ResponsesHostServer

This is the pre-implementation baseline observed on 2026-09-24, also describing
the legacy graph-selection path. The opt-in WIP described in the handoff adds
exact parent selection; the bullets below are not a claim about that new path.

- [Config construction](../../langchain_azure_ai/agents/hosting/_responses_host.py#L613) reads `(user-scoped context.conversation_chain_id, "langgraph_checkpoint")`; it does not read a client-supplied `checkpoint_id` or the selected previous response's checkpoint metadata.
- [Turn completion](../../langchain_azure_ai/agents/hosting/_responses_host.py#L971) replaces that pointer. The [store contract](../../langchain_azure_ai/agents/hosting/_responses/conversation_chain_store.py#L52) explicitly provides replacement, not per-response versioning.
- If a pointer exists, its exact checkpoint is used. If absent, config contains only the resolved thread and response context; [thread resolution](../../langchain_azure_ai/agents/hosting/_responses_host.py#L689) follows response ancestry to a conversation ID or root response ID. LangGraph then selects that thread's latest checkpoint.
- Consequently, once a shared chain pointer advances, an accepted request with an earlier `previous_response_id` does not select that response's historic graph snapshot. When the SDK assigns a fresh chain key, the normal resolved-thread fallback likewise selects latest state, not that parent's snapshot. SDK admission rules can reject a request before either path.
- [Input conversion](../../langchain_azure_ai/agents/hosting/_responses_host.py#L469) sends only current input when a graph checkpointer exists. Without one, it prepends Responses history instead. Branching chat history through an old response ID is not replay of graph state, node progress, or interrupts.

### SDK Chain Identity

The [2.1.0b2 release implementation][sdk-chain-release] and [upstream implementation][sdk-chain-main] agree on these cases:

| Request / option | `conversation_chain_id` basis |
| --- | --- |
| Explicit conversation | Conversation partition plus agent/session scope |
| No conversation, steering enabled | Embedded partition of `previous_response_id` or initial `response_id`, plus agent/session scope |
| No conversation, steering disabled | Current `response_id` verbatim |

Steering defaults to `False` in the checked [release options][sdk-options].
The SDK chain key is not a parent-response checkpoint ID, nor is it universally
the literal root response ID. Shared embedded partitions identify steerable
chains; non-steerable requests have separate keys. The host's ancestry-derived
LangGraph thread ID is a separate concept. Thus the "one mutable pointer for
the whole chain" explanation applies to shared-key modes, not every SDK mode.

## Responses Parent Selection Feasibility

This section records the pre-implementation feasibility investigation, not
completion of the current WIP. Invocations time travel remains outside scope;
the later shared read-error change is recorded in the handoff above.

### External Contract

The OpenAI [migration guide][openai-migration] explicitly says that
`previous_response_id` can create "response chains or forks". The
[create reference][openai-create] makes it mutually exclusive with
`conversation` and states that previous top-level `instructions` are not carried
forward. The [conversation-state guide][openai-state] describes stored response
context and its retention limitations.

For a completed response A and a response B created with parent A:

| User operation | New request | Intended state |
| --- | --- | --- |
| Continue B | Parent B, new input D | A, B, D |
| Fork from A | Parent A, new input C | A, C, independent of B |
| Regenerate B | Parent A, resend B's input | A, new B attempt with a new response ID |
| Regenerate the first turn | No parent, resend its input | New independent root response |

Referencing B means continuing after B, not rerunning B. Regeneration creates
another response; it is not a promise of idempotent HTTP retry, identical model
output, or exactly-once tool effects. Crash recovery of B and SSE event replay
remain separate operations. A response ID selects a response boundary, not an
arbitrary node or super-step inside that response.

The parent is the specific response named in the request, not the newest
response in its thread. Creating B must not consume A or move A's boundary:
later and concurrent children may still select A. Repeating the same parent
and input in a fresh create request is another generation, not implicit
deduplication. Send only the additional input when the parent supplies history;
do not automatically append the full transcript again.

With no explicit conversation, omitting `previous_response_id` or setting it
to `null` starts an independent root. Empty strings, wrong types, or unresolved
IDs must not silently become roots. The child's returned `previous_response_id`
must identify its immediate selected parent, consistently in JSON, response
events, and later retrieval. Internal thread/chain IDs and inferred conversation
associations must not replace that public value or manufacture a public
`conversation` field for a response-ID chain.

### Useful OpenAI References

- [Create a response][openai-create]: linkage fields, instruction lifetime, storage, response shape, and `Response.error`.
- [Migration guide: multi-turn conversations][openai-migration]: explicitly describes response chains and forks and distinguishes manual history replay.
- [Conversation state][openai-state]: response context, Conversations API, and retention boundaries. WebSocket cache behavior is not an HTTP persistence guarantee.
- [Background mode][openai-background]: asynchronous creation, polling, cancellation, and stream reconnection.
- [Streaming Responses][openai-streaming]: typed SSE events, separate from graph replay or creating another response.
- [MCP tools and approval][openai-mcp] and [function calling][openai-function-calling]: approval responses and tool outputs as input items, with their own linkage IDs.

### Local Implementation Route

The required selection is:

```text
authorized previous_response_id
-> saved parent (thread_id, checkpoint_id)
-> graph execution with only the new request input
-> new checkpoints and a new response ID
```

No graph copy or additional public `checkpoint_id`, `retry`, or `fork` field was
needed for the completed, non-paused turns proved below. LangGraph can retain
multiple descendants in one thread when every invocation pins its parent
checkpoint. This result does not prove isolation of HITL resume state or strict
failure when the selected checkpoint disappears during restore.

An important existing capability makes this practical: without an explicit
conversation and with steering disabled, the SDK uses the **current response
ID** as its chain key [source][sdk-chain-release]. The host already
[stores each run's checkpoint under that key](../../langchain_azure_ai/agents/hosting/_responses_host.py#L971).
However, [config construction](../../langchain_azure_ai/agents/hosting/_responses_host.py#L613)
reads the incoming response's chain key instead of the selected parent's key.
Its fallback then loads latest state from the common LangGraph thread.

For this mode, looking up the user-scoped **parent response ID** in the existing
[chain store](../../langchain_azure_ai/agents/hosting/_responses/conversation_chain_store.py)
and passing its reference to
[`HostingRunnableConfig.create_from_checkpoint`](../../langchain_azure_ai/agents/hosting/_responses/hosting_runnable_config.py)
is sufficient for the successful-turn scenarios proved below. Formalizing an
immutable per-response mapping is preferable to assuming every SDK mode will
always use response IDs as chain keys. Keep any mutable conversation-head pointer
separate from that mapping.

An alternative source is the per-response metadata already written by
[`TaskStorageManager`](../../langchain_azure_ai/agents/hosting/_responses/task_storage_manager.py).
The SDK [provider protocol][sdk-provider] has tenant-context-aware `get_response`,
but [ResponseContext][sdk-context] does not expose a public parent-response
getter; the existing ancestry resolver reaches its private `_provider` field.
The design below requires the stored response as completion authority, not just
a host-owned mapping. Access to the effective provider through a supported
integration must therefore be verified before implementation.

### Local Proof

A disposable subclass outside the repository overrode only
`build_runnable_config`: on a fresh parent-linked request it read the parent's
existing store entry and returned an exact checkpoint config. The production
host, SDK, and dependencies were not modified. The graph stored a non-message
`ledger` channel, so these checks demonstrate graph-state branching, not just
filtered chat history. Concurrent requests synchronized inside the graph before
writing, ensuring overlapping execution.

| Probe | Observed result |
| --- | --- |
| Unchanged host: A, B after A, C after A | `A, B, C` |
| Subclass: same requests | `A, C` for C |
| Continue B after C exists | `A, B, D` |
| Continue C after D exists | `A, C, E` |
| Regenerate B from A | `A, B`, with a distinct response ID |
| Concurrent siblings from A | `A, parallel-left` and `A, parallel-right` |
| Parent mapping after all descendants | A's checkpoint reference unchanged |

All subclass checks passed through real local HTTP handling in three modes:
foreground JSON, foreground SSE, and background SSE with
`resilient_background=True`. All used `steerable_conversations=False`, stored
responses, and no explicit conversation. Foreground cases used the SDK's
in-memory response provider; the background case used its file-backed provider
in an isolated temporary state directory. All graph checkpoints and the custom
chain store were in memory. Runtime versions match the verification section
below. This proves execution feasibility, not cross-process recovery or cloud
storage correctness. The disposable script and state were removed afterward.

Follow-up design-review probes used isolated in-memory graphs on LangGraph
`1.2.11`, without changing production code:

| Probe | Observed result and design consequence |
| --- | --- |
| Check existence, delete the checkpoint, then invoke with its original config | LangGraph executed new input from empty state. A preflight lookup alone is insufficient. |
| Replay root input on the same thread after the saver advanced but without a response-side checkpoint | Additive state was duplicated. The agreed new-mode policy below terminates failed roots instead of replaying them. |
| Resume the same paused checkpoint with `approve`, then with `reject` | Both results contained `approve`. Ordinary HITL continuation and independent historical approval branches must not be treated as equivalent. |

The checked SDK [checkpoint persistence implementation][sdk-execution] logs
provider failures without raising them back into the handler. Yielding
`stream.checkpoint()` is consequently not an acknowledged write for newly
required origin metadata. These findings motivate the design; they are not
verification of its proposed fixes.

### SDK Constraint

With a task manager active, the SDK [primitive selector][sdk-orchestrator]
uses a one-shot task for non-steerable response chains without `conversation`.
It uses a multi-turn task for explicit conversations or steering, and passes
`previous_response_id` as `if_last_input_id` only on that multi-turn path. The
[execution orchestrator][sdk-execution] propagates failed head preconditions;
the endpoint translates them to `conversation_fork_not_supported`.

This task routing also applies to stored foreground requests, not only resilient
background requests. If the task subsystem is disabled, the SDK can fall back
to in-process execution, so disabling crash recovery alone is not a reliable
fork policy. The earlier steering probe observed HTTP 409 for an old parent.
Supporting historical forks **while retaining steering** needs a separate
upstream-compatible admission/task-identity design; a graph config override
cannot bypass rejection before the handler runs. Do not silently disable an
existing steering configuration.

### Production Requirements

The proof establishes successful-turn feasibility, not a production feature.
The following design addresses the discussed compatibility, persistence, and
failure requirements. Its acceptance checks remain implementation work, not
additional results from the local probe.

## Proposed Responses Branching Design

Status: initially recorded on 2026-09-24 and revised on 2026-09-28 for OpenAI
Responses compatibility; partially implemented as described in
[Implementation Handoff](#implementation-handoff). Preserve existing supported
behavior except for the agreed strict-read correction and the protocol
corrections below. Earlier preservation of nonstandard behavior does not
override the revised public contract.
Agreed behavior and proposed mechanisms are separated below. Integration points
that still need a prototype are listed as implementation gates, not as solved
capabilities.

### Host Interface

Add a keyword-only `enable_response_branching: bool = False` parameter to
`ResponsesHostServer`. It controls graph-state branching, not automatic saver
creation. Current opt-in usage:

```python
graph = builder.compile(checkpointer=saver)
server = ResponsesHostServer(graph, enable_response_branching=True)
```

The saver belongs to the compiled graph. Reuse it; do not require a second saver
argument on the host. Client requests keep using `previous_response_id` with
the existing Responses schema. No new endpoint or public `checkpoint_id`,
`retry`, or `fork` request field is needed.

| Configuration | Required behavior |
| --- | --- |
| Flag omitted or `False` | Retain legacy graph branching/storage behavior, not protocol violations. Shared strict reads and the public-contract corrections below still apply to new requests. Already admitted tasks retain their recorded recovery mode. |
| Flag `True`, graph has no usable saver | Raise `ValueError` in `__init__`, before creating the SDK host or registering its handler. |
| Flag `True`, graph has a history-preserving saver | Enable exact parent selection for eligible response-ID chains. |
| Flag `True` with steering enabled | Reject the unsupported configuration explicitly; never silently disable steering. |
| Explicit `conversation` without `previous_response_id` | Keep the existing conversation path, outside graph response-ID branching. |
| Both linkage fields non-null | Reject before execution; no conversation-priority or parent-priority fallback. |
| Root request on the new path fails or crashes | Do not automatically resume or replay it, even when background resilience is enabled. Successful roots still publish a boundary for later requests. |

Initialization validates local configuration only, without storage network
requests. It must also validate the graph when attaching to an existing `app`.
A disabled or inherited checkpointer marker is not itself a saver configured
for this root host. Keep new validation separate from legacy detection so
disabled-feature behavior does not change.

When attaching to an existing `app`, validate its effective steering settings,
not a host `options` argument that the attached app ignores. If those settings
cannot be established through a supported interface, fail the new opt-in
configuration explicitly rather than assuming steering is disabled. Existing
attachment behavior with branching disabled remains unchanged.

Suggested missing-saver error:

```text
enable_response_branching=True requires a graph compiled with a checkpoint saver.
Configure checkpointer when creating the graph.
```

A history-preserving `InMemorySaver` is sufficient for local, single-process
use. Production durability and cross-process recovery require persistent
storage, such as `FoundryCheckpointSaver`. A shallow/latest-only saver cannot
satisfy historical selection. Configuration checks do not guarantee that any
particular checkpoint still exists; availability is checked at restore time.
The presence of a saver object does not certify a custom implementation's
history support. State historical reads as a required saver contract, validate
the usable local interface, and enforce exact reads at runtime; do not probe
storage or infer capability solely from a class name during initialization.
Do not automatically supply an in-memory saver or downgrade to message-history
branching when this feature is requested.

### Compatibility Scope

- Use the [OpenAI create schema][openai-create] for public fields, response objects, and related behavior. No public `checkpoint_id`, `retry`, `fork`, caller-selected `response_id`, or branching header is added by this feature. Foundry-required headers remain platform integration details; internal admission markers are not a client API.
- Keep the graph feature flag off by default as previously agreed. Legacy requests do not gain new graph-boundary storage or exact historical checkpoint selection. A checkpointed host using that legacy path is not OpenAI-equivalent for historical branching. Changing the default requires an explicit rollout/migration decision. Recovery follows the recorded admission mode, not a later flag change.
- Reject a new HTTP request containing both non-null `conversation` and `previous_response_id` before execution, regardless of the graph feature flag. Validate the external fields before SDK ancestry resolution, not a derived internal conversation identity. Preserve valid conversation-only requests. Migration for mixed callers is to choose one linkage mechanism, not silently prioritize one field.
- Apply the current request's top-level `instructions`; do not inherit the parent's top-level instructions through response history or graph checkpoints. Omitting or nulling that field does not inherit the previous value. Do not remove explicit system/developer items from `input` history or application-owned graph prompts. Instruction handling is part of parent-selection compatibility, not unrelated cleanup.
- Preserve unrelated cancellation, SSE replay, and model configuration behavior. This change establishes the response-linkage contract; it does not implement every OpenAI-hosted tool or claim complete API conformance before verification.
- Do not change `resilient_background` or steering settings automatically. Background execution remains independently configured.
- Keep Invocations time-travel behavior, public override-hook signatures, and the existing chain-store interface unchanged. The later shared strict-error decision also requires Invocations to surface read failures through its existing error paths. Custom hooks that replace the default pipeline must honor the new contract when opting in.
- Failed-root termination applies only to requests admitted on the new path. Do not change legacy root recovery or disable the SDK's resilience configuration globally.
- Ordinary HITL continuation remains supported. Do not reject all paused checkpoints simply because branching is enabled; independent historical approval branches are a separate, out-of-scope capability.

The mutual-exclusion and instruction corrections also apply when branching is
disabled. Document these as behavior corrections with migration guidance:
mixed callers choose one linkage field, and callers needing stable top-level
instructions resend them each turn. Do not reinterpret already-admitted durable
tasks using new HTTP validation rules during recovery.

### Request Context and Identity

Top-level instructions are request context, not inherited conversation state.
Checkpoint restore must not reintroduce the parent's host-injected instruction
message through an additive messages reducer. Track the provenance of such
injections or use supported request-scoped context; any removal/replacement must
be branch-local and leave the parent's snapshot unchanged. Do not indiscriminately
drop system/developer messages: explicit `input` items and graph-configured
prompts are distinct. Recovery of the same admitted response retains that
response's original instructions. Verify what the graph receives, not just the
`instructions` field echoed on the wire.

For legacy checkpoints, prove the absence of inherited top-level instructions
or identify their host-injected representation through reliable provenance.
Matching message text or role alone is insufficient. If neither is possible,
reject that continuation before graph execution with a compatible error rather
than retaining old instructions or deleting user input. This applies to new
requests on both routes; disabling branching is not an instruction-migration
workaround. Do not mutate historical snapshots to retrofit provenance.

The [OpenAI create body][openai-create] has no top-level `response_id` for choosing
the new response's ID. The OpenAI-facing boundary must reject that nonstandard
field rather than use it as a retry or overwrite command. Foundry may supply
the new identity through its platform header; preserve that integration and
treat returned IDs as opaque, without requiring an OpenAI-specific prefix.
SDK support for an extension alone does not establish it as a required public
field. Document any separately required platform-only route explicitly.

Deduplication/ownership is keyed by the new response identity and trusted
isolation context, never by `previous_response_id`. A conflicting new admission
must not execute or overwrite an existing response. Re-entry of the same durable
task is recovery, not a second HTTP create or a new child. These controls do not
promise exactly-once external effects across recovery. Paused-graph continuation
needs its own ownership rule, as described under HITL.

### State Ownership

Keep these state roles distinct:

| Reference | Purpose | Update rule |
| --- | --- | --- |
| Existing conversation checkpoint pointer | Preserve legacy next-turn behavior | Keep the existing identity and semantics. |
| Confirmed origin record | Fix the starting point of a parent-linked request before graph execution | Write once for that response; identical re-entry of the same admitted task is idempotent, not fresh HTTP creates. |
| Per-response boundary checkpoint | Select a stable parent for a new response | The persisted completed response is authoritative; an independent index is only a lookup aid. |
| Current response execution checkpoint | Recover the same interrupted parent-linked task | Advances through the existing durable response-checkpoint mechanism; it does not authorize recovery of a failed new-mode root. |

Reuse `ConversationChainStoreProtocol.get/set` for origin records and boundary
indexes, with distinct versioned keys and trusted user/deployment-scoped
response identities.
Do not replace the existing `langgraph_checkpoint` record or assume every SDK
chain key is a response ID. The response store owns response status and recovery
metadata; the LangGraph saver owns actual graph state. Neither a response object
nor a boundary-reference dictionary substitutes for that state.

The minimal origin record contains a schema version, admitted mode, parent
response ID, and exact parent reference (`thread_id`, `checkpoint_ns`,
`checkpoint_id`; the root graph namespace is `""`). The current response is
identified by the scoped record key. Keep origin and execution progress separate;
an origin reference is not evidence that any node has run. Persist the admitted
route in the SDK-owned response/admission state as well, so recovery does not
infer it from a later flag value. Exact key names and version handling must be
fixed and tested before release; malformed or unknown new-mode records fail
explicitly rather than falling back to legacy execution.

### Boundary Publication

For responses retained by the supported response lifecycle, put the exact final
reference captured from the specific run into their internal metadata, including
successful retained roots. Never obtain it afterward through a latest-thread
lookup. The reference and `completed` status must be part of the same retained
response envelope; an independent index cannot declare completion. Internal
metadata must remain absent from client-facing output.

A foreground `store=false` response may complete successfully without a stored
envelope or reusable boundary. Do not force storage or create durable branch
publication records solely to make it a parent. Intentional non-retention is not
a storage-write failure. Existing saver writes and admitted-task recovery records
remain governed by their own contracts.

For temporarily retained background responses, permit parent selection only
while the authorized completed envelope, boundary, and checkpoint are actually
available. This is time-limited eligibility, not promotion to durable storage;
after expiry, the same ID must fail parent lookup even if an index or checkpoint
survives. Verify SDK support for this lifecycle before advertising it.

Proposed order for retained responses, within their permitted retention window:

1. Capture the run-specific final checkpoint reference and attach it to the response's internal metadata.
2. Await the boundary-index write. A write failure stops publication and surfaces an error; it must not publish a usable parent.
3. Emit the terminal response through the SDK, which owns response persistence and the final wire event. Do not write a competing terminal envelope directly from the handler.
4. A future parent lookup checks the authorized persisted response, its terminal status and boundary metadata, any required index consistency, and actual checkpoint availability.

| Publication state | Parent eligibility |
| --- | --- |
| Index exists, but the persisted response is absent, queued, in progress, incomplete, failed, or cancelled | Not eligible under this host's completed-boundary policy. The index alone is insufficient. |
| Persisted response is completed, references agree, and the checkpoint is readable | Eligible, subject to the HITL continuation rules below. |
| Successful foreground `store=false` response has no retained envelope | Execution may succeed, but the response is not an eligible stored parent. |
| Temporarily retained background response is completed and references/checkpoint remain available | Eligible only during its actual authorized availability window; no retention extension. |
| Index and completed response disagree | Fail explicitly; do not choose either value heuristically. |
| Completed response or a required reference/checkpoint becomes unavailable | Fail explicitly; do not reconstruct state from a mutable conversation head. |

The SDK `2.1.0b2` [terminal persistence path][sdk-execution] attempts to save the
response before returning the terminal event and maps storage failures to an
error. Unlike its intermediate `stream.checkpoint()` handling, this provides a
completion decision the new feature can consume. Preservation of the internal
boundary reference must be tested through foreground JSON, SSE, and background
paths, including subsequent retrieval and restart.

No distributed transaction between the index and response store is proposed.
An index written before a crash is harmless only while parent admission still
requires the authoritative completed response. Retry identical publication
idempotently; never replace an already committed response boundary with a
different checkpoint. Concurrent branches have different response identities.
Publication for one response must have a single owner across recovery; verify
the supported SDK guarantees or add acknowledged atomic ownership before
execution. The duplicate-ID regression shows that SDK admission alone cannot
currently be assumed to provide it. An unguarded `get` followed by `set` does not
supply compare-and-set or multi-writer immutability. Do not lock or consume an
ordinary parent response to establish ownership of a child.

Boundary publication happens after graph execution, so its failure does not
undo tool effects already performed. Do not rerun completed graph work merely
to repair an index. Keep transient index records under the existing retention
policy instead of adding a cleanup transaction. Residual index entries must
never extend response availability or resurrect an expired temporary parent.

### Checkpoint Retention

Creating, retrying, or completing a branch must not delete its parent checkpoint
or prune the parent's checkpoint history. Each descendant records its own
checkpoints, and multiple branches may reference the same retained parent.
Checkpoint deletion is not a prerequisite or a step in branching.

All stores needed for recovery must survive restarts. Checkpoint expiration or
deletion belongs to the configured storage TTL/retention policy or explicit
external cleanup, not to the branching flow. The feature must not introduce
automatic cleanup or change those policies. Retention across stores should be
coordinated, but referencing a parent does not guarantee it will be retained
forever. If it becomes unavailable, restore must fail explicitly.

External tools, shared long-term stores, files, and other resources outside the
checkpoint are not rolled back or automatically branch-isolated. Applications
remain responsible for idempotent or replay-safe side effects.

### Parent Selection and Recovery

For a fresh request on the new path:

1. Validate the public linkage fields before admission. With no parent (omitted or `null`) and no explicit conversation, start an independent root. Record its route in the response lifecycle, but do not create a special recoverable empty origin. Invalid parent values must not be normalized to absence.
2. With `previous_response_id`, preserve provider existence and authorization checks, require an eligible stored parent, and resolve the exact reference recorded in its completed response. An opaque response ID or a standalone index is not authorization or proof of completion.
3. Save the parent-linked origin record through `ConversationChainStoreProtocol.set` and await its successful return before graph execution. If the write fails, do not run graph nodes. Duplicate admission must not change the recorded origin.
4. Restore through the strict-read path below and pin graph execution, state lookups, and interrupt handling to the selected checkpoint. Use the current request input/instructions, or the existing HITL resume command for a matching pending interrupt. Do not prepend inherited transcript items a second time.

Route selection uses the client's linkage fields, not a conversation identity
inferred by the SDK from ancestry. A selected parent with a conversation
association must not silently send a parent-only request to the mutable
conversation-head path. Resolve its exact response boundary or fail explicitly.

Use a storage call whose errors reach the caller to confirm the origin write.
Yielding `stream.checkpoint()` alone is insufficient: the checked SDK logs
intermediate persistence errors without raising them into the handler. Keep
the normal response-stream snapshots for execution recovery, but do not claim
that every emitted checkpoint event was successfully persisted.

For a failed or crashed root admitted on the new path:

- Do not automatically resume or replay it, even if its saver already contains intermediate checkpoints. A user retry creates a new root response.
- Do not publish a branchable boundary from partial work. If the SDK later re-enters the handler for recovery, terminate that response as failed without invoking the graph; do not defer it into a recovery loop.
- Do not add a special retention or cleanup mechanism. Existing response lifecycle records and saver-managed intermediate records may remain under their configured TTL; this is not a promise that the attempt leaves no stored data.
- A successfully completed root still publishes its final boundary. A root response that intentionally completes with an HITL approval request is not a crashed root.

For recovery of an already admitted parent-linked response:

1. Prefer its own durably recorded execution checkpoint and continue with input `None`. Do not replace it with a parent checkpoint or the latest thread state.
2. If no execution checkpoint has yet been durably recorded, replay the original input or approval from its confirmed parent origin. This is different from a recorded execution checkpoint becoming unavailable, which must fail.
3. Once the origin is confirmed, recovery must not depend on rereading the parent response. The referenced graph state must still exist and remain readable. A missing, corrupt, or unsupported origin record is an error, not permission to choose latest state.
4. Preserve the admitted mode across restart or configuration changes. Existing legacy tasks keep their legacy recovery path; a new-mode task must not silently become legacy because required metadata is missing. Proving this distinction at the first durable admission is an implementation gate.

Only confirmed response-side progress authorizes mid-run continuation. If the
graph saver advanced beyond that snapshot, unconfirmed work may run again after
recovery. This design does not provide exactly-once tool effects. Preserving
the existing snapshot protocol avoids an unrelated recovery rewrite, but its
failure and retry behavior must remain explicit to applications.

The first scope uses completed response boundaries, not mutable checkpoints of
queued, in-progress, incomplete, failed, or cancelled responses. This is an
explicit host restriction, not a restriction established by OpenAI's create
reference. A running response having an ID is not evidence of a usable graph
boundary. Wider parent-status support requires a defined snapshot contract;
do not claim full OpenAI parent-status compatibility before that is settled.

The [create reference][openai-create] defaults `store` to true when omitted.
`store=false` may consume an eligible available parent, but must not silently
be changed to true to make its child reusable. Foreground unstored responses
are not durable HTTP parents. OpenAI also permits `background=true` with
`store=false`, using temporary response storage for asynchronous execution and
polling ([background guide][openai-background]); temporary availability is not
a durable-parent guarantee. Retention also depends on applicable platform data
policies; the schema default alone does not promise identical retention for
all background requests. Preserve Foundry's applicable policy instead of copying
OpenAI service-specific retention periods. If the SDK cannot honor the requested
storage combination, record the gap explicitly rather than claiming OpenAI
rejects it, and never silently enable longer retention.

Admission always requires an authorized, available completed response and its
actual checkpoint, regardless of residual index entries. Parent unavailability
is an error; the client may explicitly start a new root with manual input
context, but the host must not do this automatically or claim it restores
non-message graph state. Response storage, temporary execution records, and
application-configured LangGraph checkpoint retention are separate: `store=false`
does not itself erase saver history or guarantee zero retention of graph state.

### HITL Compatibility

HITL is an existing workflow to preserve, not a requirement to add arbitrary
approval-history branching. In this host, the response and the graph have
different lifetimes:

1. The response ID exists before graph execution and is carried by the initial response events.
2. At an HITL pause, the host emits approval/tool-call items and then `response.completed`. The response is finished, while the graph remains paused in its checkpoint.
3. The client submits a new request with the approval result. It may use top-level `previous_response_id` to reference the completed paused response, or the existing explicit-conversation path. `approval_request_id` / `call_id` identifies the interrupt, not the parent response.

This is visible in the [host's completion path](../../langchain_azure_ai/agents/hosting/_responses_host.py#L996)
and the [background approval example](../../../../samples/hosting/langgraph-hosted-agents/responses/10_resilient/README.md#L197).
Filtering only on response status `completed` therefore does not exclude HITL.

The [OpenAI MCP guide][openai-mcp] uses a new response with
`previous_response_id` and an `mcp_approval_response` input for ordinary approval.
The [function-calling guide][openai-function-calling] similarly links tool outputs
with `call_id`. Keep these standard item formats; do not add public graph-resume
fields. The cited documentation does not establish a single-use approval rule
or guarantee isolated competing approval branches.

| Request | First-scope behavior |
| --- | --- |
| Normal approval of an active pending interrupt | Preserve the existing matching, validation, rejection, and resume protocol; do not blanket-reject the paused parent. |
| Ordinary input with no matching approval while the selected graph is paused | Preserve existing pending-interrupt handling; do not bypass the pause to execute fresh work. |
| Independent branches from a completed, non-paused graph boundary | Support the parent-selection behavior proved above. |
| Return to an already answered approval checkpoint with a different answer, or create independent sibling approvals from the same pause | No new support in the first scope. Do not advertise isolated HITL forks based only on checkpoint pinning. |

The follow-up probe showed that repeated resume against one paused checkpoint
can reuse the first answer. The implementation must distinguish ordinary active
continuation from unsupported historical or competing approval branches and
define explicit handling for the latter before release. It must not silently
substitute an earlier decision. The current atomic pause-epoch claims pass local
historical-alias and partial-approval tests; distributed competing-approval
validation is still required. Preserving normal HITL alone is not that proof.
Existing duplicate/unmatched approval behavior on the legacy path stays intact.

Historical/competing approval rejection is therefore a first-scope host
limitation, not an OpenAI requirement. Ownership must identify the actual paused
checkpoint/interrupt continuation, including waiting-response aliases that
point to the same pause; claiming only one response ID is insufficient.
It must not consume an ordinary completed parent or prevent unrelated sibling
forks. Supporting independent historical approval decisions later requires
isolation of pending resume writes, not just a different child response ID.

### Restore Failure Contract

For the new branching path, restore the selected checkpoint or fail explicitly.
Never silently use the latest checkpoint, an empty state, a different parent,
or a full-history rerun as a substitute for a failed restore.

| Failure | Required handling |
| --- | --- |
| Missing, expired, deleted, or invalid parent checkpoint reference | Fail with a stable internal reason such as `checkpoint_unavailable`; map it to the appropriate public error envelope below. |
| Reference exists but the actual checkpoint is absent | Fail before graph execution. A surviving index entry does not prove recoverability. |
| Storage timeout or other transient backend failure | Preserve the failure category. Any retry must follow the configured policy and target the same checkpoint. |
| Authorization or deserialization failure | Preserve the appropriate error category without leaking inaccessible state; do not reinterpret it as an empty graph. |

Do not report TTL expiration as a confirmed cause merely because a record is
absent. The saver may return `None` rather than raise on a missing checkpoint;
the host must handle this explicitly. Proposed mechanism: a request-scoped
checkpointer adapter or a supported LangGraph invocation hook must guard the
actual reads of required checkpoint IDs. A `None` result, a mismatched checkpoint
identity, or a failed read must raise before graph nodes execute; it must not
reach LangGraph's empty-state fallback. Preserve backend error categories and
only retry the same requested checkpoint under the configured retry policy.

Do not globally replace or mutate the user's saver on a shared graph, and do
not change legacy reads or ordinary writes. The adapter/hook must work for
state and interrupt inspection as well as execution, preserve saver lifecycle,
and distinguish required restores from legitimate creation of new checkpoints.
Its supported integration point and concurrency behavior require a focused
prototype. A preflight lookup followed by an unguarded `astream` is not an
acceptable substitute, because expiration or external cleanup can occur between
them. A failed restore must not execute graph nodes or produce new tool effects.

The review's deliberate deletion was fault injection in a temporary in-memory
saver, simulating a checkpoint becoming unavailable between validation and
restore. It did not access production storage and is not an implementation step
for branching. The intended behavior is to retain and reference the parent,
then fail explicitly if that checkpoint is no longer available.

Use the published wire schema, not arbitrary SDK-accepted error strings. The
[create reference][openai-create] and [ResponseError schema][openai-response-error]
enumerate `Response.error.code`; `checkpoint_unavailable`, `invalid_branch_state`,
`unsupported_approval_branch`, and `branch_execution_error` are not members.
They may identify internal failure reasons, but must not be serialized directly
there, including inside a terminal `response.failed` event.

| Surface | Error contract |
| --- | --- |
| Before acceptance | Use an HTTP error envelope with `error.message`, `type`, `param`, and nullable `code` ([schema][openai-request-error]). Reject mixed/invalid linkage with HTTP 400 and `type="invalid_request_error"`, naming the offending parameter. Preserve safe authorization/not-found/backend HTTP handling; do not pretend backend failures are invalid user input. |
| Accepted response failure | Persist a failed Response and emit `response.failed` for SSE. Map checkpoint/internal execution failures to the valid `server_error` code with a safe message; use another published code only when its meaning actually applies. JSON results, terminal SSE, and retrieval must agree. |
| Standalone SSE `error` | Follow its separate [event schema][openai-stream-error]: top-level `code`, `message`, `param`, `sequence_number`, and `type="error"`. It is not the HTTP wrapper or a substitute for persisting an accepted response's terminal state. |

HTTP and standalone SSE error codes allow strings, but host-defined values are
not thereby OpenAI-defined codes. Keep precise backend/branch categories and
causes in internal diagnostics; the public terminal enum may intentionally map
several reasons to `server_error`. The earlier requirement for a distinct custom
terminal code for every branch failure is withdrawn in favor of schema
compatibility. Do not invent public fields, leak internal metadata, or force
clients to parse messages to recover those internal reasons. Verify safe failure,
terminal status, and valid envelopes across JSON, SSE, retrieval, and recovery.

These strict rules apply to the new path, including its recovery operations;
they do not add response-boundary selection to legacy paths. The later shared
strict-error decision additionally removes swallowed state-read exceptions in
legacy handlers. Fresh roots do not require a parent
checkpoint. Failed new-mode roots terminate rather than recover; parent-linked
tasks without confirmed execution progress may replay only from their confirmed,
still-readable origin.

### Long-Running Behavior

Assume A is completed and B is a long-running response started from A:

| Operation | Intended behavior after implementation |
| --- | --- |
| Continue or fork after B completes | Use B's fixed boundary checkpoint. |
| Create C from A while B is still running | Run an independent sibling; do not automatically cancel B. |
| Retry B from A after B fails | Submit B's input again with parent A, creating a new response ID. |
| Recover parent-linked B after a process crash | Use its confirmed execution checkpoint, or its confirmed parent origin when no progress was durably recorded. Required state missing means failure. |
| The initial root request fails before publishing a completed response | Do not resume or replay it automatically. A client retry starts a new root. |
| B completes its response with an HITL approval request | Continue through a new ordinary approval request; do not keep B's response open while waiting. |
| Fork from an arbitrary internal step while B is running | Out of scope; B's response ID does not identify a particular intermediate checkpoint. |

Long runtime does not change selection semantics or the failed-root policy.
The failed-root policy is a host recovery decision, not a meaning of OpenAI's
`previous_response_id`. As the [background guide][openai-background] explains,
polling and cancellation operate on the existing response, and
`starting_after` resumes its event stream after a sequence number. None creates
a child or requests graph replay. Reconnecting to a background stream requires
that response to have been created with `stream=true`. Leaving `queued` or
`in_progress` is terminal, not necessarily successful or eligible as a parent.

Recovery of eligible parent-linked tasks still requires the existing resilience
configuration and persistent task, response, origin/boundary-reference, and
graph stores. The local proof covered
background execution and concurrent branches, not hours-long runs, process
restart, or a deployed Foundry backend.

### Existing Data

Do not rewrite existing records or require a destructive migration. A trusted
legacy per-response snapshot may be reused only when its identity, eligibility,
exact checkpoint, and instruction provenance are verifiable. A shared mutable
conversation pointer is never evidence of a historical response boundary. If
no trustworthy reference exists, reject its use on the new branching path
rather than guessing; start a new root for verifiable future boundaries.
Legacy latest-state operation remains an opt-out from historical selection,
not an opt-out from public linkage or instruction rules. Missing instruction
provenance follows the fail-before-execution policy above on either route.
Callers can explicitly start a root with curated input context, but this is not
restoration of the old graph's non-message state.

### Implementation Scope and Acceptance

Keep the implementation local to Responses hosting, but treat the earlier file
counts as a preliminary estimate, not a limit or verified guarantee:

| Area | Expected changes |
| --- | --- |
| Initially estimated 2-3 hosting source modules | [Host constructor](../../langchain_azure_ai/agents/hosting/_responses_host.py#L318), selection, publication, and execution handling; origin/boundary records and recovery metadata in existing helpers. Public linkage validation, request-local instructions, error mapping, strict reads, and HITL admission may require additional focused work. |
| Approximately 1-2 test modules | Extend existing Responses host tests and reuse current fixtures. |
| Approximately 1-2 documentation/sample locations | Explain opt-in setup, saver requirements, branching, and background behavior. |
| No planned changes | Invocations time-travel capabilities, dependencies, endpoints, chain-store protocol, or persistent backend implementations. Shared strict-error propagation and its Invocations SSE fix are the later exception recorded in the handoff. Ordinary completed-turn forks were proved without an SDK change; the full design still depends on the integration gates below. |

Implementation gates:

- Establish OpenAI-compatible linkage validation before SDK normalization/admission, distinguish platform headers from public fields, and preserve immediate-parent identity in every response representation.
- Prove request-local top-level instructions do not leak through checkpoint/history restore, without dropping explicit input messages, modifying the parent, or breaking same-task recovery.
- Prove a supported, request-scoped strict-read integration that cannot affect concurrent legacy runs or silently restore empty state.
- Verify effective provider/options access for both host-created and injected `app` instances, including tenant-context-aware parent reads and stored internal metadata.
- Prove the first durable admission identifies new-mode roots/children versus legacy tasks. Define missing/unknown metadata behavior and ensure root recovery is terminated rather than silently reclassified or retried.
- Verify per-response single-writer ownership, idempotent boundary publication, and the SDK terminal-persistence failure paths. Do not assume atomic replacement is conditional creation.
- Preserve ordinary HITL while detecting unsupported historical or competing approvals; specify and test their behavior without claiming independent paused-checkpoint branches.

If a gate cannot be satisfied through current supported interfaces, revise the
implementation scope explicitly instead of weakening the contract or adding
private SDK patches implicitly. The earlier local proof does not settle these
gates.

Required checks before shipping:

- Requests admitted with the flag omitted or explicitly `False` preserve valid existing behavior except for the explicit strict-read, linkage-validation, and instruction-lifetime corrections. Legacy graph storage/selection remains unchanged and is documented as a compatibility limitation.
- `previous_response_id` omitted/null creates a root only without an explicit conversation; empty/wrong-type/unresolved IDs fail without execution. Cover conversation string/object forms and nulls, with both non-null linkage fields rejected regardless of the feature flag.
- The selected immediate parent and new child identity survive JSON, SSE, and retrieval unchanged. Reusing a parent for siblings succeeds; duplicate new platform-assigned identities cannot execute competing work or overwrite the original. No private checkpoint fields, admission markers, or synthetic conversation association leak into public responses.
- Parent top-level instructions are absent on child requests that omit/null them; replacement instructions apply only to the child. Explicit system/developer input items and graph-owned prompts survive. Include ambiguous legacy messages with identical text: fail before execution when provenance cannot be established. Test graph-visible context and same-task recovery, not only response serialization.
- Missing/invalid saver fails during construction before SDK setup, including `app` attachment; initialization performs no storage I/O and does not auto-create a saver. Effective steering configuration is validated rather than guessed.
- Root creation, linear continuation, sibling forks, regeneration, and continuation of each branch preserve non-message graph state and immutable parent references.
- Branch creation, regeneration, and completion do not delete parent checkpoints, prune parent history, or change configured TTL/retention policies.
- Concurrent branches cannot capture one another's checkpoint; verify on the intended persistent saver, not only `InMemorySaver`.
- Origin writes are acknowledged before graph execution; failed writes invoke no nodes. Missing/corrupt origin data on recovery fails explicitly instead of selecting latest state.
- Boundary-index writes alone do not admit parents. Cover terminal-persistence failure, missing/mismatched references, conflicting publication, and recovery between index and terminal writes.
- Missing references, expired/deleted checkpoints, restore-time disappearance, authorization failures, and backend errors fail explicitly without graph/tool execution or latest-state fallback.
- Foreground JSON, SSE, and background execution obey the public response/error schemas, including the terminal error-code enum. Cover `store` omitted/true/false, temporary background retention, unavailable parents, and retrieval/reconnection without silently changing storage policy or restarting execution. Any remaining SDK incompatibility is an explicit release gap.
- New-mode roots that fail before completion are not replayed, including SDK recovery re-entry after saver advancement. Successful retained roots publish boundaries usable only within their response/checkpoint availability window; foreground `store=false` roots may complete without a reusable boundary. Legacy root recovery is unchanged.
- Parent-linked recovery before the first durable execution checkpoint, mid-run recovery, and interruption around publication use the recorded origin/progress and admitted mode. Missing required progress must not fall back to the parent.
- Ordinary HITL approvals, rejections, parallel pending interrupts, and response-ID/conversation linkage preserve existing behavior. Historical/competing approval requests must not silently inherit another branch's answer; independent historical approval forks remain unsupported.
- Cancellation, legacy conversations, flag changes, and unsupported steering combinations retain their defined behavior. Failed roots must not enter an automatic recovery loop.
- User/deployment isolation, old records, unstored responses, retention mismatch, and the supported Python/dependency versions are covered.

The main verification effort is compatibility and recovery correctness, not
the parent lookup itself. Automatic recovery of failed new-mode roots, arbitrary
node-level time travel, independent historical HITL forks, steering-compatible
historical forks, and unrelated OpenAI field changes remain outside this scope.
Linkage validation, instruction lifetime, and compatible error envelopes are
now inside scope because they directly affect `previous_response_id`. The
default-off historical-selection gap, completed-parent restriction, and lack
of independent approval forks must remain visible limitations, not claims of
full OpenAI conformance.

## InvocationsHostServer

- The [default parser](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L616) requires non-empty `message` text or structured HITL items. It reads `stream`; [execution options](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L664) read `background` and `previous_invocation_id`. No default field selects a graph checkpoint or supplies state edits.
- [Foreground config](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L698) and [task config](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L722) select a user-scoped session thread, without a client-selected checkpoint.
- [Task admission](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L1123) maps `previous_invocation_id` to `if_last_input_id`. The [SDK precondition][sdk-task] compares it with the last accepted input ID, not graph history. A mismatch on an existing head produces [HTTP 409](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L944); with no stored head the SDK accepts and seeds it. Without task-backed mode, the host [rejects the option](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L863) with HTTP 400.

## Backend and Related Features

- `FoundryCheckpointSaver` is [async-only](../../langchain_azure_ai/agents/hosting/_foundry_checkpoint_saver.py#L111). It implements [exact-ID/latest lookup](../../langchain_azure_ai/agents/hosting/_foundry_checkpoint_saver.py#L171), [newest-first history](../../langchain_azure_ai/agents/hosting/_foundry_checkpoint_saver.py#L206), [append-only checkpoint writes with parent IDs](../../langchain_azure_ai/agents/hosting/_foundry_checkpoint_saver.py#L296), and [restored parent configs and pending writes](../../langchain_azure_ai/agents/hosting/_foundry_checkpoint_saver.py#L475). These are the backend primitives for async graph time travel, not evidence of an HTTP feature or runtime conformance.
- Its history listing requires a known thread; namespaces, `before`, metadata filters, and limits are supported. Use `aget_state_history`, `aupdate_state`, and `ainvoke` with this saver. Retained checkpoints and the correct user partition are required; [default TTL](../../langchain_azure_ai/agents/hosting/_foundry_checkpoint_saver.py#L43) is 30 days, and expiration/deletion removes available history.
- Crash recovery: [Responses](../../langchain_azure_ai/agents/hosting/_responses_host.py#L891) and [task-backed Invocations](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L1296) restore internally recorded checkpoint IDs and pass graph input `None`. This continues interrupted work, not an arbitrary user-chosen past run. Cross-process recovery requires surviving stores; Responses also requires the [resilience flags and providers][sdk-resilience].
- SSE replay: the [Responses reconnect contract][sdk-resilience] returns stored events after `starting_after`, then live-tails. [Invocations event subscription](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L1860) also consumes events. Replaying transport events is not re-executing graph nodes.
- HITL resume: [Responses](../../langchain_azure_ai/agents/hosting/_responses_host.py#L558) and [Invocations](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L760) map matching approval/tool-output items to a resume command for currently pending interrupts. This does not expose historical checkpoint selection.

## Extension Hooks

- Direct Python access through `host.graph` retains native LangGraph APIs when the hosted object and saver support them. Preserve the host's user-scoped thread identity and storage context.
- Responses provides `build_runnable_config`, `build_input`, and [custom `handle_create`](../../langchain_azure_ai/agents/hosting/_responses_host.py#L797); a custom SDK response handler is another option. A true time-travel interface must define authorized checkpoint selection, replay input `None`, forks, and branch bookkeeping. Replacing the chain store alone does not supply those controls.
- Invocations provides `parse_request`, `parse_execution_options`, `build_input`, and separate foreground/task config hooks. Overriding only the foreground config misses task-backed execution. Simply returning `None` from `build_input` is also insufficient: default handlers [short-circuit foreground](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L881) and [fresh task-backed](../../langchain_azure_ai/agents/hosting/_invoke_host.py#L1329) `None` inputs instead of invoking a replay. A custom execution route/handler is needed for that operation.

## Verification Boundaries

The results in this section belong to the initial investigation before source
implementation. Current WIP verification and failures are recorded separately
in [Implementation Handoff](#implementation-handoff).

Local verification used Python 3.14, LangGraph `1.2.11`, Agent Server Core and
Responses `2.1.0b2`, and Invocations `1.1.0b1` in a temporary uv-managed overlay.
The project environment, dependency files, and host implementations were not
changed. The 18 selected tests in the existing Responses and Invocations host
modules passed; SDK telemetry setup was disabled in the verification process.

Additional probes used real local HTTP adapters through Starlette `TestClient`,
a deterministic two-node graph, `InMemorySaver`, and the existing fake Foundry
state-store fixtures. Each row below used an isolated graph unless noted.

| Probe | Observed result |
| --- | --- |
| Checkpointed Responses: A, then B after A, then C referencing A | Graph saw `A, B, C`, not `A, C`. |
| Same graph: next request also supplies A's top-level `checkpoint_id` | The field did not select A; graph saw `A, B, C, E`. |
| Responses without a graph checkpointer: A, B after A, C after A | History branch saw `A, C`; this is transcript branching only. |
| Steerable Responses: A, B after A, C after A | HTTP 409, `conversation_fork_not_supported`. |
| Foreground Invocations: A, B, C with A's `checkpoint_id` and `config` | Graph saw `A, B, C`; supplied checkpoint configuration was ignored. |
| Foreground Invocations with `previous_invocation_id` | HTTP 400 because task-backed mode was not enabled. |
| Task-backed Invocations: A, B after A, C with A's invocation ID | HTTP 409: `previous_invocation_id does not match the latest turn.` |
| Native graph replay from A's pre-report checkpoint | `ainvoke(None, snapshot.config)` produced only A. |
| Native graph fork from that checkpoint, adding D | `aupdate_state(..., as_node="remember")`, then `ainvoke(None, ...)`, produced `A, D`. |

SDK source review additionally covered release tag
`azure-ai-agentserver-responses_2.1.0b2` and commit
`4993311d43a35b34cb47220eff0df969d73d7832` ([Responses version `2.2.0b2`][sdk-version]).
The [package requirements](../../pyproject.toml#L1) allow more versions than were
executed. No deployed Foundry service, persistent Foundry saver, or model was
called; backend capabilities above are source-verified, not cloud-tested.

[sdk-chain-release]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/hosting/_chain_id.py
[sdk-chain-main]: https://github.com/Azure/azure-sdk-for-python/blob/4993311d43a35b34cb47220eff0df969d73d7832/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/hosting/_chain_id.py
[sdk-options]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/_options.py
[sdk-task]: https://github.com/Azure/azure-sdk-for-python/blob/4993311d43a35b34cb47220eff0df969d73d7832/sdk/agentserver/azure-ai-agentserver-core/azure/ai/agentserver/core/tasks/_decorator.py
[sdk-resilience]: https://github.com/Azure/azure-sdk-for-python/blob/4993311d43a35b34cb47220eff0df969d73d7832/sdk/agentserver/azure-ai-agentserver-responses/docs/resilience-contract.md
[sdk-version]: https://github.com/Azure/azure-sdk-for-python/blob/4993311d43a35b34cb47220eff0df969d73d7832/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/_version.py
[openai-migration]: https://developers.openai.com/api/docs/guides/migrate-to-responses
[openai-create]: https://developers.openai.com/api/reference/resources/responses/methods/create
[openai-state]: https://developers.openai.com/api/docs/guides/conversation-state
[openai-background]: https://developers.openai.com/api/docs/guides/background
[openai-streaming]: https://developers.openai.com/api/docs/guides/streaming-responses
[openai-mcp]: https://developers.openai.com/api/docs/guides/tools-connectors-mcp
[openai-function-calling]: https://developers.openai.com/api/docs/guides/function-calling
[openai-response-error]: https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_error.py
[openai-request-error]: https://github.com/openai/openai-python/blob/main/src/openai/types/shared/error_object.py
[openai-stream-error]: https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_error_event.py
[sdk-request-parsing]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/hosting/_request_parsing.py
[sdk-provider]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/store/_base.py
[sdk-context]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/_response_context.py
[sdk-orchestrator]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/hosting/_resilient_orchestrator.py
[sdk-execution]: https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-agentserver-responses_2.1.0b2/sdk/agentserver/azure-ai-agentserver-responses/azure/ai/agentserver/responses/hosting/_orchestrator.py
