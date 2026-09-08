# `langchain-azure-compute` (`libs/azure-compute`)

Two **different Azure products** live here, not two tiers of one:

| | Dynamic sessions | Sandboxes |
|---|---|---|
| ARM resource | `Microsoft.App/sessionPools` | `Microsoft.App/sandboxGroups` |
| Access | HTTP via pool management endpoint | per-sandbox data plane SDK |
| State | ephemeral, destroyed after cooldown | stateful: suspend, resume, snapshot |
| Storage | none | volumes |

Do not apply one side's lifecycle, endpoint, or credential assumptions to the
other.

## Coverage gate

This package enforces `fail_under = 100` on unit-test coverage, and integration
tests are not measured. A new branch — including a new `except` or an early
return — without a unit test **fails CI**. This is the one package where "add a
test for this branch" is a mechanical finding rather than a judgment call.

## Extras and imports

Extras control dependencies, not module presence: every module ships in the
wheel. Importing a subpackage without its dependencies must raise the existing
actionable error naming the extra, and must not break unrelated imports.
`deepagents` is installed only on Python 3.11+, so import-time code must
tolerate its absence on 3.10.

Both Deep Agents backends are marked `@beta` via `langchain_core._api` (this
package uses `langchain_core._api`, not a local `_api.base`). The `@beta`
marker is load-bearing and covered by tests; removing or relocating it changes
observable warning behavior.

## Data-plane realities

These are documented service behaviors the code deliberately works around.
Changes that drop the workaround are defects even though the code looks
simpler afterwards:

- The dynamic sessions data plane **intermittently drops a command's final
  line of output** (measured at 1.6–4%). `ls`, `read`, `glob`, and `grep`
  detect this with a completion marker and retry. `execute()` is deliberately
  not covered, because wrapping it would change the exit status it returns.
- The data plane **silently caps stdout and stderr at 4,096 bytes**. `read`
  pages beneath the cap in base64 chunks; listings that exceed it must report
  `truncated` honestly. Returning a truncated result as complete is a
  High-severity finding.
- Session file transfer is a **flat store rooted at `/mnt/data`**. Only
  `/mnt/data/<name>` is storable; anything else must be rejected with
  `invalid_path` rather than silently rewritten. `write()` goes through the
  shell and has no such limit.
- Shell command payloads are capped by Linux at 128 KiB per string; `write()`
  and `edit()` refuse above roughly 90 KB, and `edit()` budgets `old_string`
  and `new_string` together because both ride in one command.

## Protocol conformance

`SessionsBashBackend` and `ACASandbox` implement Deep Agents protocols, so the
rules in [ecosystem-contracts.md](ecosystem-contracts.md) apply exactly:
errors in the result `error` field rather than raised, input-ordered
`upload_files` / `download_files` results, the four `FileOperationError`
literals, `ReadResult` pagination invariants, literal-string `grep`, and
`exit_code` for failure detection.

`ACASandbox` conformance is checked against
`langchain_tests.integration_tests.SandboxIntegrationTests`.

`glob()` follows Python-glob semantics for common shapes (`*.py` stays in one
directory level, `**/name` recurses); patterns with several `**` segments use a
documented `find -path` approximation. As in Python, a wildcard does not match a
leading dot.

## Client ownership

`ACASandbox` wraps a **caller-constructed** client and never builds endpoints or
handles credentials itself. A change that makes it construct, replace, or close
that client, or that moves sandbox lifecycle control inside the backend, breaks
the ownership contract — see the lifecycle section of
[azure-sdk-contracts.md](azure-sdk-contracts.md).

For dynamic sessions, preserve token scopes, custom `access_token_provider`
support, session ID propagation and reuse, cooldown and deletion semantics,
request timeouts, and context-manager cleanup.
