# Azure SDK contracts

Read this when a change constructs an Azure client, handles credentials, maps
service errors, logs, paginates, or adds telemetry. These rules come from the
Azure SDK for Python design guidelines and the `azure-sdk-tools` pylint
checkers, and they apply to every package here because all of them wrap
`azure-*` clients.

Where a rule below is enforced by a pylint checker, the code is given so the
finding can be stated precisely.

- [Credentials](#credentials)
- [Sync and async separation](#sync-and-async-separation)
- [Errors](#errors)
- [Retries, timeouts, and cancellation](#retries-timeouts-and-cancellation)
- [Client lifecycle](#client-lifecycle)
- [Pagination and long-running operations](#pagination-and-long-running-operations)
- [Logging](#logging)
- [Telemetry and user agent](#telemetry-and-user-agent)
- [API surface hygiene](#api-surface-hygiene)
- [Where Azure and LangChain conventions conflict](#where-azure-and-langchain-conventions-conflict)

## Credentials

- Accept `credential` as a single parameter and support `TokenCredential` /
  `AsyncTokenCredential`, not just keys (`C4717`). A parameter that accepts
  only an API key is a design regression in a package whose other integrations
  accept Entra ID.
- Connection strings belong in a `from_connection_string` factory, never in
  `__init__` (`C4736`).
- Async credentials must be used with the async protocol; passing a sync
  credential into an `.aio` client is a real defect, not a typing nit.
- **Never let a credential, key, SAS token, or connection string reach a log,
  an exception message, a `__repr__`, or model-visible output.** In LangChain
  classes this is sharper than usual: `_identifying_params` is logged *and*
  used as a cache key, so a secret leaked there is persisted, not just printed.

## Sync and async separation

- Sync and async clients are separate classes with the **same name**,
  distinguished by namespace (`azure.x` vs `azure.x.aio`).
- Do not put `Async` in a class name or suffix methods with `_async` (`C4731`).
- A single class that switches at runtime between a sync client and an `.aio`
  client violates the separation rule.
- An `@overload` set must be uniformly sync or uniformly async (`C4765`).
- The LangChain-side consequence: `_agenerate`, `aadd_texts`,
  `asimilarity_search`, and friends should be backed by a genuine `.aio`
  client. `run_in_executor` around the sync client is acceptable only when no
  async client exists, and is worth flagging when one does.

## Errors

The `azure-core` hierarchy is:

```
AzureError
├── ServiceRequestError      # request never left the client
├── ServiceResponseError     # no response received; retryable if idempotent
└── HttpResponseError        # any non-success response; .status_code, .error
    ├── ResourceNotFoundError, ResourceExistsError
    ├── ResourceModifiedError, ResourceNotModifiedError
    ├── ClientAuthenticationError
    └── DecodeError, TooManyRedirectsError, ...
```

- Prefer these types over inventing new ones. A new exception type is justified
  only when the caller can remediate the error programmatically; otherwise add
  context to an existing type.
- Always chain with `raise ... from e` so the original error survives.
- **Do not signal failure by return value.** Returning `None`, `False`, or `[]`
  where the service actually failed hides outages. The deliberate exceptions
  are `exists`-style checks (return `False` on 404) and LangChain's
  `get_by_ids` (return fewer documents rather than raising).
- Do not raise for normal responses.
- Validate parameters the client itself consumes, especially URL components.
  Do not add null/empty/range checks for values the service validates — that
  duplicates validation and drifts as the service evolves.
- A blanket `except Exception` around a service call that returns a
  success-shaped default is a High-severity finding: it converts an outage into
  silently wrong data.

## Retries, timeouts, and cancellation

- `azure-core` supplies retry and timeout policies; per-call keyword arguments
  must use the same names as the constructor policy options so callers can
  override per request (`C4771` for a missing retry policy).
- Any method performing I/O should thread `**kwargs` through to the client so
  per-request policy options survive (`C4728`).
- Custom retry loops layered on top of `azure-core`'s retry policy usually
  produce accidental exponential retry multiplication; check whether the SDK is
  already retrying before accepting a hand-rolled loop.

## Client lifecycle

Azure clients own network resources. In practice every `azure-*` client exposes
`close()` and the (async) context-manager protocol, and credentials and
pipelines require it by rule. A LangChain integration that constructs a client
must either own and expose a way to release it, or use `with` / `async with`
internally.

The ownership question is the one to review: when the caller passes a client or
session in, the integration **must not** close it; when the integration creates
it, it must. Getting this backwards either leaks connections or closes a client
the caller is still using.

## Pagination and long-running operations

- `list_*` methods return `ItemPaged[T]` / `AsyncItemPaged[T]` (`C4733`).
  Iterating yields items; `.by_page()` yields pages. Continuation tokens are
  reached only through `.by_page(continuation_token=...)`, never as a method
  parameter.
- Code that consumes a paged result must iterate it to completion (or page
  deliberately). Taking the first page and treating it as the full result is a
  common and silent correctness bug — flag it wherever a listing feeds a
  document set, a delete set, or a search result.
- `begin_*` methods return `LROPoller` / `AsyncLROPoller` (`C4734`, `C4735`);
  `delete_*` returns `None` (`C4752`).

## Logging

- Use the stdlib `logging` module with a module-named logger.
- **Never log sensitive information above `DEBUG`**, and redact headers.
- Do not log an error and re-raise it — that double-reports. Do not use
  `logging.exception()` in that position (`C4762`).
- Do not log exceptions at levels other than `DEBUG`, because exception text
  can carry service payloads and credentials (`C4766`).
- Body-level network logging must stay opt-in (`logging_enable=True`).

The two most reviewable versions of these rules: an f-string log line that
interpolates an endpoint or URL that may carry a SAS token, and a
`logger.exception(...)` sitting next to a `raise`.

## Telemetry and user agent

- The SDK user-agent format is
  `[<application_id> ]azsdk-python-<package>/<version> <platform>`, supplied by
  `UserAgentPolicy`; `application_id` is capped at 24 characters.
- Every package in this repository stamps its own partner user agent on client
  construction. A new client path that skips it drops attribution — see the
  gotcha in `SKILL.md`.
- Telemetry values must never contain PII.

## API surface hygiene

- Non-public API takes a single leading underscore; `__all__` lists the public
  names. Anything under a `_private` module is internal even if the leaf name
  looks public.
- Optional or rarely used arguments are keyword-only; no more than five
  positional parameters (`C4721`).
- `**kwargs` is for pass-through to a lower layer only, and the target must be
  documented. Arguments the method itself consumes must be explicit.
- Prefer properties over getter/setter pairs; avoid `@staticmethod` in favor of
  module-level functions (`C4725`).
- An optional keyword-only `api_version` should be accepted, default to the
  latest non-preview version, and never be silently ignored (`C4748`).

## Where Azure and LangChain conventions conflict

These are the places where a reviewer applying one ecosystem's rules to the
other produces a false positive:

| Topic | Azure SDK | LangChain | Which wins here |
|---|---|---|---|
| Missing item lookup | raise `ResourceNotFoundError` | `get_by_ids` returns fewer, never raises | LangChain, for `get_by_ids` |
| Type hints | prefers `typing.Optional` / `Union` | uses `X \| Y` with `from __future__ import annotations` | LangChain style, as used in this repo |
| Naming | approved verb prefixes (`get_`, `list_`, `begin_`) | LangChain method names (`add_texts`, `similarity_search`) | LangChain, for the integration surface |
| Errors from tools | raise | `ToolException` so the agent recovers | LangChain, inside tools and backends |

Outside these, the Azure rule applies: this repository ships Azure client
libraries, and its users expect Azure error types, credential handling, and
logging behavior.
