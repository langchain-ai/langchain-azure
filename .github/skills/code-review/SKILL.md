---
name: code-review
description: >-
  Reviews changes in the langchain-azure monorepo using package-specific
  knowledge of langchain-azure-ai, langchain-azure-compute,
  langchain-azure-cosmosdb, langchain-azure-postgresql, langchain-azure-storage,
  langchain-sqlserver, and langchain-azure-dynamic-sessions, together with the
  LangChain, LangGraph, Deep Agents, and Azure SDK contracts each package must
  satisfy. Use this skill whenever reviewing a pull request or diff, checking
  code for bugs or regressions, or assessing changes under libs/, samples/, or
  .github/ in this repository, including when the request is only to "review",
  "check", "look at", or "give feedback on" a change, and even when no package
  is named explicitly.
license: MIT
---

# Reviewing langchain-azure changes

Every directory under `libs/` is a separately versioned, separately released
package with its own maintainers, dependency manager, conventions, and upstream
contracts. A finding is only useful if it is correct *for the package it lands
in*, so the first job in any review is to work out which package changed and
load that package's rules before judging anything.

## What to report

Report defects the change introduces: incorrect behavior, broken edge cases,
regressions in released public API, violations of an upstream contract
(LangChain, LangGraph, Deep Agents, Azure SDK), breakage on a supported Python
version, and security, credential-leak, data-loss, resource-leak, or
concurrency problems.

Stay silent about everything else. In particular, do not comment on formatting,
naming, or docstring wording that `ruff` and `mypy` already enforce; do not
restate what the diff does; do not raise pre-existing issues the change merely
touches; and do not suggest refactors that are not required for correctness.
Returning no comments on a correct change is a good review — never manufacture
findings to look thorough.

Copilot code review never sees `**/*.lock`, `**/*.svg`, `**/*.log`, or
`**/dist/**`, so `uv.lock` is invisible to you. Never write a finding about
lockfile contents; instead, see the lockfile gotcha below.

## Review workflow

1. **Identify the packages touched.** Group changed files by `libs/<package>/`.
   Treat each package as an independent review.
2. **Load the package's rules** from the routing table below, plus any
   `AGENTS.md` or `.github/copilot-instructions.md` along the changed path. The
   repository root `AGENTS.md` is already in your context; do not re-derive it.
3. **Read enough surrounding code** to know what the changed lines actually do:
   the function, its callers, its sync or async twin, and the nearest tests.
   Never review a hunk in isolation.
4. **Check the upstream contract** for the base class being implemented, using
   [ecosystem contracts](references/ecosystem-contracts.md) and
   [Azure SDK contracts](references/azure-sdk-contracts.md).
5. **Confirm each finding** before writing it. A finding must satisfy all four:
   the changed code causes it; a realistic supported input or code path reaches
   it; the consequence is concrete; and you can point at the specific lines. If
   any of these is missing, drop it.

## Package routing

Read the file for each package that changed. Skip the rest.

| Changed path | Package | Read |
|---|---|---|
| `libs/azure-ai/` | `langchain-azure-ai` | [azure-ai.md](references/azure-ai.md) |
| `libs/azure-compute/` | `langchain-azure-compute` | [azure-compute.md](references/azure-compute.md) |
| `libs/azure-cosmosdb/` | `langchain-azure-cosmosdb` | [azure-cosmosdb.md](references/azure-cosmosdb.md) |
| `libs/azure-postgresql/` | `langchain-azure-postgresql` | [azure-postgresql.md](references/azure-postgresql.md) |
| `libs/azure-storage/` | `langchain-azure-storage` | [azure-storage.md](references/azure-storage.md) |
| `libs/sqlserver/` | `langchain-sqlserver` | [sqlserver.md](references/sqlserver.md) |
| `libs/azure-dynamic-sessions/` | deprecated | [azure-dynamic-sessions.md](references/azure-dynamic-sessions.md) |
| `.github/`, `samples/`, root docs | repo infrastructure | [repo-infrastructure.md](references/repo-infrastructure.md) |

Also read [ecosystem contracts](references/ecosystem-contracts.md) when the
change implements or overrides a LangChain, LangGraph, or Deep Agents base
class, and [Azure SDK contracts](references/azure-sdk-contracts.md) when it
constructs an Azure client, handles credentials, or maps service errors.

## Repository gotchas

These are the mistakes that pass local review and break later. They are
specific to this repository and override any general instinct.

- **A `pyproject.toml` dependency edit needs a matching `uv lock`.** CI runs
  `uv lock --check` and fails if `uv.lock` is stale. You cannot see the
  lockfile, so when a diff changes dependencies, check that `uv.lock` is in the
  changed-file list and flag it when it is absent.
- **CI only runs Python 3.10 and 3.14, but the support range is 3.10–3.14.**
  A construct that breaks only on 3.11–3.13 passes CI. Reason about the whole
  range rather than trusting a green build.
- **`langchain-azure-compute` enforces 100% coverage** (`fail_under = 100`).
  A new uncovered branch there fails CI, so a new `if` or `except` without a
  test is a real finding in that package only.
- **`langchain-azure-ai` lazy imports must be updated in three places** — the
  `TYPE_CHECKING` import, `__all__`, and `_module_lookup`. Updating fewer makes
  the symbol import-time-invisible or `__all__`-inconsistent, and unit tests
  catch only some of these.
- **Deprecation decorators differ per package.** `azure-ai` and
  `azure-dynamic-sessions` use their own `_api.base` (`deprecated`,
  `experimental`); the other packages use `langchain_core._api` (`beta`,
  `deprecated`). Do not flag one package for using the other's convention.
- **New Azure client construction must stamp the package user agent.** Each
  package defines its own constant or helper (`USER_AGENT`, `_user_agent`,
  `get_user_agent`, `with_user_agent`). A new client path that omits it
  silently drops partner telemetry attribution.
- **`asyncio_mode = "auto"`** in every package: async tests need no
  `@pytest.mark.asyncio`. Do not ask for it.
- **`--strict-markers` and `--strict-config`** are set: a new `pytest.mark.*`
  must be registered in that package's `pyproject.toml` or collection fails.
- **`azure-cosmosdb`'s local instructions still describe `poetry`**, but its
  `Makefile` and CI use `uv run --frozen`. The `Makefile` is authoritative;
  do not flag correct `uv` usage there.
- **`azure-postgresql`'s local instructions ask for Sphinx-style docstrings**
  while its `ruff` config sets `pydocstyle` convention to `google`. Follow the
  style already used in the file being changed and raise no docstring-style
  findings in that package.
- **Unit tests must not touch the network.** `azure-ai` enforces this with
  `pytest-socket`; the same expectation applies everywhere. A new unit test that
  reaches a live service is a finding.

## Severity

Copilot code review labels comments High, Medium, or Low. Use that vocabulary.

- **High** — data loss, credential or secret exposure, a regression in released
  public API, or a failure most users of the changed path will hit.
- **Medium** — a real correctness, compatibility, or resource-handling defect
  on a narrower but supported path.
- **Low** — a genuine but minor defect worth fixing.

If a finding does not clear the Low bar, leave it out.

## Comment format

Keep each comment to the smallest useful line range and this shape:

> **[Severity] Short imperative title**
>
> What breaks, and the specific input or code path that triggers it. Which
> contract or package rule it violates. One concrete suggested fix, only when
> it is short and unambiguous.

Cite the contract by name (for example, "`VectorStore.get_by_ids` must not
raise for missing IDs") rather than linking to documentation, and prefer one
precise sentence over a paragraph of hedging.
