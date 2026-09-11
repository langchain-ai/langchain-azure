# Repository infrastructure

Read this when a PR touches `.github/`, `pyproject.toml`, `Makefile`, `uv.lock`,
version numbers, or `samples/`.

## CI shape

`check_diffs.yml` detects which packages a diff touches (via
`.github/scripts/check_diff.py`) and fans out to reusable workflows:
`_lint.yml`, `_test.yml`, `_codespell.yml`, `_compile_integration_test.yml`,
and `_test_release.yml`. Releases run through `_release.yml`.

Two consequences for review:

- **`check_diff.py` decides what gets tested.** A change that adds a package
  directory, moves source between packages, or adds a new dependency edge may
  need matching changes here, or the new code will silently never be linted or
  tested.
- **The Python matrix is `["3.11", "3.14"]` only**, while all packages declare
  support for 3.11 through 3.14. Syntax or behavior that breaks only on
  3.12–3.13 passes CI. Flag constructs whose behavior differs across those
  versions even though CI is green.

`_lint.yml` runs `uv lock --check` on every touched package, so a stale or
missing lockfile always fails CI. Copilot code review cannot see `uv.lock` (it
is in the excluded-file list) and excluded files are omitted from the changed-
file list the reviewer receives, so the reviewer can neither confirm nor deny
that a lockfile was regenerated. Do not infer a missing lockfile from its
absence in your file list, and do not raise it as a finding: CI is the
authority here.

## Package independence

Each `libs/*` package versions, releases, and pins independently. Do not
propose sharing code across package boundaries: duplication between packages is
a deliberate choice here, not an oversight.

Cross-package coupling that *is* real and worth flagging: two packages whose
extras pull incompatible versions of the same third-party dependency cannot be
co-installed. `langchain-azure-compute` and `langchain-azure-dynamic-sessions`
already have this problem with their `deepagents` floors.

Version bumps belong in the package's own `pyproject.toml`, and the README
changelog for that package is where the change is announced.

Reserve `**[Breaking change]:**` in a changelog for API removals, renames,
signature changes, and behavior changes on a still-supported runtime. Raising
the minimum supported Python version is a support-policy change, not a breaking
change, and ships as a patch release — do not ask for a breaking marker on one.

## Dependency changes

- A new runtime dependency needs justification; a new *required* dependency on
  a package that previously worked without it is a breaking change for
  installed users, and belongs behind an extra.
- Widening a version range needs evidence that the wider range actually works;
  narrowing one can break existing environments.
- Dependency edits must be reflected in `uv.lock`, but this is enforced by
  `uv lock --check` in CI and is not observable in review — do not flag it.

## PR hygiene

- Titles follow Conventional Commits: `type(scope): description`, for example
  `feat(azure-ai): add streaming support`. The scope is the package.
- A new feature should come with a usage example under `samples/` when it is
  user-facing. `samples/` code is read by users as a template, so review it for
  correctness and for credential handling — a sample that hardcodes a key
  teaches every reader to do the same.
- Public behavior changes need README or docstring updates in the same PR.

## Instruction files

`AGENTS.md` at the repository root is loaded automatically; per-package
instructions live at `libs/<pkg>/AGENTS.md` or
`libs/<pkg>/.github/copilot-instructions.md`. When a PR adds or edits one,
check it against the package's actual tooling — two of the existing ones have
already drifted from the code they describe.
