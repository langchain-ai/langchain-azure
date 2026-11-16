# Hosted Agent Samples

Applies to this directory and its descendants. Follow the [repository](../../../AGENTS.md)
and [azure-ai](../../../libs/azure-ai/AGENTS.md) instructions, plus those of other
SDKs used. Apply these rules to new samples and changed parts of existing samples.

## Sample Code

- Read the target README and [shared README](README.md) before editing.
- Give each sample one clear goal. Use native framework and hosting SDK features;
  add custom wrappers or host subclasses only when the goal requires them.
- Follow `responses/NN_name/` or `invocations/NN_name/`, use an unused number, and
  add new samples to the shared README's index.
- Include an entrypoint, dependencies, `.env.example`, Dockerfile, and deployment
  configuration for the supported setup path. Keep names and defaults consistent
  across code, configuration, and documentation. Declare required versions and extras.
- Keep credentials out of committed files. Label simulated tools and data, and
  preserve input validation, error handling, and resource cleanup.
- Keep blocking I/O off the async event loop; use async SDK calls or
  `asyncio.to_thread` for necessary synchronous calls.

## README

- Write in plain, concise English. Open with what the sample demonstrates, when
  to use it, and the expected outcome.
- Explain only what the reader needs for that goal. Omit unrelated topics or link
  to another sample or document. Link to shared setup, deployment, and tracing
  instructions instead of repeating them; state sample-specific differences.
- Include required resources, versions, authentication, permissions, and variables,
  including cloud requirements for local runs. Distinguish developer and agent access.
- Provide runnable client code with commands, inputs, and expected results. A
  request snippet is enough for a simple call; use a script for stateful flows
  such as approval/resume, stream reconnection, or artifact downloads.
- Keep snippets consistent with the code. Explain relevant limits, persistence,
  content exposure, and cleanup beside the affected steps. Keep review history
  and development plans in the PR description.

## Telemetry

- Enable telemetry by default in every sample, locally and when hosted. Include
  tracing dependencies, initialize instrumentation at startup, and configure export
  through the [shared tracing setup](README.md#tracing).
- Document exporter and content-recording defaults. Telemetry enablement does not
  require enabling content capture.

## Build and Startup

- Install dependencies before copying application code to preserve Docker layer caching.
- Use `--no-cache-dir` for pip installs. Pip generates dependency `.pyc` files by
  default; avoid redundant dependency-wide `compileall` steps.
- With uv, run `compileall` at build time for dependencies, the standard library,
  and application code using the runtime interpreter and final paths. Dependency
  and standard-library compilation is best-effort; application compilation must
  fail the build on syntax errors and exclude `.venv`.
- Exclude local environments, caches, and secrets from the build context with `.dockerignore`.

Example uv Dockerfile fragment, assuming uv is installed:

```dockerfile
WORKDIR /app/user_agent
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
RUN PYTHONDONTWRITEBYTECODE= .venv/bin/python -m compileall -q \
    $(.venv/bin/python -c "import sysconfig as s; print(s.get_paths()['stdlib'], s.get_paths()['purelib'])") || true; \
    PYTHONDONTWRITEBYTECODE= .venv/bin/python -m compileall -q -x '[.]venv' .
```

## Validation

- For documentation changes, check affected commands, configuration, and links.
  For runtime or dependency changes, exercise the entrypoint and client flow in a
  clean environment with the documented versions and extras; test new sample logic.
- When changing telemetry or image builds, check default tracing initialization,
  dependency-layer order, and bytecode in the resulting image as applicable.
- Keep repository-file checks in the [docs/samples suite](../../../CONTRIBUTING.md#docs-and-samples-tests).
  Report actual results and distinguish source inspection, offline/mock checks,
  and live Foundry validation.
