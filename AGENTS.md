# Repository Instructions

The SDKs under `libs/` are maintained independently by different teams.

- Before working in `libs/<sdk>/`, read and follow `libs/<sdk>/AGENTS.md` and any more specific `AGENTS.md` files along the target path.
- For changes spanning multiple SDKs, read each affected SDK's instructions and keep its conventions scoped to that SDK.
- All SDKs must support Python 3.10 through 3.14 (inclusive).
- When contributing a new feature, consider adding a usage example in an appropriate location under `samples/`.
- Pull request titles must follow Conventional Commits format: `type(scope): description` (scope optional), e.g. `feat(azure-ai): add streaming support`.
- For work in `samples/`, consult the `AGENTS.md` of each SDK the sample uses.
- If an applicable `AGENTS.md` is missing, follow the SDK's existing local instructions and documentation.
- Keep the root `README.md` concise and free of redundant information. Clearly present the repository's components, their responsibilities, documentation links, and quick-start code samples.
