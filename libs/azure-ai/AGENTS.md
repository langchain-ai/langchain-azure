# langchain-azure-ai Instructions

Apply these instructions to `libs/azure-ai/` in addition to the [repository instructions](../../AGENTS.md).

## Code Quality

- Keep implementations concise, focused, and self-documenting through descriptive names, precise type hints, and straightforward control flow. Preserve correctness and clarity when simplifying code.
- Reuse existing helpers and upstream abstractions; add new abstractions only when they solve a concrete need.
- Keep directory and module organization consistent with existing conventions. Avoid excessively long files; split code by cohesive responsibilities when it improves readability and navigation, without unnecessary fragmentation.
- Keep internal comments sparse. Explain non-obvious rationale or constraints rather than narrating what the code does.

## Change Scope and Reviewability

- Keep feature PRs focused and reviewable; exclude unrelated refactors, cleanup, and pre-existing bug fixes unless the feature requires them.
- Compare suspected issues against the PR base before reporting them; pre-existing issues are non-blocking unless the feature requires a fix.
- Prefer consistent behavior across existing and new feature paths. If alignment expands scope, document temporary differences with TODOs for a follow-up PR. Accept deferral of unrelated fixes or alignment when the feature works correctly without regressions.
- When scope analysis shows a feature is too large for one reviewable PR, plan independently reviewable, working increments, each with its required tests and documentation. Record the plan and a TODO checklist in the relevant design or feature documentation, clearly separating the current PR's scope from deferred work.
- Before opening a PR, consider using an available AI tool to review the proposed changes for issues. This is optional and does not replace human review.
- After pushing new commits to any PR, review its title and description; update them as needed to describe the current scope, behavior, and validation results for reviewers, and remove outdated details from earlier revisions.

## Public APIs and Documentation

- Treat public APIs as downstream compatibility contracts. Preserve existing import paths, signatures, defaults, return types, and documented behavior; prefer backward-compatible changes.
- Before exposing or changing an API, work through downstream usage, edge cases, error handling, sync/async behavior, and compatibility with upstream contracts. Cover affected public behavior with tests.
- Obtain explicit approval before intentional breaking changes, and provide a deprecation or migration path.
- Every public API, class, and method must have complete Google-style docstrings. Document purpose, parameters and defaults, return values, raised exceptions, and important behavior or constraints as applicable; include usage examples where helpful.

## Upstream Alignment and Compatibility

- Align class design, naming, interfaces, and extension patterns with the corresponding LangChain or LangGraph abstractions. Prefer supported upstream extension points over private implementation details.
- Ensure extensions work across the Python versions required by the repository instructions and the upstream dependency ranges declared in [pyproject.toml](pyproject.toml). Check feature availability before relying on version-specific APIs or syntax.
- Consult the upstream documentation and relevant implementations when designing integrations or when API contracts or compatibility are uncertain. Verify behavior against the supported versions, not only the latest release.

## Agent Hosting

- Treat performance as a critical requirement when changing hosting code. Avoid unnecessary or duplicate reads and writes to storage backends, checkpoint stores, response stores, and other persistence layers. Reuse data already available within the same request or execution when safe, without weakening durability, consistency, or isolation guarantees.

When working on agent hosting, consult these upstream SDK implementations as needed:

- [azure-ai-agentserver-core](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/agentserver/azure-ai-agentserver-core)
- [azure-ai-agentserver-responses](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/agentserver/azure-ai-agentserver-responses)
- [azure-ai-agentserver-invocations](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/agentserver/azure-ai-agentserver-invocations)

## Bug Fixes

1. Add a focused regression test that reproduces the bug using the existing test suite and conventions.
2. Run it and confirm it fails for the expected reason before changing the implementation.
3. Make the smallest corrective change, then rerun the regression test and relevant existing tests until they pass.

## Instruction Maintenance

- When you discover outdated guidance in an applicable `AGENTS.md`, verify it against the current codebase and authoritative documentation, then update or remove the stale content as part of the task.

## References

- [LangChain documentation](https://docs.langchain.com/build-overview)
- [LangChain source](https://github.com/langchain-ai/langchain)
- [LangGraph source](https://github.com/langchain-ai/langgraph)
- [Published langchain-azure-ai package on PyPI](https://pypi.org/project/langchain-azure-ai/)
