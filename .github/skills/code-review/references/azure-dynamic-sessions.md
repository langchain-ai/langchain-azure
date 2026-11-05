# `langchain-azure-dynamic-sessions` (`libs/azure-dynamic-sessions`)

**Deprecated. It will not receive further fixes.** It has been superseded by
`langchain-azure-compute`, whose `dynamic-sessions` extra provides the same
functionality.

Review consequences:

- New features do not belong here. If a PR adds capability to this package,
  the finding is that it should go to `libs/azure-compute` instead — say so
  once, with a link, rather than reviewing the implementation in depth.
- Acceptable changes are security fixes, packaging fixes, and documentation
  that points users to the replacement.
- Deprecation warnings come from the local `langchain_azure_dynamic_sessions._api.base`
  decorators, not `langchain_core._api`. `tests/unit_tests/test_deprecation.py`
  asserts the warning behavior, so warning text and category are tested
  contract, not incidental.
- The public names (`SessionsPythonREPLTool`, `SessionsBashBackend` and the
  `backends` / `tools` subpackages) must keep working for existing users.
- This package's `deepagents` extra and `langchain-azure-compute`'s cannot be
  co-installed while their version floors differ. A dependency change here can
  therefore break installs of the *other* package; check the pins on both
  sides before approving one.

Where a change does need review, the underlying dynamic-sessions data-plane
behavior — 4,096-byte output cap, dropped final line, the flat `/mnt/data`
store — is documented in [azure-compute.md](azure-compute.md) and applies
equally.
