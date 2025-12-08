---
description: Apply coding guidelines to Python source files
applyTo: tests/**/*.py
---

Follow the general [coding guidelines](../copilot-instructions.md) for this
project.

When making changes to the Python test code, ensure that you:

- Check `tests/**/conftest.py` files for available fixtures and utilities, and
  reuse them to avoid code duplication and inconsistencies,
- Create module-level utility functions when writing the same test logic in
  both sync and async tests, and reuse them to avoid inconsistencies in different
  test code paths,
- Propose running the newly created tests, and,
- When/if running the newly created tests, try using
  `uv run pytest -k '<filter_conditions>'` to select only the tests you want to
  run.
