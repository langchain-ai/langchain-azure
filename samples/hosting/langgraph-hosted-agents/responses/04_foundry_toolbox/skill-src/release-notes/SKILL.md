---
name: release-notes
description: Turn raw commits, tickets, or change summaries into structured customer-facing release notes with highlights, breaking changes, and upgrade steps.
---

# Customer-facing release notes

Use this procedure when asked to draft or organize release notes from raw
engineering changes.

1. Group every provided change under one of these sections, omitting empty
   sections:
   - **Highlights**: user-visible features and improvements.
   - **Fixes**: resolved bugs and reliability improvements.
   - **Breaking Changes**: removed or incompatible behavior, with the required
     action for callers.
2. Rewrite internal commit jargon into clear, customer-facing impact statements.
   Keep exact API names, flags, version numbers, and defaults unchanged.
3. If any change requires a configuration, dependency, or code update, add an
   **Upgrade Steps** section with short numbered steps.
4. Add an **Open Questions** section when an input is ambiguous or missing
   customer impact (for example, an unclear default or migration note). Never
   invent dates, benchmarks, issue numbers, or behavior that was not provided.

Keep bullets concise and ordered by customer impact.
