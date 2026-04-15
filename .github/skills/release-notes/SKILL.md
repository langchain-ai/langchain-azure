---
name: release-notes
description: >-
  Skill for compiling and writing release notes for langchain-azure packages.
  Use when a new version of a package is being released and the README.md
  changelog section needs to be updated with a summary of merged PRs.
user-invocable: true
---

# Compiling Release Notes for langchain-azure Packages

Each package maintains a `## Changelog` section in its `README.md`. When a new version is released, update that section with a concise, user-friendly summary of what changed.

## Required policy

- ALWAYS append release notes as a new version entry at the top of the changelog.
- NEVER modify the content of previously published version entries.

## How to compile release notes

1. **Identify the two release tags** you are comparing. Tags follow the pattern `<package>==<version>`, for example `langchain-azure-ai==1.2.1`. List available tags with:

   ```bash
   gh release list --repo langchain-ai/langchain-azure
   # or
   git tag --list | grep langchain-azure-ai
   ```

2. **List the PRs merged between the two tags** using the GitHub CLI:

   ```bash
   # Get the commits between the two tags and look for merge commits
   git log <old-tag>..<new-tag> --oneline --merges

   # Or use the GitHub compare API to get a full list of commits
   gh api repos/langchain-ai/langchain-azure/compare/<old-tag>...<new-tag> \
     --jq '.commits[].commit.message'
   ```

   You can also browse the merged PRs on GitHub directly by filtering by merge date range:

   ```
   https://github.com/langchain-ai/langchain-azure/pulls?q=is%3Apr+is%3Amerged+merged%3A<start-date>..<end-date>+label%3A<package-label>
   ```

3. **Write the release notes**. For each meaningful PR (skip internal/CI-only changes), add a bullet using friendly, user-facing language. Follow these conventions:
   - Start sentences with "We" to keep a consistent, team-friendly voice (e.g., "We fixed a problem when loading class X").
   - Use past tense ("We introduced", "We fixed", "We improved").
   - Mark breaking changes with `**[Breaking change]:**`.
   - Mark new features with `**[NEW]**` when the addition is significant.
   - Reference the PR number with a link, e.g., `[#123](https://github.com/langchain-ai/langchain-azure/pull/123)`.

4. **Update the package `README.md`**. Add the new version entry at the top of the `## Changelog` section:

   ```markdown
   ## Changelog

   - **<new-version>**:

       - We fixed an issue with X. [#123](https://github.com/langchain-ai/langchain-azure/pull/123)
       - We introduced support for Y. [#124](https://github.com/langchain-ai/langchain-azure/pull/124)

   - **<previous-version>**:
       ...
   ```

   Do not edit existing entries for previously released versions; only append a new section for the release being prepared.

   Each package's `README.md` is the source of truth for its changelog. The files to update are:

   | Package | README location |
   |---------|----------------|
   | `langchain-azure-ai` | `libs/azure-ai/README.md` |
   | `langchain-azure-dynamic-sessions` | `libs/azure-dynamic-sessions/README.md` |
   | `langchain-sqlserver` | `libs/sqlserver/README.md` |
   | `langchain-azure-storage` | `libs/azure-storage/README.md` |
   | `langchain-azure-postgresql` | `libs/azure-postgresql/README.md` |
   | `langchain-azure-cosmosdb` | `libs/azure-cosmosdb/README.md` |
