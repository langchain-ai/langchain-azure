---
name: azure-integration-docs
description: 'Maintain Azure integration documentation in the LangChain docs repo. Use when given a langchain-azure namespace, module path, or GitHub URL and you need to identify the integration building block, read the folder-local TEMPLATE.mdx for that exact docs folder, create or update the matching page under src/oss/python/integrations, structure the page around the Azure product surface when appropriate, fill it from verified source code and docstrings, and update Azure discovery pages such as the component index and Microsoft provider page.'
argument-hint: 'Provide a langchain-azure namespace path, folder path, or GitHub URL.'
---

# Azure integration docs

## When to use

- Add or update documentation for Azure integrations maintained in the LangChain ecosystem.
- Convert a `langchain-azure` namespace or GitHub URL into the correct docs page under `src/oss/python/integrations`.
- Keep Azure docs aligned with the product surface users recognize, such as Microsoft Foundry, Azure AI Content Safety, Azure Storage, or Azure Cosmos DB, instead of forcing a class-by-class layout when the package represents one cohesive product area.
- Keep Azure provider pages, component index pages, and usage examples aligned with new or changed Azure integrations.

This skill is optimized for Azure-related packages currently maintained in `langchain-ai/langchain-azure`, including:

- `azure-ai`: Microsoft Foundry
- `azure-dynamic-sessions`: Azure Dynamic Sessions
- `azure-storage`: Azure Storage
- `cosmosdb`: Azure CosmosDB

## Inputs

Expected inputs usually look like one of these:

- A GitHub URL such as `https://github.com/langchain-ai/langchain-azure/tree/main/libs/azure-ai/langchain_azure_ai/agents/middleware`
- A repo path such as `libs/azure-ai/langchain_azure_ai/chat_models`
- A Python namespace such as `langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql`

## Workflow

### 1. Normalize the source path

Extract these parts from the input:

- The package root under `libs/<provider>/`
- The import package such as `langchain_azure_ai`
- The component path such as `chat_models`, `embeddings`, `vectorstores`, `tools`, or `agents/middleware`

If the input is only a URL, strip the repository prefix and work from the path after `tree/main/`.

### 2. Map the namespace to a LangChain docs building block

Use the module path to choose the docs folder.

| Source path pattern | Docs building block | Target folder |
| --- | --- | --- |
| `agents/middleware`, `middleware` | middleware | `src/oss/python/integrations/middleware/` |
| `chat_models`, `chat` | chat | `src/oss/python/integrations/chat/` |
| `embeddings` | embeddings | `src/oss/python/integrations/embeddings/` |
| `vectorstores`, `vector_stores` | vectorstores | `src/oss/python/integrations/vectorstores/` |
| `tools`, `toolkits`, `agent_toolkits` | tools | `src/oss/python/integrations/tools/` |
| `retrievers` | retrievers | `src/oss/python/integrations/retrievers/` |
| `document_loaders` | document_loaders | `src/oss/python/integrations/document_loaders/` |

If no mapping is clear, stop and ask which building block the namespace belongs to.

### 3. Find the template before writing

Look for `src/oss/python/integrations/<building-block>/TEMPLATE.mdx`.

This is the authoritative template for that exact integration type. Different building blocks such as `chat`, `middleware`, `tools`, and `vectorstores` can have different required structures, section names, table formats, and expectations.

Rules:

- Always read the `TEMPLATE.mdx` in the same folder as the target page before drafting or editing.
- Never reuse the structure of a different building block just because it looks similar.
- Never assume a generic integration format if a folder-local `TEMPLATE.mdx` exists.
- When updating an existing page, compare the current page against the folder-local template and normalize it toward that template unless the folder already has a deliberate stronger pattern.

- If the template exists, use it as the starting point.
- If the template does not exist, inspect an existing page in the same folder and use that as the canonical pattern. Prefer the ones that start with `azure` in the file name.
- If neither a template nor a strong sibling example exists, stop and ask whether to create the page style first.

Known template-backed folders in this repo currently include:

- `chat`
- `embeddings`
- `middleware`
- `retrievers`
- `vectorstores`

### 4. Choose whether to create or update a page

Search the target folder for an existing Azure page before creating anything.

- If a page already exists for that integration, update it.
- If the provider already has one page per building block, extend that page instead of splitting it.
- If there is no page yet, create a new snake_case file name that matches the integration name used elsewhere in the folder.

Before editing an existing page, read:

- The current target page
- The folder-local `TEMPLATE.mdx`
- At least one strong sibling page in the same folder

Then decide whether the page should:

- Be lightly corrected to match the template more closely
- Be substantially restructured to match the template
- Preserve a deliberate folder-specific pattern that is stronger than the raw template
- Preserve or extend a product-oriented Azure page structure when the current page already groups several public classes under one Azure product or capability area

Use existing file names as the source of truth. Examples already in this repo include:

- `src/oss/python/integrations/chat/azure_ai.mdx`
- `src/oss/python/integrations/tools/azure_dynamic_sessions.mdx`
- `src/oss/python/integrations/document_loaders/azure_blob_storage.mdx`
- `src/oss/python/integrations/vectorstores/azure_cosmos_db_no_sql.mdx`

### 5. Gather source-of-truth content from code

Read the public modules and exported classes in the source package.

Prioritize these sources in order:

1. Class docstrings and function docstrings
2. Public exports in `__init__.py`
3. Constructor signatures and default values in the source code
4. Package README files and package-local docs or notebooks
5. Tests that confirm import paths, default behavior, and supported parameters
6. Inline examples in the package source
7. Existing provider landing pages in this docs repo for cross-links and naming consistency

Capture:

- Primary class names and import paths
- Package name for installation
- Required credentials or environment variables
- Supported capabilities and limitations
- Default parameter values and notable non-default behavior
- Short code snippets that reflect the public API

For classes that contain a `project_endpoint` and `endpoint` parameter, state that you can use one or the other,
but `project_endpoint` is recommended for better defaults and future compatibility. Explain that most of the time,
those parameters can be inferred from environment variables so they don't need to be indicated on each instantiation.

Accuracy rules:

- Verify every documented class name against a public export path, not only against an internal module.
- Verify every documented default value against the actual signature or tests.
- Verify every environment variable name against source code or README text.
- Verify every installation command against the package name used by the source repo.
- If examples in tests, README, and docstrings disagree, prefer the public export path and current constructor signature, then mention uncertainty only if it cannot be resolved.
- Do not infer JS support, serialization, or advanced features unless they are explicitly documented or clearly implemented.

Do not invent features that are not stated or implied by the public API or docstrings.

### 6. Fill the page from the template

Replace placeholders with real values and keep the page specific to the chosen building block.

Apply these rules:

- Treat the folder-local `TEMPLATE.mdx` as a checklist, not inspiration. Every placeholder section must be either filled correctly or removed intentionally.
- Match the title and description style already used in the folder.
- Use the real package name in install commands.
- Remove the credentials section when the integration does not require credentials.
- If one page documents multiple Azure integrations for the same building block, add a short summary table near the top and give each integration its own section.
- Use first-party terminology consistently, for example `Azure AI Foundry` rather than stale names unless the source package still uses the old identifier as part of its API.
- Prefer relative internal links such as `/oss/integrations/chat/azure_ai`.
- Use `@[` API links only when they fit existing repo conventions for that page.
- Keep section ordering aligned with the template unless the folder's strongest existing examples consistently use a better structure.
- If a sibling page in the same folder demonstrates a stronger pattern than the template, follow that folder pattern deliberately and preserve template-required content.

When the target page documents multiple classes in one file:

- Keep the template's top-level scaffolding such as overview, setup, installation, instantiation, and agent usage.
- Add a summary table near the top that lists the documented classes and what each one does.
- Give each class its own subsection with API reference, configuration highlights, and an example.

### 6a. Prefer product-oriented structure when it matches the Azure surface

The folder-local template remains mandatory, but do not mechanically mirror it when the clearer presentation is product-first.

Use this product-oriented layout when one page documents several classes that together form one Azure product experience, for example Microsoft Foundry middleware backed by Azure AI Content Safety:

- Keep the template-required top-level sections: intro, overview, setup, installation, instantiation, agent usage, and final API reference.
- Use the intro and setup sections to explain the Azure product surface first, including the package, endpoint choices, credential model, and where the integration fits in LangChain.
- In the overview table, prefer user-facing middleware or capability names with descriptions instead of exposing only raw class names when that better matches how the product is organized.
- After the shared setup and agent usage sections, group the detailed content by Azure product capability area, for example `Azure AI Content Safety`, then place the specific middleware subsections inside that group.
- Within each capability section, keep the concrete class name, import path, configuration options, and example code explicit and verified.
- Use first-party Azure product terminology consistently. Prefer names such as `Microsoft Foundry`, `Azure AI Foundry`, and `Azure AI Content Safety` when those are the concepts users need to navigate the docs.
- Preserve class-by-class organization only when the product framing would hide important distinctions or when the folder's established examples clearly favor class-first pages.

This means the template governs required content and section coverage, while the strongest Azure page pattern governs presentation and grouping.

### 7. Update Azure discovery surfaces

When you add a new page or materially expand an existing one, update the places that help users find it.

Usually this means:

- `src/oss/python/integrations/<building-block>/index.mdx`
- `src/oss/python/integrations/providers/microsoft.mdx`

Update both the table entry and the card list when the index page has both.

`src/docs.json` normally does not list each individual integration page directly for these sections, but if a navigation entry is actually required for the change you are making, update it as well.

### 8. Validate before finishing

Run the repo checks that make sense for the change.

- Verify frontmatter is present and the `description` field contains no markdown.
- Verify the final page still follows the correct folder-local `TEMPLATE.mdx` shape.
- Check internal links and example imports.
- Make sure the example code matches the current public API.
- Test code examples before shipping them when feasible.
- Re-check that headings, setup instructions, and tables match the conventions of the target folder rather than another integration type.

Recommended validations:

- `make lint`
- `make broken-links`

If full validation is too expensive, explain what was and was not checked.

## Branching rules

- If the namespace exposes several public classes in the same building block, prefer one provider page with multiple sections over many tiny pages.
- If the code is mostly a thin wrapper around another Azure package, document the LangChain-facing API, not the upstream SDK in general.
- If the docs repo already has a broader Microsoft page covering the same feature, keep that page aligned but do not duplicate full setup instructions unless the component page needs them.
- If the existing Azure page is already organized around a coherent product surface, keep that organization and use the template as a completeness checklist rather than flattening the page back into a generic per-class template.
- If docstrings are missing critical information such as auth model or installation package, stop and ask rather than guessing.
- If the current page conflicts with the folder-local `TEMPLATE.mdx`, treat the template as correct unless multiple strong sibling pages in that same folder prove a newer pattern.
- If a parameter default differs between docstring prose and the actual constructor signature, trust the signature and use tests as a tie-breaker.
- If a class is public in an internal subpackage but not re-exported from the documented public namespace, document the public import path users should actually use.

For Azure middleware pages specifically:

- Prefer one Microsoft Foundry or Azure AI page with shared setup and grouped capability sections over several tiny class pages when the classes ship together and share the same package, credentials, and product story.
- Keep shared setup material near the top and avoid repeating installation, credentials, or endpoint explanations in every middleware subsection.
- If the page uses product labels in the overview table, keep the corresponding subsection headings aligned with those labels so users can scan from the table into the details.

## Completion checklist

- The correct building block was identified from the namespace.
- The folder-local `TEMPLATE.mdx` for that exact building block was read and followed.
- The target page was created or updated in the right folder.
- The page content was derived from the public API and docstrings.
- The page uses the folder-local template as a completeness checklist, even if the presentation is product-oriented.
- Class names, defaults, env vars, and install commands were verified against code.
- Placeholder text from the template was removed.
- Relevant Azure index pages were updated.
- `providers/microsoft.mdx` was updated when discoverability changed.
- Validation status was reported clearly.

## Example invocation

`/azure-integration-docs https://github.com/langchain-ai/langchain-azure/tree/main/libs/azure-ai/langchain_azure_ai/agents/middleware`

For that example, the expected first conclusion is:

- Building block: `middleware`
- Template: `src/oss/python/integrations/middleware/TEMPLATE.mdx`
- Likely target page: `src/oss/python/integrations/middleware/azure_ai.mdx`

Then inspect the middleware classes under that namespace and decide whether the page should document one middleware or several Azure AI middleware exports.
