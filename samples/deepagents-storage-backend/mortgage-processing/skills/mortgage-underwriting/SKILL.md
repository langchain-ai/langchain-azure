---
name: mortgage-underwriting
description: Apply packet evidence and underwriting policy to produce a cited mortgage decision.
---

# Mortgage Underwriting

Read these files, preferably in parallel:

- `/output/03-extracted-facts.json`
- `/output/01-packet-index.json`
- `/source/underwriting-policy.md`

Calculate loan-to-value using the lower of purchase price or appraised value. Calculate
debt-to-income using verified monthly income. Verify assets and identify missing documents.

Write only `/output/04-underwriting-decision.md` with concise sections named:

- Decision
- Calculations
- Conditions
- Evidence

Cite `/source/` paths for factual evidence. Apply policy from the packet rather than
embedding policy assumptions in the decision.