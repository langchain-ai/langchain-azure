---
name: mortgage-fact-extraction
description: Extract supported financial and property facts from a mortgage packet with source paths.
---

# Mortgage Fact Extraction

Read these files, preferably in parallel:

- `/source/loan-application.json`
- `/source/income-verification.txt`
- `/source/bank-assets.csv`
- `/source/property-appraisal.md`

Write only `/output/03-extracted-facts.json`. Include:

- `packet_id`
- declared and verified income
- monthly debt
- requested loan amount
- purchase price and down payment
- latest liquid assets
- appraised value

Include a `/source/` path for every fact. Do not infer values that are not explicitly
supported by the source files. Keep the output valid JSON.