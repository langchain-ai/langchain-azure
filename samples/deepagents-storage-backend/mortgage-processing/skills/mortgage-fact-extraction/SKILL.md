---
name: mortgage-fact-extraction
description: Extract mortgage facts and write /output/03-extracted-facts.json.
---

# Mortgage Fact Extraction

Read these files, preferably in parallel:

- `/source/loan-application.json`
- `/source/income-verification.txt`
- `/source/bank-assets.csv`
- `/source/property-appraisal.md`

This skill produces exactly one artifact: `/output/03-extracted-facts.json`. Use that exact
path; do not rename it or create a subdirectory.

The artifact includes:

- `packet_id`
- declared and verified income
- monthly debt
- requested loan amount
- purchase price and down payment
- latest liquid assets
- appraised value

Include a `/source/` path for every fact. Do not infer values that are not explicitly
supported by the source files. Keep the output valid JSON.