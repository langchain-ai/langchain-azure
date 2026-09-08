---
name: document-classification
description: Classify mortgage documents and write /output/02-classification.json.
---

# Document Classification

Read `/source/packet-manifest.json` and classify every listed source file.

This skill produces exactly one artifact: `/output/02-classification.json`. Use that exact
path; do not rename it or create a subdirectory.

The artifact contains:

- `packet_id`
- `documents`, with one entry per source file containing `file`, `document_type`, and
  `confidence`

Use the manifest and file names as evidence. Keep the output concise and valid JSON.