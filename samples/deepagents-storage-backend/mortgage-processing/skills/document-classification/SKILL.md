---
name: document-classification
description: Classify every file in a mortgage packet and record confidence for each classification.
---

# Document Classification

Read `/source/packet-manifest.json` and classify every listed source file.

Write only `/output/02-classification.json` with:

- `packet_id`
- `documents`, with one entry per source file containing `file`, `document_type`, and
  `confidence`

Use the manifest and file names as evidence. Keep the output concise and valid JSON.