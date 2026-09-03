---
name: packet-intake
description: Check a mortgage packet manifest for completeness and write a page-aware packet index.
---

# Packet Intake

Read `/source/packet-manifest.json` and compare the available documents with its expected
document list.

Write only `/output/01-packet-index.json` with:

- `packet_id`
- `documents`, with one entry per available document containing its file name and page range
- `missing_documents`

Use only facts present in the manifest. Keep the output concise and valid JSON.