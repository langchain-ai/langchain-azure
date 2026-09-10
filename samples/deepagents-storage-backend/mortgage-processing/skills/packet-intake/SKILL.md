---
name: packet-intake
description: Check a mortgage packet manifest and write /output/01-packet-index.json.
---

# Packet Intake

Read `/source/packet-manifest.json` and compare the available documents with its expected
document list.

This skill produces exactly one artifact: `/output/01-packet-index.json`. Use that exact
path; do not rename it or create a subdirectory.

The artifact contains:

- `packet_id`
- `documents`, with one entry per available document containing its file name and page range
- `missing_documents`

Use only facts present in the manifest. Keep the output concise and valid JSON.