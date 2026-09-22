# Cloud multimodal E2E — 2026-09-22

Real deployed Foundry Responses endpoint; same agent code and fixtures, baseline SDK in version 1 and combined fixed SDK in version 2. Requests explicitly specify agent version; sessions are bound to that version. No synthetic cloud responses.

| Check | Before (v1) | After (v2) | Modes |
|---|---|---|---|
| Agent receives inline PNG + PDF | 0 attachments | 2 attachments, exact normalized block SHA-256 matches | JSON + SSE |
| Client receives native tool result | Text only | Text + image + file, exact attachment payload equality | JSON + SSE |
| Second-turn input history | 0 attachments | Both attachments preserved | JSON + SSE |
| Second-turn tool history | 0 attachments | Both attachments preserved | JSON + SSE |
| gpt-4.1 reads codes only present in attachments | MISSING | MAPLE-7319 and ORCHID-4827 | JSON + SSE |
| Private tool artifact | Not exported | Not exported | JSON + SSE |

Baseline: 9365b59b6256386aaa245180897ee881580bd7d8. Combined build: e5b8b84af9d1e68c7043b7582207eeb0c9763bcd; PR #1072 ab63b71678eac1a1456adcc12b21e977e7c8f965 + PR #1073 09f0c1683289ba64047f73004242e92c1eddf9c1. Wheel hashes are in transport summaries. Python 3.13, Responses SDK 2.1.0b2.

Transport verification uses deterministic graph inspection and a real ToolNode/Command tool result. Real-model verification separately sends the inline PNG/PDF to gpt-4.1. Tests check data, not Inspector previews. Cloud history uses previous_response_id without a graph checkpointer.

Limits: URL/file-ID authorization, other formats/models, cloud checkpoint persistence, arbitrary artifacts, and UI rendering were not tested. The test found no additional attachment-preservation bug on these paths.

Harness observations: azd invoke forced SSE, so final evidence uses direct protocol HTTP requests for both modes. Pin both agent_session_id and agent_reference.version for an unambiguous version comparison; a session-only request can execute the old build while returning the default version label. This routing-label observation is outside the two adapter fixes.
