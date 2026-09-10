# Citation Trace — Stage 3 Foundation

Report and Research share `EvidenceLedger`. The ledger is user-supplied data, not a source-verification service. Report, Research and Citation remain Foundation, not Complete.

## Data and trust

- Evidence: stable ID, title, author/organization, date/year, source type, optional URL/DOI/page, pasted source text, exact supporting span, student note and status.
- Claim: student-authored text and zero or more evidence IDs. No valid links means **Unsupported**. Having links does not prove support.
- Literature note: evidence ID (metadata and original span), separate summary, explicit Student/AI provenance and related research question. An AI summary is never substituted for original evidence.
- Outline: arbitrary editable section titles and student notes; no forced essay structure or automatic full-report writing.
- Requirements: student-entered conditions with required/unconfirmed/confirmed state. Missing professor conditions are never inferred.

| Status | Meaning in this client |
|---|---|
| Unverified | Empty/new entry; no check performed |
| User supplied | Student edited data; previous checks invalidated |
| Span confirmed | Nonblank candidate exists as an exact JavaScript substring of the pasted text |
| Needs verification | Explicit span check failed |
| Verified | Reserved for future independently authenticated verification; never assigned or trusted from client storage |

Substring checks are case/whitespace/Unicode-sensitive and do not normalize text. They demonstrate only that the candidate appears in what the student pasted, not that the source is authentic, the quote is accurate to an external document, or the claim follows from it. Any edit invalidates prior status. Restored Span confirmed is rechecked; restored Verified is downgraded. No URL is fetched, no DOI is resolved, and untrusted URLs are rendered as plain text rather than clickable/executable markup.

Bibliography preview concatenates supplied metadata only. Missing title, author/organization, year/date and source type are listed explicitly. URL, DOI and page remain optional and empty unless entered. This is an APA-like preview, not a verified/style-complete citation.

## Persistence and deletion

Use the existing per-tab `sessionStorage` policy, with explicit Save. Keys are `unipilot-report-draft-v2` and `unipilot-research-draft-v2`. Legacy v1 field-only drafts load when v2 is absent; old keys are left intact until explicit Clear. Clear asks for confirmation and removes both versions for that workspace only. There is no API submission, external AI call or server persistence. Copy exports the structured user-authored draft for manual backup. Browser storage limits/errors are surfaced. Closing the tab can discard drafts.

Stored JSON is treated as untrusted: object/array/string shapes are checked, lists bounded to40 entries, strings to20000 characters, duplicate IDs removed, dangling links removed and client trust promotions rejected. Deleting evidence removes claim links and unsets literature references but preserves the student's summary. An unsupported claim stays visible for review.

## Scope and next gate

React and CSS only; no editor dependency. Tests cover add/edit/delete, exact/nonmatching span, missing metadata, forged Verified status, trace links, provenance, draft migration and Clear. Demo tests establish client behavior, not actual academic correctness or live model quality. Next: authorized source retrieval with provenance/license/page offsets, independent metadata verification, adversarial citation fixtures and student workflow evaluation before any status promotion.
