# Academic OS v1 architecture — Stage 8 / PHASE56

This is a code-level integration audit, not a claim of validated model quality, real institutional data or production readiness. Status definitions remain those in `UNIPILOT_COMPLETION_ROADMAP.md`; no feature is Complete. Feature15 is Foundation. Feature13/14 remain Foundation. All fifteen features are represented by sixteen routes (Ask is the shared home).

## Dependencies and responsibility map

| Feature / route | Status | Current dependencies and data flow | Responsibility / boundary |
|---|---|---|---|
| 1 Tutor `/study`, Ask `/` | Beta | ChatWorkspace → academic/learning prompt builders → chat stream/fallback → UniPilot API → SessionContext/messages | AI-assisted; model claims require independent verification; no external AI API |
| 2 Materials `/materials` | Foundation | Student excerpt → bounded learning prompt → same chat API | AI-assisted using supplied material; local retrieval/extractors are interfaces, not running RAG |
| 3 Exam `/exam` | Foundation | Student dates/scope → calendar checks → bounded prompt → API | Deterministic countdown, AI-assisted suggestions; no invented timetable |
| 4 Report `/report` | Foundation | PlanningWorkspace → writing/evidence → tab-local explicit Save | Local user-data and structured drafting; no automatic report generation |
| 5 Research `/research` | Foundation | Same writing/evidence engine, distinct tab key and field schema | Local user-data, literature summary/original separation; no invented research results |
| 6 Citation `/sources` + writing ledger | Foundation | API Source metadata → SourceInspector; supplied text → exact-span EvidenceLedger; official fixture → evidence adapter | Source/citation layer, not independent authenticity verification; Span confirmed ≠ Verified |
| 7 Degree `/degree` | Foundation | User credits + fictional versioned DegreeRule → deterministic degree engine | Deterministic only, no real official-data integration or graduation guarantee |
| 8 GPA `/gpa` | Beta | User grades + explicit generic policy → gpa engine → explicit tab Save/export | Deterministic only, never model arithmetic; official policy not established |
| 9 Planner `/planner` | Beta | Local user records → planner timezone/deadline/attendance engine → explicit local Save | Local user-data + deterministic only; saved records feed Today |
| 10 Professor Email `/email` | Beta | Explicit student facts → professorEmail template → editable preview/copy | Deterministic template, no professor impersonation, sending OFF |
| 11 Memory `/memory` | Beta | Student confidence/provenance/confirmed weakness → learningMemory → explicit local Save | Local user-data only; no inferred weakness, no server sync |
| 12 Study Plan `/plan` | Beta | Explicit selected Memory/Planner copies + capacity → studyPlan day scheduler → explicit Save → Today | Deterministic only, not AI optimization; originals remain independent |
| 13 Office Hours `/office-hours` | Foundation | Student question/excerpt → officeHours five-mode template | Local unsaved draft, no model or retrieval; quoted student material is not institutional verification |
| 14 Official `/official-search` | Foundation | Query/year → application-owned fictional registry → exact-domain/freshness/year/conflict checks → evidence metadata | External official-data boundary is **NOT_IMPLEMENTED live**; demo fixture is not real verified university information |
| 15 Career `/career` | Foundation | Student profile/facts → career schema → evidence-linked skills / PR / ES / interview templates | Local user-data, deterministic drafting; company retrieval NOT_IMPLEMENTED, application sending OFF |

```text
AcademicShell: shared navigation / health status / in-memory messages
  ├─ Tutor + Materials + Exam ─ explicit submit ─ UniPilot API
  │                                            └─ server session state (not device-local)
  ├─ Report + Research ─ writing + EvidenceLedger ─ explicit tab Save
  │                               ↑ metadata adapter
  ├─ Official fixture registry ─ source/year/freshness contract
  ├─ GPA + Degree ─ deterministic engines (no AI arithmetic)
  ├─ Planner ─ explicit Save ─┐
  ├─ Memory  ─ explicit Save ─┼─ opt-in + selection ─ Study Plan / Career copies
  │                          └─ Planner + saved Plan ─ Today (read-only)
  └─ Email + Office Hours + Career ─ labelled templates (no model submission)
```

## Layer boundaries

- **AI layer:** only explicit chat submissions through `web/lib/chat.ts` for Ask/Tutor/Materials/Exam. Streaming snapshots, session identity and startup fallback stay compatible. A partially received stream is not replayed automatically. A UniPilot-owned model hosted on a remote API is not device-local processing. Research checkpoints are not promoted into this API.
- **Retrieval layer:** Materials ingestion/chunk/retriever contracts in `learning.ts` describe future services only. Office Hours uses pasted excerpts. Official Search uses local fictional fixtures. Career has no corporate-source retrieval. These must not be described as live Retrieval-grounded services.
- **Source layer:** `academic.ts` handles safe source URLs and supplied statuses; `evidence.ts` handles student-provided exact spans/claim trace; `officialSources.ts` handles controlled-registry provenance rules. A timestamp or client-restored verified flag is not authenticity evidence. Real verification needs authoritative acquisition, source ownership, academic year, freshness, conflict resolution and auditable spans.
- **Local storage:** separate versioned keys, validation, explicit Save/Clear/export; no background sync. Study Plan/Career imports are copies with provenance, not ownership transfer or live links. Career never consumes Planner grades or marks registered coursework as completed. See `ACADEMIC_OS_PRIVACY_BOUNDARIES.md`.
- **Future server persistence:** not implemented for features9–15. Any future sync requires authenticated record ownership, explicit per-feature opt-in, purpose/retention/export/deletion policies, access control, concurrency/versioning and data-minimization tests. A browser session identifier alone must not be treated as authentication.

## Shared-module and duplication audit

| Area | Existing shared service / duplication | Stage8 decision / next action |
|---|---|---|
| Date arithmetic | learning countdown, planner timezone conversion, studyPlan integer-day scheduling, official source dates | Related but different calendar semantics; leave engines unchanged. Future common validated date-only primitive needs timezone/DST regressions before migration |
| Storage/export | Feature-specific schemas/keys and similar Save/Clear/Blob code in GPA, Planner, Memory, Plan, Career; tab save in writing | Deliberate ownership separation retained. Do not replace with a generic auto-save store. Future small export utility is possible after error/consent tests |
| Source verification | academic source display, evidence exact-span, official registry verification | Not interchangeable truth claims; no indiscriminate status merge. Shared evidence adapter retained |
| Evidence Ledger | Report/Research share EvidenceLedger; Career facts reference student statements rather than scholarly citation schema | Avoid duplicate scholarly verification. Career truth is explicitly unverified; future adapter must preserve that downgrade |
| Prompt framing | academic Tutor + learning Materials/Exam → shared chat transport | Preserve bounded student-data framing. Email/Office/Career intentionally never call model prompt transport |
| API status | One ApiStatus mounted by AcademicShell | No per-feature health pollers added. Online means a health response, not model quality or live source verification |
| Navigation | One NAV + four NAV_GROUPS | Desktop groups; mobile five primary links + keyboard-operable all-feature menu. All16 routes remain reachable without a sixteen-column bottom bar |

No broad refactor, model/architecture change, authentication bypass, API schema change or production deployment is part of this audit. Academic Constellation styling is retained.

## Safety ownership and regression mapping

| Failure | Guard / regression suite |
|---|---|
| Fake university facts | Degree fixture labels, official provenance/domain/year/staleness/conflict; degree/officialSources unit and calculators/office-official browser suites |
| Fake citations | Unknown/stale remain unverified; exact source span and restoration downgrade; academic/evidence tests and browser evidence workflows |
| Fake GPA math | Pure weighted-policy arithmetic and infeasible target handling; gpa tests and populated calculator QA |
| Fake deadlines | User-supplied calendar, invalid date and timezone handling; learning/planner/studyPlan tests and planner-email/memory-plan QA |
| Fake professor claims | Student-fact templates, missing-field placeholders, copy-only; professorEmail/officeHours tests and browser regressions |
| Fake career achievements | Whitelisted profile, evidence references, no generated result, selected-only copies; career tests and career-integration QA |
| Accidental cross-feature import/save | Separate keys, source snapshots unchanged after Career Save/export/Clear; plan opt-in regressions; no POST in local-only workflows |

Demo browser QA uses isolated fictional records and mocked health/chat where needed; it does not contact production. The full unit/pytest/lint/build and five-width browser results are recorded in `PHASE56_HANDOFF.md` after completion. Live authenticated Preview remains NOT_TESTED; real-user quality/fairness/accessibility evaluation remains an explicit next gate, not implied by a route existing.
