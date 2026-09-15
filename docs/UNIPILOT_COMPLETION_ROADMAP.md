# UniPilot — 15-feature completion roadmap

PHASE54 / Academic OS Stage 6. PHASE49–53 evidence remains historical and unchanged.

Overall Stage 6: **PARTIAL**. Consent-based Learning Memory and deterministic Study Plan join Planner, template-only Professor Email and generic GPA as Local Beta. Today's UniPilot also reads saved study blocks. Degree-rule/credit audit remains Foundation using explicit fictional fixtures. Report/Research retain evidence and claim traces. Live and manual authenticated Preview QA remain NOT_TESTED. Bounded exact-origin CORS code is preserved, not deployed. No feature is Complete. Foundation Base remains incomplete; no research model promotion.

## Status and completion gate

Not started = no integrated feature. Foundation = initial boundary/workflow, incomplete integrations. Beta = usable experimental workflow, quality/safety still being validated. Validated = documented functional, quality and safety evaluations plus real-user workflow evidence. Complete = all four gates pass for the stated supported scope, known limitations are documented and regression coverage/operational ownership exist. UI presence alone never advances a feature to Complete.

The definition of done in every row requires: **F** functional tests; **Q** representative quality evaluation with a preregistered acceptance threshold; **S** safety/privacy evaluation including failure paths; **U** a real student completing the workflow on the live authorized environment. Demo fixtures satisfy only the tested client-contract portion of F, not Q/S/U or actual model competence.

## Feature inventory

| # | Feature | Status | Definition of done (plus F/Q/S/U above) | Dependencies | Next milestone |
|---|---|---|---|---|---|
| 1 | AI先生 / Tutor | Beta | 14 subjects × 3 explanation levels × 5 modes; retained session; incremental hints; answer diagnosis; subject-specific accuracy and refusal/uncertainty rubric; no false correctness guarantee | Reliable local model; live API; math/tool verification | Test live Tutor sessions and preregister subject quality/verification evaluations |
| 2 | 講義資料から学習 | Foundation | Text/Markdown ingestion, bounded context, material-grounded answers with exact spans; PDF/PPTX/DOCX extraction quality and injection/unsupported-claim tests for each declared format | Ingestion, chunker, local retrieval, citation spans | Authorized live short-excerpt evaluation; implement one extractor with page/span tests |
| 3 | 試験対策 | Foundation | User-entered subject/date/scope/time; deterministic countdown; realistic plan/quiz/review; invalid dates and unavailable time handled; student can revise a plan | Tutor; reliable calendar arithmetic; live API; future feature 12 | Live exam workflow and workload/quiz quality rubric; no invented calendar events |
| 4 | レポートWorkspace | Foundation | Requirements → outline → evidence-backed draft → review with attribution and academic-integrity controls; no unsourced auto-writing | Shared EvidenceLedger; authorized retrieval; writing rubric | Validate Stage3 checklist/outline/claim links with students; no full-report auto-writing |
| 5 | 卒論・卒研Workspace | Foundation | Research question/method/results/discussion tied to supplied evidence; no invented experiments, data or findings | EvidenceLedger; Literature Notes provenance; reproducible analysis | Validate summary versus original-span separation and question/method/limitations workflow with students |
| 6 | Citation Engine | Foundation | Verify actual source, claim, author/title/DOI/page/URL and supporting span; bibliography audit; missing/stale evidence cannot become verified | Stage3 exact-substring checker and claim trace; future authorized retrieval and provenance | Independent external-source authenticity/metadata verification; current Span confirmed is not Verified |
| 7 | 履修・単位AI | Foundation | Degree-rule/version-aware credit audit with official provenance, exception handling and adviser escalation | Versioned DegreeRule schema and deterministic TEST UNIVERSITY fixtures; future official rules ingestion and transcript consent | Independently verify one real university/year rule set; missing source stays Unknown and conflicting versions never merge |
| 8 | GPA・成績シミュレーター | Beta | Correct weighted GPA and what-if calculations including retakes/exclusions/rounding; no fabricated grades | Generic GradingPolicy engine; user-entered grades; local Save/Clear/export | Authorized student workflow and independently verified university policies; generic Beta is not official-policy validation |
| 9 | 時間割・出席・課題管理 | Beta | User-owned schedule/attendance/tasks; timezone/deadline correctness; edit/export/delete; no silent calendar writes | Tested local Planner schema, Intl timezone handling, explicit Save and opt-in attendance threshold | Authorized student workflow; semester/holiday and notification design before integrations |
| 10 | 教授メールAI | Beta | Student-approved recipient/purpose/tone/content; no invented commitments; preview/copy and explicit send approval | Deterministic Template draft engine and missing-fact tests; no active model/mail API | Student quality/utility review; keep sending OFF and do not present templates as model-generated |
| 11 | 学習記憶 | Beta | Explicit Save, view/edit/delete/clear/export; student confidence and source provenance; no silent save or AI weakness labels | Local consent-based records, manually confirmed weaknesses, schema validation; no server sync | Authorized student privacy/workflow review; optional Tutor/Exam entry flow; see LEARNING_MEMORY.md |
| 12 | AI学習計画 | Beta | Integer duration/capacity constraints, priority/deadline order, reschedule and opt-in source copies; no personalized optimization claim | Local deterministic day scheduler and persistence, Memory/Planner opt-in, saved-only Today | Student feasibility review and separate clock-time scheduling if needed; see STUDY_PLAN_ENGINE.md |
| 13 | AIオフィスアワー | Not started | Course-scoped tutoring dialogue with escalation to instructor, source grounding and limits | Tutor; course materials; safe teacher handoff | Define one supported office-hour scenario and escalation criteria |
| 14 | 大学公式情報検索 | Foundation | University/year-specific official evidence, freshness/license/provenance, conflict handling and refusal when evidence missing | Existing Campus official-source safety; authorized retrieval; Citation Engine | Audit existing source coverage; validate one university's current documents |
| 15 | 就活接続 | Not started | Consented mapping from real coursework/skills to career options; no invented achievements, biased ranking or unauthorized submissions | User-owned records; verified careers sources; privacy/fairness review | Define opt-in profile and evidence-linked skills schema |

Existing Campus prototypes (tool cards, advice and local knowledge) are acknowledged above. They do not establish completion of the integrated 15-feature OS. SessionStorage for Tutor settings is not durable learning memory. Source Inspector is display infrastructure, not an independent Citation Engine.

## Stage 2 scope and boundaries

- Tutor: explicit Teach me / Step by step / Hint only / Check my answer / Practice problem. Subject, topic, difficulty, explanation level and method persist in this tab. Student answers stay in current component state and are not saved as a learning profile. Check mode sends both problem and student answer. Math/science show a verification notice; tool verification is not implemented.
- Materials: paste-only text/Markdown and five actions. Up to 300 Unicode characters, additionally bounded by a **288 UTF-8-byte token upper bound for the entire constructed prompt**. This reserves 192 for the existing API chat framing (189 bytes) and 32 output tokens within 512. The narrower token bound wins; Japanese excerpts must be short. No hidden truncation. No PDF/PPTX/DOCX extractor, embedding API, automated quote validation or retrieval is running.
- Exam: no fake courses, dates or history. Countdown is local-calendar arithmetic, recalculated at submit and on focus/date refresh. The plan is a suggestion; personal optimization and calendar writes are not implemented. Same full-prompt context bound as Materials.
- Report/Research Stage3: explicit requirements/checklist, arbitrary outline, shared evidence, claim→span→metadata trace, bibliography missing-field display, research purpose/hypothesis/method/analysis/limitations and literature summary provenance. Explicit tab-local Save/Copy/Clear; v1 drafts remain readable. No generation or server-save requests. Source Inspector metadata behavior remains unchanged. See `CITATION_TRACE_ARCHITECTURE.md`.
- GPA Stage4: `/gpa`, weighted Current/What-if/Target calculations, impossible-target bound, explicit retake policies, separate display/policy rounding, tab-local Save/Clear and local JSON export. No LLM arithmetic or API submission. The future-credit target assumes additional counted credits, not future retake denominator replacement. See `GPA_ENGINE.md`.
- Degree Stage4: `/degree`, university/faculty/department/year/version/effective-period/source schema and deterministic total/category/remaining credits with course/group constraints. TEST UNIVERSITY is fictional and never Verified; missing provenance returns Unknown, conflicting versions require a choice. Inputs are component-memory only. No real university rule ingestion or graduation guarantee. See `DEGREE_RULE_SCHEMA.md`.
- Planner Stage5: `/planner`, explicit timetable/assignment/attendance records, IANA timezone, deterministic overlap/deadline/percentage rules, local Save/Clear/Import/Export. Today's UniPilot on Home derives next class, near/overdue tasks and opt-in attendance warnings from saved user records only. No fake schedules or institutional attendance rules. See `PLANNER_ENGINE.md`.
- Professor Email Stage5: `/email`, missing facts remain placeholders, seven presets/three tones, editable subject/body preview, stale-input notice and Copy only. Always labelled Template draft; AI/API path not enabled. See `PROFESSOR_EMAIL_SAFETY.md`.
- Existing chat body, response_mode, session_id, ToolCards, clarify options, incremental snapshots and startup-only fallback remain compatible. A partial-stream failure never silently replays the request.

## Future material architecture

`web/lib/learning.ts` declares DocumentIngestor, MaterialChunker, LocalEmbedder, LocalRetriever, SourceSpan and GroundedAnswer contracts only. The planned sequence is:

Ingestion → validated extracted text → token-budgeted chunks with Unicode offsets → local embedding/retrieval → verified source spans → grounded answer.

Every extractor needs source identifiers, format/license checks, bounded resource use, page/slide provenance and rejection of untrusted document instructions. Context must be recalculated for the actual model and retrieval wrapper; the current 192-token framing reserve covers the existing raw chat wrapper, not arbitrary future RAG context. Never enable an external embedding API implicitly.

## Live readiness and security

Read-only diagnostic identified the historical PHASE49 Preview through GitHub deployment 6296102205, not a new Stage4 deployment. The unique Preview URL redirects to Vercel login. On 2026-09-07, Render `/health` returned 200 and loaded=true; the exact Preview origin received no Access-Control-Allow-Origin and `/chat/stream` OPTIONS returned 400 (CORS FAIL). On the PHASE52 check at 2026-09-10T15:25:51Z (September11 JST), health and OPTIONS timed out, so current CORS is NOT_TESTED. Live authenticated app flow and compiled NEXT_PUBLIC_API_URL: NOT_TESTED. Expected API URL: `https://unipilot-mini.onrender.com`.

The deployment owner must test through an authorized Preview session and explicitly approve the exact origin in a bounded server-side/environment-driven allowlist. Do not switch to wildcard origins, disable Vercel protection or alter production settings for this phase. No Render/Vercel Production deployment was performed. Preview-ready code does not mean Live PASS.

API status only shows Online after a successful `/health` with status=ok and loaded=true. Connecting/Waking API indicate waiting, not proof of a cold start. Unavailable includes CORS, timeout, non-OK and invalid responses. Online describes a point-in-time health result, not model quality.

## Evidence and next milestone

Client tests: `web/tests/academic.test.mjs`, `learning.test.mjs`, `evidence.test.mjs`, `gpa.test.mjs`, `degree.test.mjs` (31 PASS), `browser-qa.cjs`, `calculators-qa.cjs`; framing contract: `web/tests/test_material_budget.py` (1 PASS). Repository pytest:501 PASS. Demo fixtures are separate from the parameterized `web/tests/live-preview-qa.py` GET/OPTIONS checks and manual authenticated Preview flow in `PREVIEW_LIVE_QA.md`. PHASE52 artifacts: `web/qa/phase52/`; prior phases remain unchanged. Screenshots use Demo fixtures, including displayed Online status. Stage4 passes45 nine-route checks plus20 populated responsive checks at360/390/768/1024/1440, retained feature1–6 interactions and GPA/Degree interactions. Keyboard focus, labels/error association, aria-live and44px targets were checked; this is not a full assistive-technology certification.

Next milestone: a separately approved deployment owner action for exact-origin CORS, then authenticated student workflow QA with preregistered quality/safety criteria. No broad auto-writing, external AI API, main merge, Production deploy or canonical checkpoint promotion is authorized here.

## Stage5 local evidence and remaining dependencies

PHASE53 adds14 Planner/Email unit cases and browser scenarios for persistence/import/export/Clear, overdue/Today, missing facts and editable Copy-only previews. Five responsive widths cover11 routes (55 checks) plus35 populated layouts. Feature1–8 client regressions remain in the QA suite. `web/qa/phase53/` is local Demo evidence, not authenticated Live quality. The new phase explicitly permits foundation-research push after local full QA even when Live is NOT_TESTED; Production remains prohibited.

| Features | Stage4 dependency / next milestone |
|---|---|
| 7 credit audit, 8 GPA | Implemented generic engines and fictional fixtures; independent real-policy provenance and authorized student Live workflow are still required |
| 9 schedule, 11 memory | Planner local records now have explicit Save/Clear/export/import; this does not implement adaptive learning memory, accounts or calendar sync |
| 10 professor mail | Local Template Beta implemented; independent student quality review and any model integration require further validation; sending stays OFF |
| 12 plan, 13 office hours | Reuse bounded Tutor/Materials/Exam inputs and source traces; validate feasibility and instructor-escalation criteria |
| 14 official search | Add authorized official-source provenance/freshness before promoting metadata to Verified |
| 15 careers | Define an opt-in skill→coursework/evidence schema; no fabricated achievements or external submissions |
