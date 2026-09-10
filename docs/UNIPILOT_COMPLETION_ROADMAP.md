# UniPilot — 15-feature completion roadmap

PHASE52 / Academic OS Stage 4. PHASE49–51 evidence remains historical and unchanged.

Overall Stage 4: **PARTIAL**. Generic deterministic GPA is Beta; degree-rule/credit audit is Foundation using explicit fictional fixtures only. Report/Research retain shared evidence, claim trace, exact-span checks, editable outlines and literature provenance. Local Demo tests pass; Live and manual authenticated Preview QA remain NOT_TESTED (protected login; current API health/OPTIONS timeout). Bounded exact-origin CORS code is preserved, not deployed. No feature is Complete. Foundation Base remains incomplete; no research model promotion.

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
| 9 | 時間割・出席・課題管理 | Not started | User-owned schedule/attendance/tasks; timezone/deadline correctness; edit/export/delete; no silent calendar writes | Consent, persistence, timezone model; optional authorized calendar | Design local data schema and explicit-change confirmations |
| 10 | 教授メールAI | Foundation | Student-approved recipient/purpose/tone/content; no invented commitments; preview/copy and explicit send approval | Existing Campus tool cards; quality rubric; optional mail connector | Validate existing template/copy path on real student examples; sending remains OFF |
| 11 | 学習記憶 | Not started | Consented, editable, deletable learning records with bounded retention; correction, export and privacy evaluation | Durable storage; identity/consent; learning event schema | Separate factual learning history from temporary conversation/session settings |
| 12 | AI学習計画 | Foundation | Plan from real availability/goals/history; feasible workload, revisions and uncertainty; measured user adherence, no personal optimization claim without evidence | Exam inputs; feature 11; deterministic constraints | Validate initial exam-plan prompt; design constraint solver interface |
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

## Remaining dependencies for features 7–15

| Features | Stage4 dependency / next milestone |
|---|---|
| 7 credit audit, 8 GPA | Implemented generic engines and fictional fixtures; independent real-policy provenance and authorized student Live workflow are still required |
| 9 schedule, 11 memory | Establish consent, ownership, bounded retention, export and deletion before durable personal-data storage; tab drafts are not memory |
| 10 professor mail | Keep preview/copy and explicit send approval; tie factual assertions to student input, not inferred commitments |
| 12 plan, 13 office hours | Reuse bounded Tutor/Materials/Exam inputs and source traces; validate feasibility and instructor-escalation criteria |
| 14 official search | Add authorized official-source provenance/freshness before promoting metadata to Verified |
| 15 careers | Define an opt-in skill→coursework/evidence schema; no fabricated achievements or external submissions |
