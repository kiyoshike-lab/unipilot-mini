# UniPilot — 15-feature completion roadmap

PHASE50 / Academic OS Stage 2. This roadmap supersedes Stage 2's “Coming next” label in the historical PHASE49 roadmap; PHASE49 evidence remains unchanged.

Overall Stage 2: **PARTIAL**. Local implementation and Demo contract tests pass; authenticated Preview integration is blocked by Vercel protection and the exact Preview origin is rejected by Render CORS. No feature is Complete. Foundation Base remains incomplete, and the research model is not promoted to production.

## Status and completion gate

Not started = no integrated feature. Foundation = initial boundary/workflow, incomplete integrations. Beta = usable experimental workflow, quality/safety still being validated. Validated = documented functional, quality and safety evaluations plus real-user workflow evidence. Complete = all four gates pass for the stated supported scope, known limitations are documented and regression coverage/operational ownership exist. UI presence alone never advances a feature to Complete.

The definition of done in every row requires: **F** functional tests; **Q** representative quality evaluation with a preregistered acceptance threshold; **S** safety/privacy evaluation including failure paths; **U** a real student completing the workflow on the live authorized environment. Demo fixtures satisfy only the tested client-contract portion of F, not Q/S/U or actual model competence.

## Feature inventory

| # | Feature | Status | Definition of done (plus F/Q/S/U above) | Dependencies | Next milestone |
|---|---|---|---|---|---|
| 1 | AI先生 / Tutor | Beta | 14 subjects × 3 explanation levels × 5 modes; retained session; incremental hints; answer diagnosis; subject-specific accuracy and refusal/uncertainty rubric; no false correctness guarantee | Reliable local model; live API; math/tool verification | Test live Tutor sessions and preregister subject quality/verification evaluations |
| 2 | 講義資料から学習 | Foundation | Text/Markdown ingestion, bounded context, material-grounded answers with exact spans; PDF/PPTX/DOCX extraction quality and injection/unsupported-claim tests for each declared format | Ingestion, chunker, local retrieval, citation spans | Authorized live short-excerpt evaluation; implement one extractor with page/span tests |
| 3 | 試験対策 | Foundation | User-entered subject/date/scope/time; deterministic countdown; realistic plan/quiz/review; invalid dates and unavailable time handled; student can revise a plan | Tutor; reliable calendar arithmetic; live API; future feature 12 | Live exam workflow and workload/quiz quality rubric; no invented calendar events |
| 4 | レポートWorkspace | Foundation | Requirements → outline → evidence-backed draft → review with attribution and academic-integrity controls; no unsourced auto-writing | Citation Engine; authorized research retrieval; writing rubric | Preserve current tab draft shell; add evidence ledger before AI drafting |
| 5 | 卒論・卒研Workspace | Foundation | Research question/method/results/discussion tied to supplied evidence; no invented experiments, data or findings | Citation Engine; research retrieval; reproducible analysis interfaces | Validate research-plan shell with students; add traceable literature notes |
| 6 | Citation Engine | Foundation | Verify actual source, claim, author/title/DOI/page/URL and supporting span; bibliography audit; missing/stale evidence cannot become verified | Authorized retrieval; source metadata/license; span verification | Implement a source-span verifier and adversarial fake-citation fixtures |
| 7 | 履修・単位AI | Not started | Degree-rule/version-aware credit audit with official provenance, exception handling and adviser escalation | University rules ingestion; structured transcript consent | Define one university/year schema and deterministic rule fixtures |
| 8 | GPA・成績シミュレーター | Not started | Correct weighted GPA and what-if calculations including retakes/exclusions/rounding; no fabricated grades | University grading policies; user-provided grades | Define calculator contract and edge-case tests before integrated UI |
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
- Report/Research remain tab-local draft shells. They do not submit generation requests. Source Inspector retains publisher/license/last_verified_at/confidence/stale and treats absent evidence as unknown, not verified.
- Existing chat body, response_mode, session_id, ToolCards, clarify options, incremental snapshots and startup-only fallback remain compatible. A partial-stream failure never silently replays the request.

## Future material architecture

`web/lib/learning.ts` declares DocumentIngestor, MaterialChunker, LocalEmbedder, LocalRetriever, SourceSpan and GroundedAnswer contracts only. The planned sequence is:

Ingestion → validated extracted text → token-budgeted chunks with Unicode offsets → local embedding/retrieval → verified source spans → grounded answer.

Every extractor needs source identifiers, format/license checks, bounded resource use, page/slide provenance and rejection of untrusted document instructions. Context must be recalculated for the actual model and retrieval wrapper; the current 192-token framing reserve covers the existing raw chat wrapper, not arbitrary future RAG context. Never enable an external embedding API implicitly.

## Live readiness and security

Read-only diagnostic identified the PHASE49 Preview through GitHub deployment 6296102205. The unique Preview URL redirects to Vercel login. On 2026-09-07, Render `/health` returned 200 and loaded=true; the exact Preview origin received no Access-Control-Allow-Origin and `/chat/stream` OPTIONS returned 400 (CORS FAIL). On the final 2026-09-10 JST attempt, health and OPTIONS timed out, so current CORS is NOT_TESTED. Live authenticated app flow and compiled NEXT_PUBLIC_API_URL: NOT_TESTED. Expected API URL: `https://unipilot-mini.onrender.com`.

The deployment owner must test through an authorized Preview session and explicitly approve the exact origin in a bounded server-side/environment-driven allowlist. Do not switch to wildcard origins, disable Vercel protection or alter production settings for this phase. No Render/Vercel Production deployment was performed. Preview-ready code does not mean Live PASS.

API status only shows Online after a successful `/health` with status=ok and loaded=true. Connecting/Waking API indicate waiting, not proof of a cold start. Unavailable includes CORS, timeout, non-OK and invalid responses. Online describes a point-in-time health result, not model quality.

## Evidence and next milestone

Client tests: `web/tests/academic.test.mjs`, `learning.test.mjs`, `browser-qa.cjs`; framing contract: `web/tests/test_material_budget.py`. Demo fixtures are separate from `web/tests/live-preview-qa.py` read-only live checks. Artifacts: `web/qa/phase50/`. All screenshots in that directory use Demo API fixtures, including displayed Online status.

Next milestone: unblock authorized Preview + bounded CORS, then run real Tutor/material/exam workflows with a preregistered quality and safety rubric. Only after that evidence should the statuses progress toward Validated. Features 4–15 remain staged by dependencies; no broad auto-writing, external AI API, main merge, production deploy or canonical checkpoint promotion is authorized here.
