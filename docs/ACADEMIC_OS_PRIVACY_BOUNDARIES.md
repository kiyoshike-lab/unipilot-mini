# Academic OS Stage8 privacy boundaries

Code-level scope: PHASE56. Client local-only workflows are not sent to a model. Shared ApiStatus still issues `GET /health` on every route: this sends ordinary connection metadata but no feature form payload. Browser QA mocks that endpoint and checks for unexpected POST requests. This is not a production network/retention audit.

| Feature | Explicitly saved locally | API payload | Unsaved / sharing boundary |
|---|---|---|---|
| 9 Planner | `localStorage: unipilot-planner-v1` on Save/import; class, task, attendance and timezone fields | None | Current edits stay React memory until Save. Today reads saved records; Plan/Career only copy selected candidates after user action |
| 10 Professor Email | No app persistence; Copy is an explicit clipboard action | None; no mail sending | Inputs/editable template in React memory; copy may leave data in OS clipboard |
| 11 Learning Memory | `localStorage: unipilot-learning-memory-v1` on explicit Save; user confidence, provenance and confirmed weaknesses | None | Unsaved edits in React memory; no Tutor history ingestion or inferred weakness |
| 12 Study Plan | `localStorage: unipilot-study-plan-v1` on Save; selected input copies, plan blocks, statuses | None | Draft/recalculation unsaved; Memory/Planner candidates require opt-in and selection; saved Plan feeds Today |
| 13 Office Hours | None | None | Course/question/student excerpt/template in React memory only; no silent server persistence |
| 14 Official Search | No personal query persistence | None; no live retrieval | Local query + fictional registry results; opening an external link elsewhere is a separate explicit browser action |
| 15 Career | `localStorage: unipilot-career-v1` only on Save; manually entered profile and selected fact copies | None; no recruiting/company/model API | Candidate read only on button; selected-only import into unsaved draft; Export downloads current draft, Clear touches only Career |

LocalStorage is origin-scoped browser storage, not encrypted private cloud storage. It survives closing the browser, can be read by code running on that origin, and may be exposed on a shared device. Use Clear or browser site-data controls. Downloaded exports and clipboard contents are outside app Clear; users must manage them separately. No account sync, cross-device deletion or server backup is claimed.

Career Clear deletes its copied evidence, not the original Memory/Planner record. Conversely, deleting the original does not revoke an already saved Career copy. The explicit copy/source identifier makes this boundary visible; future linked-revocation or retention policy needs separate design. An imported Planner record is not proof of attendance, credits, grade or successful completion.

## Features1–8 and exceptions that must not be hidden

- Ask/Tutor/Materials/Exam: explicit prompt submit sends student input and session ID to the configured UniPilot API (`NEXT_PUBLIC_API_URL`, with a localhost fallback). A remotely hosted UniPilot model still receives the data remotely; External AI API OFF does not mean no network.
- Session ID is automatically stored in sessionStorage. Tutor settings are also automatically stored in sessionStorage, unlike explicit Save in Memory. Student answers are current UI state and are not automatically made into a learning profile.
- The existing `CampusSessionStore` / V2 maintains parsed context in server-process RAM (bounded256 sessions; includes fields such as university, grade, subject, dates and tasks depending on pipeline). It is not a durable user profile, nor an authenticated ownership model. Hosting/proxy logs and deployed retention are not audited here. Do not promise that all API data is unsaved.
- Report/Research: explicit tab-local Save with independent keys and evidence/claim drafts. GPA: explicit tab-local Save. Degree: component memory only. None submits these form values to an API.
- Sources displays existing response metadata and safe explicit links; it does not independently retrieve or verify scholarly sources.

## Future server persistence gate

Require an explicit user choice and purpose, authenticated authorization per record, least-data payload, disclosure of hosting/retention, encryption/access review, export and deletion including backups, audit trails, and tests rejecting accidental feature mixing. Current local schemas must not be silently migrated to a server. Student data must never become training data by default. No such persistence was added in PHASE56.
