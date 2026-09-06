# UniPilot Academic OS roadmap

PHASE49 / TRACK B. Web capabilities and Foundation model capabilities are separate.
The Foundation Base is incomplete. Tutor UI does not imply reliable mastery of every subject.

| Stage | Scope | Current status |
|---|---|---|
| 1 | Foundation research + Academic Command Center + Tutor UI | UI implemented; model research remains gated |
| 2 | Tutor specialization, subject evaluations, structured teaching responses | Coming next |
| 3 | Report Assistant: requirements, research, outline, draft, sources, review | Input shell only |
| 4 | Research Workspace: question, methods, evidence, analysis, discussion | Input shell only |
| 5 | Citation Engine with original-source verification | Architecture only |
| 6 | University-specific RAG and authorized tools | Not connected |
| 7 | Academic AI OS with integrated academic workflows | Long-term direction |

## Implemented in Stage 1

- `/`: Ask, workspace navigation, existing UniPilot chat transport and tool cards.
- `/study`: 14 subjects, three explanation levels, five learning methods; safe explicit prompt augmentation to the existing API. Text responses remain text; no invented sections, equations, or citations.
- `/report`, `/research`: fields and a visible future workflow; save drafts only in this tab's session storage, or copy them. Nothing is submitted from these shells. No auto-writing.
- `/sources`: metadata returned in the current in-memory conversation; no fabricated papers, authors, DOI, URLs, or page numbers. Hard reload clears the in-memory conversation; closing the tab clears session drafts.
- Existing `/settings`, `/developer`, and all Campus review routes remain available and unchanged.

## Citation Engine: future boundary

Claim → authorized source retrieval → original verification → evidence span → citation → bibliography audit.

Each future citation must retain the original source, retrieval timestamp, precise supporting span, claim association, publication/license metadata, and verification result. Retrieval alone is not verification. Unknown metadata remains unknown. Missing evidence blocks a verified citation; never fill missing fields by plausible generation.

`SourceInspector` currently displays title, publisher, safe HTTPS URL, license, last_verified_at, confidence and stale. Explicit `verified: true` plus URL and verification date may display API-confirmed status; timestamp/confidence alone cannot. This is not an independent frontend verification of the claim.

## Integration and safety

`NEXT_PUBLIC_API_URL` is retained. `/chat/stream` uses newline-delimited JSON snapshots; non-OK/unavailable stream startup falls back to `/chat` with an identical body including session_id and response_mode. Partial-stream failures do not silently replay a possibly processed request. Users can explicitly retry. A 90-second timeout prevents indefinite loading.

External LLM APIs remain OFF. No OpenAI/Gemini/Claude connectors, paper web search, PDF pipeline, university RAG, or AI-optimized learning history were added. No production deployment. Research and Web changes must remain separate commits; Web builds never run alongside GPU training.

## Validation

- `npm run lint` uses the repository's existing `tsc --noEmit` script.
- `npm run build` uses the existing Next webpack build.
- Node 24+: `node --test tests/academic.test.mjs` from `web/`.
- Browser QA: start the local production build on port 3049, then `node tests/browser-qa.cjs`. Set `UNIPILOT_PLAYWRIGHT_MODULE` to an already installed Playwright module when it is not on Node's module path. No new application dependency is needed.
- API fixtures are explicitly Demo and intercepted locally. QA sends no production API requests and never mutates real model state, evaluations, or the system clipboard.
- See `web/qa/results.json` and screenshots for viewport/interaction evidence. Mocked integration establishes the client contract, not live Render availability or model answer quality.
