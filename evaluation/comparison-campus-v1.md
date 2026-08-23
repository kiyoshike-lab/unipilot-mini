# UniPilot Campus v1 evaluation

## Outcome

Campus v1 is implemented as an opt-in orchestration mode over the unchanged v0.4 model. Production v0.4, Render, Vercel, GitHub Release, and Standard v0.8 are unchanged. No external LLM or teacher model was used.

| Metric | v0.4 model only | v0.4 Grounded | v0.7 Grounded | Campus v1 |
|---|---:|---:|---:|---:|
| Correctness | 8.0% | 15.4% | 15.4% | 68.6% |
| Relevance | 13.3% | 30.1% | 30.1% | 68.8% |
| Category | 35.3% mapped | 35.3% mapped | 35.3% mapped | 68.8% |
| Hallucination proxy | 28.0% | 0.0% | 0.0% | 0.0% |
| Completion | 95.3% | 100.0% | 99.9% | 100.0% |
| Natural Japanese | 99.0% | 100.0% | 100.0% | 100.0% |
| Actionable Score | 1.198 | 2.724 | 2.723 | 3.681 |
| Human Score | unscored | unscored | unscored | unscored 100-item UI/JSON |
| Mean response | 341.5 ms | 28.3 ms | 32.2 ms | 30.6 ms |
| P95 response | 418.3 ms | 89.5 ms | 106.5 ms | 192.4 ms |
| Peak RSS | 315.23 MiB | 313.48 MiB | 314.05 MiB | 346.00 MiB |

## Campus routing and latency

- Routes: 465 tool, 410 FAQ, 104 model, and 21 safe university-policy answers.
- Tool latency: mean 4.20 ms, P95 5.56 ms.
- FAQ latency: mean 8.16 ms, P95 10.65 ms.
- Model-route total latency: mean 242.45 ms, P95 477.58 ms for the 32-token bounded benchmark.
- Router comparison: the separate development set selects BM25 at 99.43%, but its untouched blind accuracy falls to 59.1%. Rules reach 77.9%, TF-IDF 54.9%, and hybrid 68.3% on blind. Campus uses hybrid for high-precision deterministic-tool overrides, and records 68.8% end-to-end category accuracy.

## Decision

Campus is materially better than model-only and older Grounded variants, especially for instant calculations, complete email drafts, plans, cards, and university-policy refusal. It still does not satisfy the stated promotion thresholds. The gap is primarily routing/coverage: when Campus selects the correct category, 686 of 688 answers pass the automatic correctness proxy. Long-form open-ended reasoning and general knowledge remain weaker than large external models.

Do not restart Standard 50M training yet. First improve the router on a newly collected, human-labeled colloquial dataset without reusing this consumed blind set; review the 1,000 programmatically composed FAQ rows; complete the 100-question human comparison; and raise Actionable Score above 4.0. Resume 50M only if generation quality remains the largest error after routing and tool coverage pass.

**Promotion decision: REJECT.** No push or deployment is performed.
