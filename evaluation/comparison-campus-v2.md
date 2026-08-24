# UniPilot Campus v2 evaluation

Campus v2 is an opt-in candidate. The production v0.4 model, Render/Vercel settings, checkpoint, and Campus v1 files are unchanged. No external AI/API was used.

## Frozen evaluation policy

- Router selection: `data/campus_v2/router/dev.json` only.
- Final blind: `data/campus_v2/blind/evaluation-2000.json`, SHA256 `c599d8a3af026d24ddf075d62070fdbf2d405e0872a0c680e49248788bb7e996`.
- Normalized train/blind exact overlap: 0.
- Blind distribution: colloquial 500, normal 500, ambiguous 300, compound 400, hard 300.
- The blind result did not trigger any Router retraining or threshold tuning.

## Campus v1 vs Campus v2 on the new blind 2,000

| Metric | Campus v1 | Campus v2 | Target |
|---|---:|---:|---:|
| Category accuracy (all) | 42.60% | 85.15% | 92% |
| Category accuracy (1,700 routable items) | 45.88% | 99.47% | reference |
| Action accuracy | 34.00% | 93.25% | 92% |
| Correctness | 42.60% | 85.15% | 88% |
| Relevance | 42.60% | 84.95% | 90% |
| Hallucination | 0.00% | 0.00% | <=1% |
| Completion | 100.00% | 100.00% | 99% |
| Natural Japanese | 100.00% | 100.00% | 99% |
| Actionable score | 3.581 | 4.652 | 4.2 |
| Clarification accuracy | 0.00% | 90.00% | reference |
| Multi-intent recall | 38.50% | 99.63% | reference |
| Mean latency | 7.25 ms | 6.93 ms | reference |
| P95 total latency | 13.02 ms | 18.14 ms | reference |

The all-item Category score includes the 300 intentionally ambiguous prompts. Campus v2's primary failure is that 288/300 ambiguous prompts do not use `general` as the category, even though 270/300 correctly choose `CLARIFY`. The all-item production gate remains failed; the routable-only score is shown only for diagnosis.

## Router methods on development data

| Method | Category accuracy | P95 |
|---|---:|---:|
| Rules | 49.52% | 0.04 ms |
| BM25 | 99.52% | 10.06 ms |
| Character centroid TF-IDF | 99.29% | 0.39 ms |
| Character n-gram LinearSVM | 100.00% | 0.46 ms |
| Word n-gram LinearSVM | 27.14% | 0.40 ms |
| Logistic regression (word n-gram) | 27.14% | 0.50 ms |
| Naive Bayes (word n-gram) | 27.14% | 0.63 ms |
| Hierarchical character SVM | 100.00% | 1.30 ms |
| Hierarchical Hybrid | 100.00% | 1.87 ms |

The Hierarchical Hybrid was frozen because it tied for best Category accuracy while adding Level1/Level2, top-2, confidence, action selection, and multi-intent output. Japanese text without a morphological tokenizer explains the weak word n-gram methods. The separate negation adversarial accuracy is 77.00%; therefore no production promotion is permitted.

## Retrieval, tools, validation, and resources

- FAQ audit: all 1,000 rows automatically reviewed; 940 scored 5/5, 60 scored 4/5, 0 below 4 after fixes. Human-reviewed count remains 0.
- Retrieval (category-level relevant FAQ set, exact duplicates excluded): Recall@1 75.47%, Recall@3 77.59%, MRR 0.765.
- Tool audit: 22/22 deterministic cases passed (16 existing tools plus 6 university-neutral calculators).
- Validator v2 synthetic labeled set: invalid-detection precision 100%, recall 100%, 500 cases. This is not a substitute for human evaluation.
- Router P95: 5.00 ms. FAQ P95: 18.37 ms. Tool P95: 18.51 ms.
- Process RSS after Campus v2 initialization: 336.11 MB; observed peak 338.36 MB.
- Human evaluation: 100 items prepared (Easy/Medium/Hard/Compound, 25 each), 0 completed. ChatGPT/Gemini answers must be pasted manually in `/campus-v2-eval`; no external API is connected.

## Decision

**STOP — not production eligible.** Category accuracy, Correctness, Relevance, adversarial accuracy, and the unfinished human evaluation fail the gate. Production remains v0.4; no push or deployment is authorized.

