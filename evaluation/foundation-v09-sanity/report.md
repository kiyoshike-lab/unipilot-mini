# UniPilot Foundation v0.9 — PHASE 21 (1–8)

Campus v2.3 / production v0.4を固定し、新モデル別系列だけを検証した。外部LLM/APIは未使用。

## Data

- Current inventory: 60,910 rows
- v0.9 structured training: 13,214 rows (Base 1,040 / Campus 330 / Instruction 11,844)
- Human approved: 4; RAG-only current/institutional sources: 114
- Exact duplicate / holdout semantic overlap excluded: 0 / 0
- Legacy low-quality template rows excluded: 50,000
- Final Blind 1,000: sealed and unopened

## Model and sanity training

- Mini: 19,814,784 params / vocab 512 / context 256
- Standard candidate: 46,755,840 params / vocab 4096 / context 1024
- 100step: train loss 7.3870, validation loss 7.4012, natural continuation 0/8, Gate FAIL
- 500step: not executed because the 100step gate failed

## Validation 200

| Axis | Campus v2.3 | Standard step100 | Delta |
|---|---:|---:|---:|
| correctness | 93.00% | 20.00% | -73.00pt |
| relevance | 91.70% | 20.00% | -71.70pt |
| completeness | 90.74% | 10.00% | -80.74pt |
| specificity | 89.56% | 10.00% | -79.56pt |
| naturalness | 93.00% | 0.00% | -93.00pt |
| actionable | 92.98% | 10.00% | -82.98pt |

- Campus: critical 0, first response 0.070s, peak RSS 514.8MB
- Standard: critical generation failures 200, first token 0.116s, 56.44 tok/s, peak RSS 499.4MB

## Decision

Standard継続: **NO**。このcheckpointの次stepはなし。まず高品質なライセンス済みBase本文を増やし、短いsanityで日本語成立を再確認する。
