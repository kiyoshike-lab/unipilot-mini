# Campus v2.3 Evaluation Report

新Blind 500・Stress 200は改善に使わなholdoutとして一度だけ評価した。外部LLM/APIは使用していない。

## Development comparison

- Old 100 v2.2 -> v2.3: score 90.11 -> 90.21; ◎/△/× 74/25/1 -> 89/4/7
- Old Blind 300 v2.2 -> v2.3: score 91.13 -> 91.63; ◎/△/× 242/57/1 -> 290/10/0

## New Blind 500

- ◎/△/×: 461 / 39 / 0
- Score / characters: 91.69 / 535.58
- Correctness: 92.92%
- Relevance: 92.50%
- Actionable: 92.44%
- Completeness: 90.14%
- Specificity: 89.46%
- Naturalness: 92.90%
- Unsupported Claim: 0.40%
- Critical errors: 2
- Coverage / multi-intent coverage: 99.93% / 100.00%
- Routes: Safe 82.80%, RAG 12.40%, Tool 4.80%, Clarification 0.00%

## Stress 200 and retrieval

- Stress ◎/△/×: 198 / 2 / 0
- Stress critical errors / policy assertions: 1 / 0
- Selected retrieval benchmark strategy: TF-IDF
- Recall@1 / Recall@3 / Recall@5 / MRR: 100.00% / 100.00% / 100.00% / 1.0000
- False Match: 23.00% -> 0.00%
- False No Match: 0.20%
- Knowledge sources / documents / chunks: 518 / 610 / 1,096
- Numeric conflict groups / unresolved: 10 / 0

## Decision

- Human review required: 3
- Standard 50M: YES
- Production Gate: FAIL
- Beta: NO
- Production/Render/Vercel/Release changed: NO
- Push/deploy: NO
