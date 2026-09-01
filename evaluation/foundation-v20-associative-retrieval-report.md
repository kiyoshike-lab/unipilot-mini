# UniPilot Foundation v2.0 Associative Retrieval Curriculum Validation

## 最終判定

- Gate: **TASK_COMPLEXITY_ISSUE**
- Benchmark validity: **FAIL**
- Oracle: **PASS**
- Depth-init Candidate: **NO**
- Architecture fatal defect: **NO**
- Full Foundation 256kへ進めるか: **NO**
- 正式Foundation architecture: Currentのまま

## Benchmark v3.1固定条件

- Manifest: `evaluation/foundation-v20-synthetic-context-benchmark-v31-manifest.json`
- Manifest file SHA256: `002f9ce24114b9eb213ae3dee43f14d62042d560c58fdfb5368ac9ac7d482db2`
- Manifest content SHA256: `3ff51751e683201ff50a699b1f7f9d7b75bb1a6dbc8fcd092c60d48bd60f07f9`
- Vocabulary: diagnostic 256 / formal model 4,096
- Markers: `<PAIR> <KEY> <VALUE> <QUERY> <ANSWER>`
- Sequence: `<KEY_LOOKUP> (<PAIR> <KEY> key <VALUE> value){N} <QUERY> query-key <ANSWER>; predict value at <ANSWER> position`
- Loss: answer-only、final `<ANSWER>`位置のみ
- Train/Test: canonical v3 `any` mapping、testのexact mapping combinationはtrain履歴から除外
- Data: on-the-fly deterministic random generation
- Parameters: 19,514,880
- Optimizer: AdamW, LR 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0.01
- Batch/effective batch: 16 / 16、gradient clip 1.0

## Oracle

- L0: 100.00% (10,000 examples, causal failures 0, ambiguity 0)
- L1: 100.00% (10,000 examples, causal failures 0, ambiguity 0)
- L2: 100.00% (10,000 examples, causal failures 0, ambiguity 0)
- L3: 100.00% (10,000 examples, causal failures 0, ambiguity 0)
- L4: 100.00% (10,000 examples, causal failures 0, ambiguity 0)
- L5: 100.00% (10,000 examples, causal failures 0, ambiguity 0)

## Reference L3 / 4 pairs

| Updates | Examples | Unique | Tokens | Accuracy | Loss |
|---:|---:|---:|---:|---:|---:|
| 100 | 1,600 | 1,600 | 38,400 | 6.25% | 3.4409 |
| 250 | 4,000 | 4,000 | 96,000 | 15.23% | 3.2227 |
| 500 | 8,000 | 8,000 | 192,000 | 15.23% | 2.9772 |
| 1,000 | 16,000 | 16,000 | 384,000 | 22.27% | 2.2867 |
| 2,000 | 32,000 | 32,000 | 768,000 | 26.95% | 1.7993 |
| 4,000 | 64,000 | 64,000 | 1,536,000 | 28.52% | 1.6784 |
| 8,000 | 128,000 | 128,000 | 3,072,000 | 18.36% | 1.7355 |

PASS: **FALSE**。Final novel mapping: 26.17%。

### L3 sample efficiency

- 50%: 未到達
- 75%: 未到達
- 90%: 未到達
- 95%: 未到達
- 98%: 未到達

### L3 controls

- Fixed mapping: 100.00%
- Novel mapping final: 26.17%
- Counterfactual: 8.98%
- Shuffled/original target: 26.56% (drop -0.39%)
- Removed relation/original target: 0.00%
- Correct query: 26.17%
- Wrong query/new target: 19.14%
- Wrong query/original target: 26.56%
- Removed query/original target: 25.39%
- Controls PASS: **FALSE**

### L3 optimizer sanity

- LR 3x (9e-4), 500 updates: 8.98%
- weight decay 0, 1,000 updates: 23.83%
- LR 10x: 3xが悪化したため未実施

## Reference L4 / 8 pairs

| Updates | Examples | Unique | Tokens | Accuracy | Loss |
|---:|---:|---:|---:|---:|---:|
| 250 | 4,000 | 4,000 | 176,000 | 1.95% | 3.5679 |
| 500 | 8,000 | 8,000 | 352,000 | 5.86% | 3.4662 |
| 1,000 | 16,000 | 16,000 | 704,000 | 6.64% | 3.4567 |
| 2,000 | 32,000 | 32,000 | 1,408,000 | 10.16% | 3.3010 |
| 4,000 | 64,000 | 64,000 | 2,816,000 | 8.98% | 3.2616 |
| 8,000 | 128,000 | 128,000 | 5,632,000 | 9.38% | 3.1397 |
| 16,000 | 256,000 | 256,000 | 11,264,000 | 8.98% | 3.0433 |

PASS: **FALSE**。Final novel mapping: 11.33%。

### L4 sample efficiency

- 50%: 未到達
- 75%: 未到達
- 90%: 未到達
- 95%: 未到達
- 98%: 未到達

### L4 controls

- Fixed mapping: 100.00%
- Novel mapping final: 11.33%
- Counterfactual: 49.22%
- Shuffled/original target: 15.23% (drop -3.91%)
- Removed relation/original target: 0.39%
- Correct query: 11.33%
- Wrong query/new target: 12.89%
- Wrong query/original target: 11.33%
- Removed query/original target: 11.33%
- Controls PASS: **FALSE**

## Attention retrieval curve — L3

| Updates | Accuracy | Entropy | Key mass | Value mass | K+V mass | Rank | Margin | Max prob |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 6.25% | 0.9137 | 0.0368 | 0.0679 | 0.1048 | 9.49 | -0.3195 | 0.1261 |
| 250 | 15.23% | 0.9057 | 0.0374 | 0.0684 | 0.1058 | 9.22 | -0.3695 | 0.1344 |
| 500 | 15.23% | 0.8317 | 0.0338 | 0.0666 | 0.1004 | 8.03 | -0.8782 | 0.1719 |
| 1,000 | 22.27% | 0.4569 | 0.0345 | 0.0330 | 0.0675 | 9.75 | -8.2129 | 0.3890 |
| 2,000 | 26.95% | 0.4107 | 0.0392 | 0.0380 | 0.0772 | 9.15 | -10.2309 | 0.4197 |
| 4,000 | 28.52% | 0.3444 | 0.0304 | 0.0360 | 0.0664 | 8.18 | -13.0049 | 0.5202 |
| 8,000 | 18.36% | 0.2875 | 0.0369 | 0.0456 | 0.0825 | 7.86 | -16.9350 | 0.6153 |

## Attention retrieval curve — L4

| Updates | Accuracy | Entropy | Key mass | Value mass | K+V mass | Rank | Margin | Max prob |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 1.95% | 0.9111 | 0.0204 | 0.0281 | 0.0485 | 14.48 | -0.7519 | 0.1017 |
| 500 | 5.86% | 0.6826 | 0.0206 | 0.0323 | 0.0529 | 15.69 | -3.7093 | 0.2581 |
| 1,000 | 6.64% | 0.6447 | 0.0197 | 0.0280 | 0.0477 | 16.24 | -5.4404 | 0.2391 |
| 2,000 | 10.16% | 0.4145 | 0.0202 | 0.0195 | 0.0397 | 17.87 | -11.0418 | 0.4941 |
| 4,000 | 8.98% | 0.3531 | 0.0120 | 0.0226 | 0.0346 | 16.97 | -14.4790 | 0.5690 |
| 8,000 | 9.38% | 0.3087 | 0.0194 | 0.0389 | 0.0583 | 15.71 | -17.9447 | 0.6288 |
| 16,000 | 8.98% | 0.2535 | 0.0161 | 0.0289 | 0.0451 | 17.03 | -28.3005 | 0.6744 |

## Current / Depth

Benchmark ValidityがFAILのため、Reference-first規則に従いSequential Curriculum、Current、Depth-initは未実行。
Depth-initはArchitecture Candidateへ昇格しない。PHASE 29 Japanese diagnostic値は再利用し、再学習していない。

## Root cause

Oracle 100%、causal/ambiguity failure 0、exact sequence/mapping overlap 0でgenerator不具合は否定された。
L3は128,000 examples / 3,072,000 tokens、L4は256,000 examples / 11,264,000 tokensでもGateへ収束せず、
LR 3xは悪化、weight decay 0も解決しなかった。
L2まではPHASE 30で100%学習可能だったため、Custom固有のfatal architecture defectではなく、4/8-pairで急増するtask/objective sample complexityと分類する。

## Integrity / protection

- L3 checkpoint: strict=True SHA256 `b15d0fe34d5a5c6b837c48deaa29f1ea05ed19d32749107572b50b3e2db498bb`
- L4 checkpoint: strict=True SHA256 `c1554557de9b156505f318c350f043200d4b211d0d98010676f6757521a94f46`
- Final Blind: content unopened、SHA256 match=True
- PHASE 31 focused tests: 20 passed
- Full pytest: 339 passed、3 warnings（既存FastAPI/Starlette deprecation）
- Oracle / v3.1 / curriculum gate / novel mapping / counterfactual / query / relation ablation / determinism: PASS
- Resume reproducibility: PASS
- Independent checkpoint strict reload / SHA256: PASS
- Production / Render / Vercel: 変更・deployなし

## 次PHASE推奨

Full 256kへは進まない。Benchmark修復を継続し、3-pair中間Level、distractor数とrelation数の分離、
query/value copy補助diagnostic、複数seedでの再現性をReferenceだけで確認してからv3.2 Gateを再定義する。
