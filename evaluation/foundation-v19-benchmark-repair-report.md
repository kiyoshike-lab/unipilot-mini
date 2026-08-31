# UniPilot Foundation v1.9 Synthetic Benchmark Repair Report

## 最終判定

- Benchmark validity: **FAIL**
- Final Gate: **RELATION_TRAINING_ISSUE**
- Depth-init Architecture Candidate: **NO**
- 次にFull 256k: **NO**
- 正式Foundation architecture: Currentのまま
- Current / Depth v3大量比較: Reference-first gate未達のため未実行

## 旧Key Lookupのroot cause

旧taskはcausalで曖昧性もなく、実装上すでにanswer-only lossだった。したがって未来参照、重複正解、all-token structural lossは旧失敗の原因ではない。Phase 29ではKey Lookupが全更新の16.675%で、12個のpair×distance cellへ分散し、各cellは80〜172 update相当しかなかった。v3のReference Level 2は1,800 update付近までchance plateauが続き、2,800 updateで99.2%に到達したため、主因はassociative retrieval形成前に終わる過少・分散supervisionだった。

ただし4-pair Level 3は4,800 updateでも基準未達だった。v3は1/2-pair問題を修復したが、Required Gate全体はまだ成立していない。

## 監査

- 旧Key例: 10件をtoken IDs / target / mask / query/key/value/answer位置付きで保存
- 旧全走査: 15,000件、causal failure 0、ambiguity 0
- v3全走査: 10,000件、causal failure 0、ambiguity 0
- Level 2-5 exact train/test overlap: 0
- mapping overlap: 223（in-context taskでは許容し、別途計測）

## Loss / marker比較（Reference、Level 2）

| Variant | Updates | Accuracy | Answer loss | Non-answer loss |
|---|---:|---:|---:|---:|
| answer-only-markers | 400 | 47.66% | 1.4740 | — |
| answer-only-no-markers | 400 | 42.58% | 1.4517 | — |
| all-token-1x-markers | 400 | 35.94% | 2.4828 | 1.1137 |
| all-token-4x-markers | 400 | 39.06% | 1.8877 | 1.1067 |
| all-token-16x-markers | 400 | 42.19% | 2.0083 | 1.1781 |

旧runはanswer-only。all-tokenは反実仮想比較であり、1x/4x/16xはいずれも400 update時にanswer-onlyを上回らなかった。markersありは47.66%、なしは42.58%で、構造markerは小さいが正の効果。canonical v3は単純なanswer-only + markersを採用した。

## Curriculum / Reference

| Level | Pairs | Accuracy | Chance(candidate) | Threshold | PASS |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 100.00% | 100.00% | 99% | PASS |
| 1 | 1 | 100.00% | 100.00% | 99% | PASS |
| 2 | 2 | 100.00% | 50.00% | 98% | PASS |
| 3 | 4 | 51.56% | 25.00% | 95% | FAIL |
| 4 | 8 | 14.84% | 12.50% | 90% | FAIL |
| 5 | 16 | 7.03% | 6.25% | diagnostic | diagnostic |

Level 4まで学習到達しなかったため、Level 4 counterfactual/ablationはVALID判定材料としてFAIL。
- correct 14.84%; counterfactual 9.38%
- shuffled 10.94% (drop 3.91%)
- removed relation 0.00% (drop 14.84%)
- wrong query/original target 13.67%; wrong query/new target 10.94%; removed query 14.45%

## Level 2 attention learning curve

| Update | Accuracy | Correct K+V mass | Rank | Margin | Entropy |
|---:|---:|---:|---:|---:|---:|
| 10 | 2.73% | 0.1474 | 5.78 | -0.12 | 0.9937 |
| 25 | 3.52% | 0.1685 | 4.33 | -0.49 | 0.8667 |
| 50 | 11.72% | 0.1813 | 4.48 | -0.33 | 0.8860 |
| 100 | 41.02% | 0.1583 | 5.25 | -0.95 | 0.8260 |
| 200 | 46.88% | 0.1634 | 5.62 | -0.58 | 0.8563 |
| 400 | 45.70% | 0.1861 | 4.78 | -1.50 | 0.6549 |
| 800 | 51.17% | 0.1768 | 4.34 | -1.75 | 0.5868 |
| 1200 | 49.61% | 0.1748 | 4.34 | -2.47 | 0.5316 |
| 1600 | 47.27% | 0.1546 | 4.90 | -3.92 | 0.4962 |
| 2400 | 89.45% | 0.1005 | 4.93 | -15.06 | 0.1625 |
| 3200 | 99.22% | 0.0833 | 5.28 | -27.93 | 0.0845 |
| 4000 | 100.00% | 0.0677 | 5.57 | -34.54 | 0.1210 |

accuracyは上昇したが、final-markerから見た単純なdirect K/V attention massは単調増加しなかった。このmetricだけでは複数layerに分散したretrieval計算を説明できず、attention supervisionは使用していない。

## Numeric / Symbolic

- Foundation tokenizer単数字atomic率: 100.00%
- raw numeric/token ID監査: standalone 24件、numeric sequence 20件
- Phase 29 symbolic (arithmetic不要): Current / Depth / Reference = 99.61% / 100% / 100%（>=95% PASS）
- Phase 29 numeric: 39.45% / 41.02% / 42.19%。これはFoundation tokenizerではなくatomic synthetic IDs上のmodular additionで、単純pattern continuationではない。Architecture Gateから除外しactual numericもdiagnosticのみ。

## Phase 29再利用 / Architecture

Copy、Long Range、Context-conditioned、Position、SymbolicはCurrent / Depth / ReferenceすべてPASS。Japanese 128kではDepthがCurrentよりloss、punctuation collapse、Layer9 RMSを改善したが、v3 Reference gateがFAILのためDepth候補復帰条件A/Bを満たさない。Current / Depthのv3比較は実行していない。

| Model | v3 Key L0/1/2/3/4/5 | Copy 4/8/16 min | Symbolic | Context | Position min |
|---|---|---:|---:|---:|---:|
| Current | SKIPPED (Reference gate) | 100.0% | 99.6% | 100.0% | 100.0% |
| Depth-init | SKIPPED (Reference gate) | 99.6% | 100.0% | 100.0% | 100.0% |
| Reference | 100.0%/100.0%/100.0%/51.6%/14.8%/7.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## Integrity / protection

- Reference diagnostic parameters: 17,892,864; formal architecture parameters: 19,514,880
- Checkpoint strict reload: True; optimizer state: True; SHA256 `8e3c87afd4e05ca86ba9bc0faa01f27426e154c41e3fd295c2e45f5ea59b98a0`
- Final Blind: content unopened; SHA256 `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b`; match=True
- focused v3/Reference/Numeric: 40 passed; full pytest: 319 passed, 3 warnings
- Full corpus、46M、tokenizer本体、architecture、本番、Render、Vercel、Releaseは未変更。

## 次の推奨

Full 256kへは進まない。次PHASEは4/8-pair relation curriculumの再設計（3-pair中間段階、seed安定性、direct-attention metricの限界）に限定し、Reference Level 3/4とcontrolsがPASSしてからCurrent/Depthを比較する。
