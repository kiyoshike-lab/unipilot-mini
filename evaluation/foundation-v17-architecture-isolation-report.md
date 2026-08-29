# UniPilot Foundation v1.7 Residual / Position / Initialization Isolation

## 最終判定

- Architecture decision: **MULTI_COMPONENT_FIX_REQUIRED**
- Architecture Gate: **FAIL**
- 正式architecture変更: **NO**
- Full 256kへ進めるか: **NO**
- Final Norm: **PRESENT**
- Combined ablation: **NOT EXECUTED**（単独候補が総合PASSしなかったため）

depth-scaled residual projection initは実Corpus、activation、Copy、Positionを改善したが、Key Lookup、numeric pattern、Full Corpus frequency gateを解決しない。position両scaleはtoken/position ratioを戻したもののresidual全体を過大化しSyntheticを悪化させた。

## 3-seed real corpus（64k、mean ± population std）

| Config | loss | Top-1 | Top-5 | Top-10 | Context | Layer9 RMS | punctuation mass | non-top1 Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current | 7.1325 ± 0.0086 | 5.46% ± 0.22% | 12.05% ± 0.25% | 16.72% ± 0.03% | 1.8448 ± 0.5422 | 4.4827 ± 0.4461 | 92.52% | 0.00% |
| sqrt A | 7.0318 ± 0.0101 | 7.06% ± 0.10% | 12.57% ± 0.16% | 17.00% ± 0.11% | 3.4668 ± 0.5123 | 2.4376 ± 0.1537 | 82.65% | 0.00% |
| depth-scaled residual init | 7.0879 ± 0.0055 | 6.53% ± 0.09% | 12.24% ± 0.08% | 16.90% ± 0.10% | 2.7215 ± 0.2135 | 2.1584 ± 0.1260 | 90.98% | 0.00% |

## Position scale / gradient

| Config | update | token mean/std/RMS/norm | position mean/std/RMS/norm | combined RMS | ratio |
|---|---:|---:|---:|---:|---:|
| Current | 0 | -0.0001/0.0201/0.0201/4.45 | -0.0001/0.0200/0.0200/4.44 | 0.0284 | 1.00 |
| Current | 10 | -0.0001/0.0201/0.0201/4.45 | -0.0001/0.0200/0.0200/4.44 | 0.0284 | 1.00 |
| Current | 50 | -0.0001/0.0203/0.0203/4.49 | -0.0001/0.0200/0.0200/4.44 | 0.0285 | 1.01 |
| Current | 100 | -0.0001/0.0205/0.0205/4.55 | -0.0001/0.0200/0.0200/4.44 | 0.0287 | 1.03 |
| sqrt A | 0 | -0.0011/0.3932/0.3932/87.17 | -0.0001/0.0200/0.0200/4.44 | 0.3937 | 19.63 |
| sqrt A | 10 | -0.0011/0.3931/0.3931/87.15 | -0.0001/0.0200/0.0200/4.44 | 0.3936 | 19.62 |
| sqrt A | 50 | -0.0012/0.3966/0.3966/87.92 | -0.0001/0.0200/0.0200/4.44 | 0.3971 | 19.80 |
| sqrt A | 100 | -0.0012/0.4020/0.4020/89.12 | -0.0001/0.0200/0.0200/4.44 | 0.4025 | 20.08 |
| sqrt token+position | 0 | -0.0028/0.3931/0.3931/87.16 | 0.0039/0.3944/0.3944/87.43 | 0.5545 | 1.00 |
| sqrt token+position | 10 | -0.0028/0.3930/0.3931/87.14 | 0.0039/0.3943/0.3943/87.43 | 0.5544 | 1.00 |
| sqrt token+position | 50 | -0.0028/0.3965/0.3965/87.90 | 0.0039/0.3933/0.3933/87.19 | 0.5562 | 1.01 |
| sqrt token+position | 100 | -0.0028/0.4020/0.4020/89.13 | 0.0039/0.3922/0.3922/86.95 | 0.5593 | 1.03 |
| depth-scaled residual init | 0 | -0.0001/0.0201/0.0201/4.45 | -0.0001/0.0200/0.0200/4.44 | 0.0284 | 1.00 |
| depth-scaled residual init | 10 | -0.0001/0.0201/0.0201/4.45 | -0.0001/0.0200/0.0200/4.44 | 0.0284 | 1.00 |
| depth-scaled residual init | 50 | -0.0001/0.0203/0.0203/4.50 | -0.0001/0.0200/0.0200/4.44 | 0.0285 | 1.01 |
| depth-scaled residual init | 100 | -0.0001/0.0206/0.0206/4.56 | -0.0001/0.0200/0.0200/4.43 | 0.0287 | 1.03 |

| Config | position grad RMS | position delta RMS |
|---|---:|---:|
| Current | 0.00034953 | 0.00094143 |
| sqrt A | 0.00005608 | 0.00095958 |
| sqrt token+position | 0.00037313 | 0.00092117 |
| depth-scaled residual init | 0.00072622 | 0.00100251 |

sqrt Aはposition gradient RMSもCurrentより小さく、effective ratioは約20。両scale候補はratioを約1へ戻すが、representation全体を約19.6倍にするためresidualに対するbranch contributionが低下した。

## Architecture matrix（Syntheticはseed 42）

| Config | Params | Validation | Top-1/5 | Copy 4/8/16 | Key min 2/4/8 | Pattern basic/numeric | Position min | Context | Layer9 | punct. | tok/s | RAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current | 19,514,880 | 7.1325 | 5.46%/12.05% | 100.00%/100.00%/92.97% | 42.97%/19.92%/7.81% | 100.00%/22.27% | 92.97% | 100.00% | 4.4827 | 92.52% | 4958 | 881.3 |
| sqrt A | 19,514,880 | 7.0318 | 7.06%/12.57% | 21.48%/12.89%/8.20% | 49.22%/23.83%/10.16% | 94.92%/1.95% | 6.25% | 100.00% | 2.4376 | 82.65% | 4706 | 879.7 |
| sqrt token+position | 19,514,880 | 7.1050 | 6.23%/11.82% | 72.66%/34.38%/17.58% | 46.88%/21.88%/13.28% | 100.00%/10.55% | 17.58% | 100.00% | 4.9279 | 95.26% | 4574 | 821.0 |
| depth-scaled residual init | 19,514,880 | 7.0879 | 6.53%/12.24% | 100.00%/100.00%/98.44% | 44.53%/17.97%/9.38% | 100.00%/23.44% | 99.22% | 100.00% | 2.1584 | 90.98% | 5283 | 824.6 |

## Residual stream

Final updateの全layer。attn ratio=attention output/pre-attention residual、MLP ratio=MLP output/post-attention residual。

| Config | layer | residual RMS | output RMS | attn ratio | MLP ratio |
|---|---:|---:|---:|---:|---:|
| Current | 0 | 0.0432 | 0.2224 | 1.1227 | 4.9965 |
| Current | 1 | 0.2715 | 0.5197 | 0.5800 | 1.2967 |
| Current | 2 | 0.6047 | 0.8866 | 0.3545 | 0.6227 |
| Current | 3 | 1.0029 | 1.2977 | 0.2513 | 0.3900 |
| Current | 4 | 1.4382 | 1.7476 | 0.1793 | 0.2777 |
| Current | 5 | 1.9111 | 2.2362 | 0.1513 | 0.2203 |
| Current | 6 | 2.4199 | 2.7628 | 0.1233 | 0.1794 |
| Current | 7 | 2.9425 | 3.3027 | 0.1005 | 0.1534 |
| Current | 8 | 3.4945 | 3.8636 | 0.0869 | 0.1304 |
| Current | 9 | 4.0741 | 4.4590 | 0.0782 | 0.1148 |
| sqrt A | 0 | 0.4049 | 0.4437 | 0.0944 | 0.4628 |
| sqrt A | 1 | 0.4473 | 0.5030 | 0.1060 | 0.4448 |
| sqrt A | 2 | 0.5118 | 0.5909 | 0.1270 | 0.4152 |
| sqrt A | 3 | 0.6105 | 0.7259 | 0.1489 | 0.3959 |
| sqrt A | 4 | 0.7730 | 0.9411 | 0.1749 | 0.3644 |
| sqrt A | 5 | 1.0363 | 1.2698 | 0.1984 | 0.3253 |
| sqrt A | 6 | 1.4144 | 1.6953 | 0.1858 | 0.2650 |
| sqrt A | 7 | 1.8591 | 2.1766 | 0.1504 | 0.2192 |
| sqrt A | 8 | 2.3752 | 2.7078 | 0.1323 | 0.1748 |
| sqrt A | 9 | 2.9255 | 3.2866 | 0.1128 | 0.1501 |
| sqrt token+position | 0 | 0.5616 | 0.5948 | 0.0740 | 0.3392 |
| sqrt token+position | 1 | 0.6051 | 0.6622 | 0.1172 | 0.3428 |
| sqrt token+position | 2 | 0.7006 | 0.8154 | 0.1803 | 0.3650 |
| sqrt token+position | 3 | 0.9173 | 1.1289 | 0.2491 | 0.3640 |
| sqrt token+position | 4 | 1.3082 | 1.5922 | 0.2482 | 0.2998 |
| sqrt token+position | 5 | 1.8075 | 2.1670 | 0.1972 | 0.2502 |
| sqrt token+position | 6 | 2.4238 | 2.8149 | 0.1584 | 0.1968 |
| sqrt token+position | 7 | 3.0608 | 3.4611 | 0.1228 | 0.1588 |
| sqrt token+position | 8 | 3.7424 | 4.1414 | 0.1087 | 0.1283 |
| sqrt token+position | 9 | 4.4161 | 4.8515 | 0.0861 | 0.1171 |
| depth-scaled residual init | 0 | 0.0297 | 0.0664 | 0.2787 | 1.9569 |
| depth-scaled residual init | 1 | 0.0705 | 0.1726 | 0.2834 | 1.8347 |
| depth-scaled residual init | 2 | 0.1861 | 0.3209 | 0.1860 | 0.8710 |
| depth-scaled residual init | 3 | 0.3501 | 0.4943 | 0.1616 | 0.4817 |
| depth-scaled residual init | 4 | 0.5428 | 0.6948 | 0.1470 | 0.3265 |
| depth-scaled residual init | 5 | 0.7623 | 0.9283 | 0.1326 | 0.2491 |
| depth-scaled residual init | 6 | 1.0141 | 1.1915 | 0.1171 | 0.1991 |
| depth-scaled residual init | 7 | 1.2894 | 1.4900 | 0.1006 | 0.1743 |
| depth-scaled residual init | 8 | 1.6066 | 1.8172 | 0.0920 | 0.1457 |
| depth-scaled residual init | 9 | 1.9378 | 2.1557 | 0.0770 | 0.1247 |

step 0/10/25/50/100の全layer値はsummary JSONに保存。depth-initはattention/MLP output projectionだけを0.02/sqrt(20)=0.004472へ変更し、QKV/MLP inputは0.02を維持。runtime residual scalingは使用していない。

## Copy / Numeric failure

| Config | Copy classification (96 probes) | Numeric classification (256 probes) |
|---|---|---|
| Current | `{"correct": 94, "position_shift": 2}` | `{"correct": 76, "seen_pattern_token_wrong_phase": 171, "value_token_outside_pattern": 9}` |
| sqrt A | `{"correct": 14, "position_shift": 82}` | `{"correct": 8, "seen_pattern_token_wrong_phase": 248}` |
| sqrt token+position | `{"correct": 38, "position_shift": 58}` | `{"correct": 44, "seen_pattern_token_wrong_phase": 210, "value_token_outside_pattern": 2}` |
| depth-scaled residual init | `{"correct": 95, "position_shift": 1}` | `{"correct": 70, "seen_pattern_token_wrong_phase": 161, "value_token_outside_pattern": 25}` |

sqrt AのCopy誤り82/96は別position tokenへの置換で、固定offsetはなく広く分散。balancedも58/96がposition shift。Current/depth-initは94/96、95/96正解。Numeric誤りは全構成で主に既出pattern tokenのwrong phaseで、非value/frequency tokenへのcollapseではない。従ってnumeric failureは位置比だけでなくsequence phase推論の問題。

Synthetic audit: final answer位置だけteacher forcing、他target=-100、EOS/packingなし、入力末尾は固定ANSWER sentinelで実answerは含まない。train/test overlapは全構成0。

## Frequency / hidden similarity

tied LM headでは `logit = ||hidden|| × ||token embedding|| × cosine`。全構成でcorrect-token cosineより句読点・助詞方向の平均cosineが高く、hiddenが頻出token方向へ寄る現象を確認した。詳細なtoken別cosine/logitはsummary JSONに保存。

Clean Japaneseではdepth-initがloss 6.9165→6.8260、Top-1 8.73→9.31%、punctuation mass 57.86%、Top 1%外Top-1>0を達成。一方Full Corpus 3-seedではpunctuation massは90.98%へわずかに低下したが、Top 1%外Top-1は0%。

## Norm audit

Final LayerNormは全構成で存在し、`Embedding -> Blocks -> Final LayerNorm -> LM Head`。各layerのinput/normalized mean/std、gamma、betaをsummary JSONに保存し、非finite値やFinal Norm欠落はなかった。PHASE 26で単独改善しなかったためRMSNorm再実験は未実施。

## Gate

| Check | Result |
|---|---|
| three_seed_validation_loss_better_than_current | PASS |
| three_seed_top_1_better_than_current | PASS |
| three_seed_top_5_better_than_current | PASS |
| context_better_than_current | PASS |
| layer_9_rms_better_than_current | PASS |
| full_corpus_non_top1_accuracy_above_zero | FAIL |
| full_corpus_punctuation_mass_below_current | PASS |
| clean_japanese_loss_better_than_current | PASS |
| clean_japanese_frequency_gate | PASS |
| synthetic_gate | FAIL |

結論はMULTI_COMPONENT_FIX_REQUIRED。ただし現時点で採用するarchitectureはない。formal FoundationはCurrent構成を維持し、depth-scaled residual initは次フェーズの実験用partial fixとしてのみ保持する。単独総合PASSがないため組合せは今回実行していない。

Full 256k、512k、1M、46M、Corpus/Tokenizer変更、Campus、Instruction/DPO、Production、push/deployは未実施。

Final Blindは内容を開かずSHA256のみ確認: `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b` (MATCH)。
