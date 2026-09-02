# UniPilot Foundation v2.1 Controlled 256k Architecture A/B Report

Final Gate: **CURRENT_RETAIN**
Formal Foundation Architecture: **Current**
Foundation Base complete: **NO**
Next token budget: **512k**

## Six runs

| Architecture | Seed | Tokens | Best val | Final val | Top-1 | Top-5 | Top-10 | tok/s | Peak RAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 42 | 256,000 | 6.7804 | 6.7804 | 8.09% | 14.11% | 18.48% | 873.7 | 991.4 |
| current | 123 | 256,000 | 6.7801 | 6.7801 | 7.90% | 14.04% | 18.23% | 875.0 | 1096.9 |
| current | 2026 | 256,000 | 6.7805 | 6.7805 | 8.06% | 14.70% | 18.36% | 886.9 | 1075.4 |
| depth_init | 42 | 256,000 | 6.7283 | 6.7283 | 8.20% | 14.49% | 18.64% | 878.0 | 1007.8 |
| depth_init | 123 | 256,000 | 6.7239 | 6.7239 | 7.91% | 14.72% | 18.98% | 874.6 | 1014.5 |
| depth_init | 2026 | 256,000 | 6.7253 | 6.7253 | 8.28% | 14.98% | 18.79% | 890.5 | 1073.3 |

## 3-seed final mean ± sample std

- validation_loss: Current 6.780339 ± 0.000244; Depth 6.725800 ± 0.002256.
- top_1: Current 0.080160 ± 0.001038; Depth 0.081299 ± 0.001938.
- top_5: Current 0.142822 ± 0.003613; Depth 0.147298 ± 0.002442.
- top_10: Current 0.183553 ± 0.001282; Depth 0.188029 ± 0.001715.
- punctuation_mass: Current 0.391113 ± 0.058865; Depth 0.347900 ± 0.021659.
- top_1_percent_outside: Current 0.000272 ± 0.000472; Depth 0.000272 ± 0.000472.
- layer_9_rms: Current 4.430074 ± 0.063852; Depth 2.645060 ± 0.210173.
- context_advantage: Current 0.237858 ± 0.193921; Depth 0.221899 ± 0.145628.

## Validation loss learning curve

| Tokens | Current mean ± std | Depth mean ± std | Depth−Current paired mean |
|---:|---:|---:|---:|
| 0 | 8.3866 ± 0.0211 | 8.3771 ± 0.0244 | -0.0096 |
| 64,000 | 7.1372 ± 0.0127 | 7.0950 ± 0.0089 | -0.0423 |
| 128,000 | 6.9989 ± 0.0101 | 6.9464 ± 0.0119 | -0.0525 |
| 192,000 | 6.8689 ± 0.0028 | 6.8183 ± 0.0022 | -0.0505 |
| 256,000 | 6.7803 ± 0.0002 | 6.7258 ± 0.0023 | -0.0545 |

## Punctuation and residual learning curves

| Tokens | Current punct. mass | Depth punct. mass | Current Layer9 RMS | Depth Layer9 RMS |
|---:|---:|---:|---:|---:|
| 0 | 0.00% ± 0.00% | 1.52% ± 1.22% | 0.631 ± 0.015 | 0.135 ± 0.003 |
| 64,000 | 93.36% ± 1.66% | 92.10% ± 2.63% | 4.533 ± 0.566 | 2.098 ± 0.157 |
| 128,000 | 87.83% ± 4.26% | 65.63% ± 12.08% | 3.961 ± 0.293 | 2.165 ± 0.081 |
| 192,000 | 61.89% ± 5.65% | 46.10% ± 2.98% | 4.401 ± 0.118 | 2.566 ± 0.084 |
| 256,000 | 39.11% ± 5.89% | 34.79% ± 2.17% | 4.430 ± 0.064 | 2.645 ± 0.210 |

## Final frequency buckets (3-seed mean)

| Bucket | Architecture | Top-1 | Top-5 | Top-10 | Correct prob. | Cross entropy |
|---|---|---:|---:|---:|---:|---:|
| top_1_percent | current | 31.61% | 53.72% | 67.79% | 0.0919 | 4.0240 |
| top_1_percent | depth_init | 32.06% | 55.00% | 68.40% | 0.0997 | 3.9833 |
| top_5_percent_excluding_top_1 | current | 0.00% | 2.03% | 3.55% | 0.0023 | 6.3414 |
| top_5_percent_excluding_top_1 | depth_init | 0.00% | 2.57% | 4.71% | 0.0025 | 6.2715 |
| top_20_percent_excluding_top_5 | current | 0.08% | 1.15% | 1.96% | 0.0008 | 7.4841 |
| top_20_percent_excluding_top_5 | depth_init | 0.08% | 1.22% | 2.23% | 0.0008 | 7.4287 |
| middle_20_to_80_percent | current | 0.00% | 0.00% | 0.01% | 0.0002 | 8.7478 |
| middle_20_to_80_percent | depth_init | 0.00% | 0.00% | 0.01% | 0.0002 | 8.6925 |
| rare_bottom_20_percent | current | 0.00% | 0.00% | 0.00% | 0.0000 | 10.6192 |
| rare_bottom_20_percent | depth_init | 0.00% | 0.00% | 0.00% | 0.0000 | 10.5543 |

## Context utilization (final 3-seed mean loss)

| Context tokens | Current | Depth |
|---:|---:|---:|
| 512 | 6.8842 | 6.7579 |
| 64 | 6.8828 | 6.7809 |
| 16 | 6.9071 | 6.7391 |
| 2 | 7.0047 | 6.8657 |
| 1 | 7.1221 | 6.9798 |

## Representative generation at 256k (seed 42)

| Architecture | Mode | Valid | Natural proxy | Semantic proxy | Completion | Runaway | Repetition |
|---|---|---:|---:|---:|---:|---:|---:|
| current | greedy_no_penalty | 80.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.0667 |
| current | sampling_t07_topk40_topp09_no_penalty | 80.00% | 30.00% | 0.00% | 0.00% | 100.00% | 0.0056 |
| depth_init | greedy_no_penalty | 85.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.6287 |
| depth_init | sampling_t07_topk40_topp09_no_penalty | 65.00% | 35.00% | 0.00% | 0.00% | 100.00% | 0.0135 |

## Selection gates

- A_validation_loss_mean_improves: PASS
- B_improvement_consistent_across_seeds: PASS
- C_top_1_5_10_not_worse: PASS
- D_punctuation_collapse_clearly_improves: PASS
- E_residual_rms_stable: PASS
- F_synthetic_smoke_no_regression: PASS
- G_context_no_major_regression: PASS
- H_generation_trend_not_worse: FAIL

## Interpretation

Depth-init improved validation loss in all three paired seeds, improved the 3-seed mean Top-1/5/10, reduced mean punctuation collapse at every trained milestone, and kept Layer9 RMS substantially lower. Context utilization remained positive without a major mean regression.

It is not promoted because the fixed representative generation evaluation showed a clear greedy repetition regression (+0.5621) and lower sampling character validity (-0.1500). PHASE 32 explicitly forbids promotion when language-model metrics improve but generation regresses. Current is therefore retained; this does not claim that Depth has lower learning capacity.

Synthetic smoke: PASS for both variants. Novel random Key Lookup and modular addition were not gates.
Final Blind content was not parsed; SHA256 only was verified.
No Production, Campus, Render, Vercel, tokenizer, or corpus change was made.
