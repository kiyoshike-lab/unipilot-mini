# Foundation v3.4 LM Gate Failure Root-Cause Review

## Decision

- Primary cause: **SEED_LOCAL_SHORT_TERM_VARIANCE: seed 42 deterministic loss/Top-1; seed 123 sampling Semantic**
- Plateau classification: **HEALTHY_SHORT_TERM_VARIANCE**
- Next Gate: **CONTINUE_SHORT_GPU_GATES_EOS_1_5**
- Recommended next interval: **256,000 tokens**
- Recommended EOS weight: **1.5**
- 20M permission: **NO**
- Foundation Base completion: **NO**

Gate 2 combined two different seed-local signals: seed 42 caused the deterministic loss/Top-1 decline, while seed 123 caused the sampling Semantic decline. Mean validation loss, Top-5/10, context, and both Middle/Rare cross-entropy continued improving, so this is not a global plateau or EOS-recipe-wide regression.

## Per-seed trajectory

| Seed | Tokens (M) | Loss | Top-1 | Top-5 | Top-10 | Semantic | Full context |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 15.360 | 4.4517 | 25.40% | 43.75% | 51.56% | 64.0% | 4.7678 |
| 42 | 15.616 | 4.4287 | 26.81% | 43.91% | 51.90% | 68.0% | 4.7709 |
| 42 | 15.872 | 4.4482 | 25.48% | 44.20% | 52.23% | 68.0% | 4.7679 |
| 123 | 15.360 | 4.4443 | 26.17% | 43.55% | 51.31% | 60.0% | 4.8370 |
| 123 | 15.616 | 4.4448 | 26.33% | 43.76% | 51.81% | 66.0% | 4.8709 |
| 123 | 15.872 | 4.4201 | 26.21% | 43.97% | 52.09% | 51.0% | 4.8392 |
| 2026 | 15.360 | 4.4598 | 25.28% | 43.43% | 51.61% | 62.0% | 4.8157 |
| 2026 | 15.616 | 4.4502 | 25.85% | 43.55% | 51.38% | 54.0% | 4.8045 |
| 2026 | 15.872 | 4.4409 | 26.03% | 44.41% | 51.82% | 58.0% | 4.7954 |

## Mean / standard deviation

| Tokens (M) | Loss | Top-1 | Top-5 | Top-10 | Semantic | Naturalness |
|---:|---:|---:|---:|---:|---:|---:|
| 15.360 | 4.4519 ± 0.0077 | 25.62% ± 0.48 | 43.58% ± 0.16 | 51.49% ± 0.16 | 62.00% ± 2.00 | 71.00% ± 4.36 |
| 15.616 | 4.4412 ± 0.0112 | 26.33% ± 0.48 | 43.74% ± 0.18 | 51.70% ± 0.28 | 62.67% ± 7.57 | 70.67% ± 6.81 |
| 15.872 | 4.4364 ± 0.0146 | 25.90% ± 0.38 | 44.19% ± 0.22 | 52.05% ± 0.21 | 59.00% ± 8.54 | 66.67% ± 6.66 |

## EOS 1.0 versus 1.5

The existing matched seed-42 256k control was sufficient, so no duplicate training pilot was run.

- Validation loss: 4.428535 → 4.428706
- Top-1/5/10 delta: -0.024 / +0.024 / +0.012 pp
- terminal P(EOS): 0.00760 → 0.01302
- premature EOS Top-1: 0.00%
- Rare CE delta: -0.000932
- Semantic/Naturalness delta: -1.0 / -1.0 pp
- Adoption criterion: PASS; greedy runaway improvement: NO

## Rare metric resolution

- Rare CE: 9.6001 → 9.5930
- Arithmetic mean probability: 0.00027297 → 0.00020032
- Mean of per-seed medians: 0.00005814 → 0.00006851
- Resolution: YES. Cross-entropy tracks the geometric mean and the low-probability body; arithmetic mean probability was dominated by a small high-probability tail, especially seed 42. The fixed 112-target bucket and recomputation exactly matched PHASE 44.

## Generation and thermal

- Greedy runaway: 100%
- Median loop onset mean: 17.0 → 14.0
- Attractor: **STATIC** (runaway unchanged; repetition improved overall but onset moved earlier)
- Thermal: **THERMAL_THROTTLING_OBSERVED**; max 83C; software thermal slowdown observed; hardware slowdown not observed
- Throughput: 13387.20 → 12986.04 tok/s (-3.00%)
- Thermal is a throughput contributor, not the LM-quality root cause: clocks stayed in a narrow 1890–1905 MHz range during active thermal flags and all numerical/integrity checks passed.

## Integrity

- Deterministic evaluation: PASS
- Context regression: NO
- Checkpoint integrity: PASS
- Final Blind: unopened; SHA256 PASS
- Corpus exposure: 47.52%
- Parallel CPU evaluation: DISABLED
- pytest: 415 passed, 4 warnings in 84.34s
- Render/Vercel: unchanged
