# PHASE48 Rare Frequency Stability and LR Confirmation

Gate: **FREQUENCY_INSTABILITY_REVIEW**. Formal LR: **FORMAL_LR_NOT_YET_APPROVED**. Best LR: 5e-05.

Preflight pytest: 430 passed. Final pytest: 437 passed, 0 failed, 4 warnings in 83.15s. Fixed Rare set: 58 token types, 112 occurrences; hash dba4f22a4cc8d9ce52287228e4fbd5759d584daff7579abf9bd310e0144f68ca.

## Exposure audit

Counts use exact target positions from resumed sampler. Correlations use 58 token-type means with average ranks for ties; CE contributions retain occurrence weighting.

| Seed | 256k occurrences | Unseen | Once | 2-4 | 5-9 | 10+ | 512k occurrences | Unseen | rho CE | rho probability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 388 | 2 | 4 | 16 | 22 | 14 | 745 | 0 | -0.5483 | 0.4429 |
| 123 | 436 | 0 | 6 | 10 | 25 | 17 | 851 | 0 | -0.3072 | 0.1311 |
| 2026 | 413 | 0 | 3 | 16 | 26 | 13 | 826 | 0 | -0.3912 | 0.0442 |

Exposure support: False. Frequency classification: FREQUENCY_INSTABILITY_UNRESOLVED. Rare CE seed std: 0.109822 -> 0.206768.
Rare CE baseline-paired delta std: {'256000': 0.16884002132584164, '512000': 0.1740180670955019}. This is supplementary; the prespecified window gate uses raw fixed-population CE std and all-seed Rare safety.
Exposure total-count CV: {'256k': 0.05822217487479658, '512k': 0.06863408997413215}. Zero-exposure tokens alone do not explain seeds123/2026, which had none at 256k.
Mean/median per-token cross-seed exposure CV: {'256k': {'mean': 0.470241450613364, 'median': 0.4330127018922193, 'all_seed_unseen_types': 0}, '512k': {'mean': 0.34363118773389073, 'median': 0.31981060123828875, 'all_seed_unseen_types': 0}}. Historical PHASE47 outlier counts reproduced exactly; their cumulative exposures are also retained.

## LR comparison

| Budget | LR | Loss mean ± std | Top1/5/10 | Middle CE | Rare CE | Rare mean/median P | Natural/Semantic |
|---|---:|---|---|---:|---:|---|---|
| 256000 | 0.0001 | 4.415781 ± 0.005383 | 26.32%/44.50%/52.45% | 6.550395 | 9.814084 | 0.00015938/0.00004822 | 71.33%/60.33% |
| 256000 | 7.5e-05 | 4.384399 ± 0.007336 | 26.81%/44.89%/52.72% | 6.509329 | 9.752567 | 0.00016905/0.00005110 | 77.00%/69.33% |
| 256000 | 5e-05 | 4.362182 ± 0.010212 | 27.06%/45.10%/52.82% | 6.470270 | 9.670131 | 0.00018431/0.00005689 | 75.00%/63.33% |
| 512000 | 5e-05 | 4.347283 ± 0.008019 | 27.29%/45.13%/53.08% | 6.461308 | 9.568183 | 0.00021379/0.00006741 | 69.33%/60.33% |

256k winner: C; extended arms: ['C']. Clear paired multi-metric winner; extend winner only.

## Cumulative safety

- C seed42: failed checks []; Rare CE delta -0.203696.
- C seed123: failed checks ['rare', 'sampling']; Rare CE delta +0.143885.
- C seed2026: failed checks ['eos']; Rare CE delta -0.014492.

## Exposure and outlier evidence by arm

| Seed | Arm / budget | rho CE | rho probability | Low-exposure share of positive CE contributions | Low-exposure population share |
|---|---|---:|---:|---:|---:|
| 42 | A-256000 | -0.5482693294833578 | 0.4429278329048911 | 75.51% | 41.96% |
| 42 | B-256000 | -0.5468195343620729 | 0.4316379389816937 | 78.79% | 41.96% |
| 42 | C-256000 | -0.5044361620930206 | 0.4272885536178391 | 78.46% | 41.96% |
| 42 | C-512000 | -0.2724281692766608 | 0.2363470103391814 | 11.47% | 8.93% |
| 123 | A-256000 | -0.30718425846489544 | 0.13105457350981461 | 49.66% | 28.57% |
| 123 | B-256000 | -0.282252377082901 | 0.17140282268563212 | 52.95% | 28.57% |
| 123 | C-256000 | -0.24761961496491372 | 0.2661254352224288 | 55.84% | 28.57% |
| 123 | C-512000 | -0.21051039556944076 | 0.05885170602131614 | 11.61% | 6.25% |
| 2026 | A-256000 | -0.39119212662421304 | 0.04418534436167349 | 47.04% | 24.11% |
| 2026 | B-256000 | -0.3351703155401277 | 0.017134989784015033 | 46.19% | 24.11% |
| 2026 | C-256000 | -0.25829525287441113 | -0.015182902340266488 | 44.51% | 24.11% |
| 2026 | C-512000 | -0.18578523224406152 | 0.24340977765729677 | 4.89% | 6.25% |

Per-token before/after CE and probabilities, exposure bins and the largest positive contributors are retained in foundation-v37-rare-exposure.json. Low exposure means <=4 target occurrences; an association is not a causal explanation.

## Selected cumulative generation and context

- naturalness: 0.69333333 (sample std 0.015275252).
- semantic: 0.60333333 (sample std 0.035118846).
- sampling_repetition: 0.36895833 (sample std 0.024836051).
- completion: 0.083333333 (sample std 0.015275252).
- topic_retention: 0.19175742 (sample std 0.0080124197).
- terminal_eos: 0.010005807 (sample std 0.0016529735).
- terminal_eos_top1: 0 (sample std 0).
- terminal_eos_top5: 0.454 (sample std 0.048041649).
- terminal_eos_top10: 0.68333333 (sample std 0.030088758).
- nonterminal_eos: 0.00046226476 (sample std 3.58667e-05).
- premature_eos: 0 (sample std 0).
- context: 4.7117622 (sample std 0.029220018).
- context_64: 4.7747051 (sample std 0.0062598371).
- context_16: 4.8902347 (sample std 0.02485711).
- context_2: 5.4567593 (sample std 0.093535505).
- context_1: 5.8958676 (sample std 0.1173304).
- advantage: 1.1841054 (sample std 0.13304812).
- runaway: 1 (sample std 0).
- first_break: 0 (sample std 0).
- loop_onset: 18.166667 (sample std 5.299371).
- repetition1: 0.92057292 (sample std 0.0046727198).
- repetition2: 0.89488189 (sample std 0.0074329207).
- repetition3: 0.87335979 (sample std 0.010258173).
- repetition4: 0.85344 (sample std 0.012041794).
- entropy: 4.8655194 (sample std 0.20334494).
- margin: 0.23839401 (sample std 0.038964879).

Attractor: {'label': 'WORSENING', 'weakening_signals': ['later_loop_onset'], 'worsening_signals': ['higher_confidence_margin', 'lower_loop_entropy']}. 512k rolling deltas: {'loss': -0.08913993338743875, 'top1': 0.013875325520833315, 'top5': 0.00931803385416663, 'top10': 0.0103759765625, 'rare_ce': -0.02476794482640976, 'middle_ce': -0.045256674291519694}. 1.024M: unavailable (not executed).
GPU: {'tokens_per_second': 13004.739012044605, 'peak_vram_mib': 546.36181640625, 'max_temp': 83.0, 'classifications': ['HOT_BUT_STABLE', 'NO_THERMAL_CONCERN', 'SOFTWARE_THERMAL_SLOWDOWN']}. All formal/prior experimental SHA unchanged; new strict-reload checkpoints PASS. Final Blind SHA only.

## Next phase

Formal LR: None; checkpoint interval: 256k; full Gate interval: 256000; next formal target: None.
If approved, a future phase starts from the protected formal 15.872M checkpoint with the selected recipe. Existing experiments remain unpromoted.
CUDA FP32, EOS1.5, repetition auxiliary OFF. CPU parallel evaluation DISABLED. No architecture changes, no promotion, no 20M training/permission. Foundation Base incomplete.
Protected local files and READY markers retained. Render/Vercel unchanged.

## Limitations

Exposure correlations are descriptive across 58 types; they cannot establish causality.
Sampling quality scores are the same automatic proxies, not human ratings.
A singleton 512k LR cannot establish an LR-specific cause without a matched second-LR endpoint.
