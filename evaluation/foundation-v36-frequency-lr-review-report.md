# PHASE 47 Foundation v3.6 Frequency and LR Review

Next Gate: **FREQUENCY_INSTABILITY_REVIEW**. LR state: **LR_REDUCTION_HELPFUL**. Frequency: **SEED_LOCAL_VARIANCE**.

## Test gate

Previous failures: test_worker_output_directories_do_not_collide and test_ready_protocol_rejects_missing_marker expected directory separation / READY rejection but raised AttributeError because OUT was removed by existing storage routing. Tests now inspect and stub checkpoint(), preserving both assertions.

Preflight: 422 passed, 0 failed. Final pytest: 430 passed, 0 failed, 4 warnings (77.08s).
The existing checkpoint_paths.py helper and its unchanged regression tests are included as required dependencies: the committed Phase46 runner already imports this helper, which had remained untracked. Other local storage-routing changes are excluded.

A separate legacy API test wrote generated_at into the protected human-results JSON. The captured SHA256 was restored exactly, including CRLF. Both test export paths now point to tmp_path. The protected JSON remains unstaged.

## Fixed-population audit

Train token-frequency ranks; disjoint 0-1%, 1-5%, 5-20%, 20-80%, 80-100% vocabulary ranks, ceil boundaries; 8192 fixed validation targets; occurrence-weighted arithmetic mean of -log P(correct). No checkpoint-dependent membership.
Repeat deterministic: True; Phase46 metric reproduction: True.

| Seed | Rare CE delta | Positions worse | Top10 positive contribution |
|---|---:|---:|---:|
| 42 | +0.043275 | 49.1% | 71.3% |
| 123 | +0.201159 | 60.7% | 65.4% |
| 2026 | +0.418964 | 76.8% | 54.5% |

All positions and token IDs, quantiles, sample counts, top-50 contributors, taxonomy and four-checkpoint trajectories are stored in foundation-v36-rare-analysis.json and phase47/audit/*.json.

## Seed42 LR arms

| Arm | LR | Loss | Top1 / 5 / 10 | Middle CE | Rare CE | Natural / Semantic | EOS | Checks |
|---|---:|---:|---|---:|---:|---|---:|---|
| A | 0.0001 | 4.420785 | 26.18% / 44.54% / 52.66% | 6.507296 | 9.861328 | 77% / 63% | 0.007687 | PASS |
| B | 7.5e-05 | 4.392608 | 26.56% / 44.82% / 52.72% | 6.486149 | 9.836567 | 82% / 74% | 0.008230 | PASS |
| C | 5e-05 | 4.373315 | 26.70% / 44.81% / 52.80% | 6.474388 | 9.794616 | 77% / 66% | 0.009561 | PASS |

## Three-seed confirmation

| Seed | Loss | Rare CE | Rare CE vs baseline | Rare CE vs control | Failed conditions |
|---|---:|---:|---:|---:|---|
| 42 | 4.373315 | 9.794616 | -0.023437 | -0.066712 | none |
| 123 | 4.353251 | 9.586948 | -0.017129 | -0.218288 | rare |
| 2026 | 4.359980 | 9.628829 | +0.272105 | -0.146859 | rare |

| Metric | Mean | Seed std |
|---|---:|---:|
| loss | 4.362182 | 0.010212 |
| ppl | 78.430788 | 0.802158 |
| top1 | 0.270589 | 0.003137 |
| top5 | 0.450968 | 0.003001 |
| top10 | 0.528158 | 0.000462 |
| middle_ce | 6.470270 | 0.006296 |
| rare_ce | 9.670131 | 0.109822 |
| rare_median | 0.000057 | 0.000009 |
| rare_top1 | 0.000000 | 0.000000 |
| rare_top5 | 0.000000 | 0.000000 |
| rare_top10 | 0.005952 | 0.005155 |
| naturalness | 0.750000 | 0.052915 |
| semantic | 0.633333 | 0.037859 |
| terminal_eos | 0.013183 | 0.005477 |
| premature_eos | 0.000000 | 0.000000 |
| context | 4.718918 | 0.015394 |
| advantage | 1.159589 | 0.074504 |
| runaway | 1.000000 | 0.000000 |
| loop_onset | 16.833333 | 3.617089 |
| repetition1 | 0.920703 | 0.000469 |
| entropy | 4.896129 | 0.266663 |
| margin | 0.244670 | 0.054199 |

Attractor versus 15.872M (unchanged PHASE46 descriptive heuristic): {'label': 'WORSENING', 'weakening_signals': ['later_loop_onset'], 'worsening_signals': ['higher_confidence_margin', 'lower_loop_entropy']}.
Paired LM and Rare improvement in all seeds: True. This comparative finding is separate from passing the absolute safety conditions.

Best candidate: C; three-seed confirmation: True; confirmation pass: False.
Same starting model/optimizer/scheduler/sampler/RNG across arms: True; A reproduces historical control: True.

The pre-recorded selection policy tests LM, frequency, context, EOS, sampling, teacher-forced losses, greedy signals and stability. Loss alone is not a selection rule. No experimental checkpoint is promoted.

## Next recipe and operations

Recommended formal LR: None; next interval: 256k (only after the recorded next gate permits continuation). 20M permission: NO. Foundation Base completion: NO.
CUDA FP32; EOS 1.5; repetition auxiliary OFF; parallel CPU evaluation DISABLED. GPU: {'mean_tokens_per_second': 13443.269502484647, 'peak_vram_mib': 546.36181640625, 'max_temperature_c': 84.0, 'thermal_classifications': ['HOT_BUT_STABLE', 'NO_THERMAL_CONCERN', 'SOFTWARE_THERMAL_SLOWDOWN']}.
All six formal checkpoint hashes unchanged; strict model/optimizer, RNG, sampler, update and scheduler checks passed. Final Blind contents unused; SHA verified. Render/Vercel unchanged.

## Limitations

Generation scores are automatic proxies on 100 fixed prefixes.
Rare probe covers 112 target occurrences; per-seed descriptive classification is not proof of population-wide regression.
A 256k LR experiment cannot establish a long-term plateau or optimal learning-rate schedule.
