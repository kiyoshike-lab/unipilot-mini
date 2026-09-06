# PHASE49 / TRACK A — Rare Gate v2

Formal LR Gate: **FREQUENCY_GATE_REDESIGN_REQUIRED**. Approved: **False**. Selected LR: None.

Preflight: 437 passed. Final pytest: 443 passed, 0 failed, 4 warnings in 110.48s.

## Fixed population and uncertainty

Core: 236 types, 4436 occurrences, 146 documents.
Tail: unchanged 58 types / 112 occurrences / **2 documents**. SHA dba4f22a4cc8d9ce52287228e4fbd5759d584daff7579abf9bd310e0144f68ca.
Train frequency rank >= ceil(4096*0.8), validation occurrence count >= 10; include every eligible target position, excluding index 0 which has no preceding input. No model-dependent selection. Core and legacy Tail may overlap in token IDs but differ in evaluation support.
2000 deterministic paired document-cluster replicates, seed 4900. Documents identified by BOS boundaries in packed validation. Resample complete document groups with replacement, retain all selected occurrences. Report percentile 95% intervals. Fixed targets/contexts across checkpoints.
The legacy Tail cannot establish population-wide noninferiority. Widening tolerances after seeing results is not permitted. Retain Tail for monitoring; a future phase must preregister broader tail document support without opening Final Blind.

## Fair cumulative 512k comparison

| LR | Validation ± SD | Top1/5/10 | Middle | Core micro / macro CE | Tail CE | Natural/Semantic |
|---|---|---|---:|---|---:|---|
| 7.5e-05 | 4.374239 ± 0.006253 | 26.97%/44.69%/52.70% | 6.490935 | 9.272481/9.285930 | 9.560458 | 70.00%/63.00% |
| 5e-05 | 4.347283 ± 0.008019 | 27.29%/45.13%/53.08% | 6.461308 | 9.250460/9.269114 | 9.568183 | 69.33%/60.33% |

## Document-bootstrap intervals

| LR | Seed | Core CE [95% CI] | Core delta [95% CI] | Tail CE [95% CI] | Tail delta [95% CI] | Failed checks |
|---|---|---|---|---|---|---|
| 7.5e-05 | 42 | 9.22686 [9.028318496574034, 9.398304922154606] | -0.15617 [-0.21994190175473807, -0.10111336849222148] | 9.58205 [9.392974884947126, 10.71647697385574] | -0.23601 [-0.4431164366516146, -0.20148778330714057] | ['frequency_support'] |
| 7.5e-05 | 123 | 9.28294 [9.035054767989138, 9.481315769079778] | -0.24980 [-0.33031135291603825, -0.1517434296149676] | 9.79706 [9.553529840943417, 11.258219941532603] | +0.19298 [-0.013317394597917698, 0.22736367154461765] | ['middle', 'sampling', 'frequency_support'] |
| 7.5e-05 | 2026 | 9.30764 [8.975706664482457, 9.579348897104259] | +0.10247 [-0.03150012345700789, 0.23747540872126854] | 9.30227 [9.111911268166605, 10.444419770320103] | -0.05445 [-0.08104666230196315, 0.10510306750377085] | ['eos', 'sampling', 'core_ci', 'frequency_support'] |
| 5e-05 | 42 | 9.19576 [8.998696246922409, 9.366357349822604] | -0.18727 [-0.2398733117566729, -0.14058573050172204] | 9.61436 [9.422506433412266, 10.765457361047991] | -0.20370 [-0.3941360494593651, -0.1719562348419994] | ['frequency_support'] |
| 5e-05 | 123 | 9.28745 [9.034256547850367, 9.486091721391064] | -0.24529 [-0.31595794158427487, -0.16280710369447157] | 9.74796 [9.512011304368274, 11.163658297351786] | +0.14388 [-0.10787903877873506, 0.18584513496947133] | ['sampling', 'frequency_support'] |
| 5e-05 | 2026 | 9.26817 [8.94871056050664, 9.524162582355848] | +0.06299 [-0.04746036667850228, 0.17343112741691458] | 9.34223 [9.17279388174363, 10.358855841284639] | -0.01449 [-0.020164048724938172, 0.019539138468305683] | ['eos', 'core_ci', 'frequency_support'] |

## Integrity, recipe and next step

GPU: {'mean_tokens_per_second': 12647.811259394492, 'peak_allocated_vram_mib': 546.36181640625, 'max_temp_c': 82.0, 'classifications': ['HOT_BUT_STABLE', 'SOFTWARE_THERMAL_SLOWDOWN']}. Existing 16 and new 3 checkpoints PASS; protected 18 files unchanged; Final Blind SHA only.
CUDA FP32, EOS1.5, repetition auxiliary OFF, AdamW unchanged. CPU parallel evaluation DISABLED. Web code is isolated from model/checkpoint code.
Canonical candidate executed: False. Next token target: None. Next Gate: FREQUENCY_GATE_REDESIGN_REQUIRED. 20M permission: NO. Foundation Base complete: NO.
1e-4 old lineage and 15.872M formal checkpoints remain unchanged. Both 512k LR arms remain experimental.

## Limitations and execution note

Core and Tail bootstrap CIs are conditional on their fixed validation support, not all-language generalization.
Tail occupies only two documents. Its 2000 resamples have few distinct document combinations; more replicates cannot repair missing independent support.
Core uses CUDA FP32 inference with no model updates; all arms and baseline share exact contexts. Tail uses exact legacy CPU probabilities.
Initial training attempt stopped before the first update: integrity model instantiation consumed restored RNG. Verification was moved before RNG restoration; continuity checks passed on all runs.
No architecture, tokenizer, corpus, split, EOS or objective changes. No canonical promotion or 20M permission.
