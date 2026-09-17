# PHASE 57 — Registered anti-loop experiment

Final status: **EXPERIMENT_INVALID**. The automatic 64k result was `CONTROL_DRIFT_REVIEW_REQUIRED`; no 128k continuation was authorized or run. A post-evaluation contract audit then found an automatic-proxy field-name mismatch: the frozen generation runner emits `naturalness_rate` and `semantic_rate`, whereas the Phase57 safety wrapper read `natural_japanese_proxy` and `semantic_coherence_proxy`. The two registered safeguards were consequently not scored. They are not repaired or re-run after outputs have been observed.

This report is an execution record, not a claim that the intervention is ineffective or promising.

## Registered preflight

- Authorization: YES; branch: `foundation-research`; authorized start and origin: `e91bb8f8f8ad6966a9305c748e9684194db11783`.
- Resolver Gate: PASS. `UNIPILOT_CHECKPOINT_ROOT`: `Z:\AI\unipilot-mini\checkpoints`.
- Registration SHA: `c517eceaabe5dc9e0ba060f1cf6197b0d3bed8a4a17017d3c8d9a5618c091afe`.
- Parent SHA: `a55369c0e98779839750d727517ce6621bc40b5746f975a96bd9cbc0574db2e8`; tokenizer SHA: `741e450a3b976ea21ccaf452c65181da47f198822f50c2e2b6bfb059e93aa13c`; training manifest SHA: `50ea18676e17091b08e974547bf1b4111b75a07fca231f52d87a51506dc4975a`.
- Parent strict model/optimizer/scheduler/sampler/RNG integrity: PASS. Objective tests: 14 passed. Fixed cache: 128 episodes, 651 negative events; sufficiency PASS.
- CUDA: RTX 2070 SUPER, FP32, TF32 disabled; start 56°C; C free 129.45 GiB; Z free 73.04 GiB.

## 64k execution

Each arm used 62,464 LM tokens plus 1,440 charged auxiliary positions: 63,904 gradient-input tokens. Both new Z: checkpoints strict-reloaded with model and optimizer integrity PASS.

| Metric | Control | Arm A |
| --- | ---: | ---: |
| mean LM / auxiliary / total loss | 4.52195 / 0 / 4.52195 | 4.52207 / 0.03462 / 4.52380 |
| validation CE; Top-1 / 5 / 10 | 4.43069; 25.125% / 43.934% / 52.511% | 4.43175; 25.110% / 43.947% / 52.521% |
| context CE 128 / 512 | 4.52885 / 4.43353 | 4.52795 / 4.43296 |
| Core micro / macro CE | 9.30217 / 9.32888 | 9.30018 / 9.32666 |
| Supported Tail micro / macro CE | 9.89415 / 10.11158 | 9.89140 / 10.10908 |
| terminal EOS probability / Top-1 | 0.01427 / 0% | 0.01587 / 0% |
| nonterminal EOS probability / premature argmax | 0.000388 / 0% | 0.000403 / 0% |
| greedy runaway / loop onset | 100% / 28.28 | 100% / 26.40 |
| sampling runaway, RNG 44000 / 51000 / 52000 | 100% / 99% / 100% | 100% / 98% / 100% |

Control minus Arm A runaway reduction was 0.0 greedy and 0.00333 sampling mean. The automatic gate also found control Core CE drift above its registered 0.1 margin (micro +0.10641, macro +0.12328), independently blocking a continuation.

Normal teacher-forced controls were retained as descriptive evidence: Arm A mean CE 4.28042 and terminal EOS probability 0.00747. The frozen shape proxy was 0/20 (math, code, lists, definitions, terminology each 0/4), is not semantic correctness, and is not used to infer non-degradation by itself.

Hardware: Control 5,774 LM tokens/s, peak VRAM 546.36 MiB, maximum 80°C (`HOT_BUT_STABLE`); Arm A 12,000 LM tokens/s, peak VRAM 623.49 MiB, maximum 74°C (`NO_THERMAL_CONCERN`). No thermal throttle or CUDA error was recorded. Cooldown gates ran before arms/evaluation.

## Boundaries

- 128k executed: NO — the 64k automatic gate did not pass, and the final contract audit invalidated efficacy interpretation.
- Approved LR remains 5e-5. No new arm, architecture, tokenizer, training data, canonical promotion, 20M training, model promotion, or deployment occurred.
- Final Blind remained SHA-only; Reserve2 remained sealed and unscored. The two output checkpoints are EXPERIMENTAL, NOT_CANONICAL, and NOT_PRODUCTION.
- Detailed generation/evaluation raw files and checkpoints remain unstaged under `Z:\AI\unipilot-mini`.
