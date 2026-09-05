# Foundation v3.2 Base Maturity Decision

## Decision

- Scaling: **SLOWING_BUT_HEALTHY**
- Greedy attractor: **WEAKENING**
- Gate: **CONTINUE_SHORT_GPU_GATES**
- 20M GPU continuation permission: **NO**
- Foundation Base completion: **NO**

Greedy runaway remains a serious generation metric, but it is not a sufficient single-condition veto. Loss, Top-k, semantic sampling, context use, frequency learning, and synthetic capability show that standard pretraining remains productive. No architecture replacement is authorized in this phase.

## Scaling history

| Tokens | Validation loss | PPL | Top-1 | Top-5 | Top-10 | Naturalness | Semantic | Greedy rep-1 | Runaway | Median loop onset | Terminal P(EOS) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.120M | 5.1803 | 177.73 | 18.62% | 33.43% | 41.07% | 68.00% | 32.00% | 0.9513 | 100.00% | 6.0 | 0.00703 |
| 7.168M | 4.9047 | 134.92 | 21.62% | 37.39% | 45.11% | 62.00% | 36.00% | 0.9293 | 100.00% | 15.5 | 0.00766 |
| 10.240M | 4.6765 | 107.40 | 23.51% | 40.56% | 48.49% | 72.00% | 40.00% | 0.9316 | 100.00% | 14.5 | 0.00718 |
| 15.360M | 4.4519 | 85.79 | 25.62% | 43.58% | 51.49% | 70.00% | 55.00% | 0.9231 | 100.00% | 20.0 | 0.00931 |

Per-million-token improvement:

| Interval | Loss ↓/M | Top-1 pp/M | Semantic pp/M | Repetition pp ↓/M |
|---|---:|---:|---:|---:|
| 5.120M→7.168M | 0.1346 | 1.466 | 1.953 | 1.072 |
| 7.168M→10.240M | 0.0743 | 0.615 | 1.302 | -0.074 |
| 10.240M→15.360M | 0.0439 | 0.412 | 2.930 | 0.165 |

## Standard continuation pilot

Seed 42 ran for 256k tokens from an isolated copy of the formal 15.360M checkpoint. It used CUDA FP32, EOS weight 1.5, and no repetition auxiliary. Heavy CPU evaluation began only after training exited.

- Validation loss delta: -0.022982
- Top-1 / Top-5 / Top-10 deltas: +1.404 / +0.159 / +0.342 pp
- Sampling semantic: 66.00% → 65.00%
- Terminal P(EOS), matched 256k standard control→EOS 1.5: 0.00760 → 0.01302
- Greedy runaway: 100.00% → 100.00%
- Greedy repetition-1: 0.9231 → 0.9255
- Median loop onset: 20.0 → 19.0
- Middle/Rare learning: PASS
- Full context loss: 4.7299 → 4.8750; advantage vs one-token context: 1.6218 → 1.0409
- Context/Japanese maintained: FAIL (Japanese PASS; short-pilot absolute context loss FAIL)
- Result: **STANDARD_PRETRAINING_PRODUCTIVE_BUT_CONTEXT_GATE_FAILED**
- Thermal: Maximum 82C observed during the 22.5-second pilot; post-run temperature returned to 53C and no compute-throttle event was observed. Sustained-above-80 duration was not recorded, so the next run must retain a duration-aware thermal trace.

## Greedy versus sampling

At pilot, greedy semantic/runaway were 1.00% / 100.00%; temperature 0.7 semantic/naturalness were 65.00% / 73.00%. Teacher-forced metrics improve while free-running greedy progressively repeats and loses diversity. Exposure-bias-like evidence: **YES**.

Historical same-prefix evidence supports ATTRACTOR_WEAKENING: repetition-1 declined and median loop onset moved later from 5.120M to 15.360M, although runaway stayed 100%. Entropy and candidate margins at the current/pilot loop onset do not show a fixed implementation fault; the argmax basin remains.

| Tokens | Loop onset | Rep-1 | Entropy at onset | Top1–Top2 margin | EOS P at onset |
|---:|---:|---:|---:|---:|---:|
| 5,120,000 | 6.0 | 0.9513 | 5.4336 | 0.2032 | 0.002753 |
| 7,168,000 | 15.5 | 0.9293 | 5.5751 | 0.1605 | 0.002257 |
| 10,240,000 | 14.5 | 0.9316 | 5.3285 | 0.1882 | 0.001336 |
| 15,360,000 | 20.0 | 0.9231 | 5.2988 | 0.1743 | 0.000388 |
| 15,616,000,pilot | 19.0 | 0.9255 | 4.8534 | 0.2191 | 0.000859 |

## Architecture, context, frequency, and exposure

The current 10-layer, 384-hidden, 6-head, Pre-LN/GELU, learned-absolute-position, tied-weight model continues to improve LM metrics. Tiny overfit, copy, position, long-range, context-conditioned, and fixed-relation tests pass. Full context retains a positive advantage over one-token context. These do not meet the multi-evidence threshold for an architecture defect.

Corpus exposure is **45.98%** (0.460 epoch), so undertraining remains the most economical explanation. Rare-token evidence is accepted only with the probability and cross-entropy tolerance recorded in the JSON artifact.

## Operational decision

- EOS weight 1.5: SAFE_FOR_CONTINUATION (terminal improvement, non-terminal Top-1 0%, 3-seed reproducibility, no LM/Semantic/Japanese regression)
- Repetition auxiliary: rejected; no further lambda search
- Recommended mode: standard CUDA FP32 continuation with EOS 1.5, repetition auxiliary OFF
- Parallel CPU evaluation: DISABLED
- Evaluation order: training stops, then offline evaluation
- Checkpoint integrity: PASS
- Final Blind: unopened; SHA256 PASS
- pytest: 403 passed, 4 dependency deprecation warnings in 76.26s
- Render/Vercel: unchanged

The next phase may continue only in 256k-512k formal GPU increments until the context-maintenance gate passes; one-shot continuation to 20M is not authorized. Base Completion remains NO and must be judged again at the next formal checkpoint; greedy runaway remains a monitored metric, not the sole completion gate.
