# Foundation v3.3 Context Gate Recovery

## Decision

- Gate 1: **CONTINUE_PLUS_256K**
- Gate 2 executed: **YES**
- Gate 2 interval: **STOP_LM_GATE**
- Final Gate: **CONTINUE_SHORT_GPU_GATES**
- 20M permission: **NO**
- Foundation Base completion: **NO**

The pre-registered context regression rule requires an absolute and relative material delta, a paired 95% confidence interval above zero, and reproduction in at least two seeds. A positive Full Context Advantage and natural ordering against short contexts are separate mandatory checks.

## Expanded multi-seed baseline

| Context | Mean loss | Seed std |
|---:|---:|---:|
| 512 | 4.8068 | 0.0355 |
| 256 | 4.8020 | 0.0205 |
| 128 | 4.8249 | 0.0092 |
| 64 | 4.8750 | 0.0320 |
| 32 | 4.9122 | 0.0240 |
| 16 | 4.9981 | 0.0409 |
| 8 | 5.1148 | 0.0297 |
| 2 | 5.5361 | 0.0335 |
| 1 | 5.9327 | 0.0352 |

Full Context Advantage vs 1: 1.1259 ± 0.0668 across seeds.

## Gate 1

- Full loss: 4.8068 → 4.8154 (delta +0.0086, +0.18%)
- Paired 95% CI: [-0.0403, +0.0574] over 768 targets
- Context regression: NO
- LM Gate: PASS; EOS Gate: PASS

## Gate 2

- Full loss: 4.8154 → 4.8008 (delta -0.0146, -0.30%)
- Paired 95% CI: [-0.0608, +0.0317] over 768 targets
- Context regression: NO
- LM Gate: FAIL; EOS Gate: PASS
- Interval Top-1 check: FAIL; Semantic: 0.6267 → 0.5900

## Overall 15.360M → final

- Context / LM / EOS: PASS / PASS / PASS
- Full-context delta: -0.0060; context regression: NO
- 20M remains denied because every required condition must pass; Gate-2 interval pass=NO, strict Semantic maintenance=NO.

## Final endpoint

- Tokens: 15,872,000
- Validation loss / PPL: 4.4364 / 84.48
- Top-1 / Top-5 / Top-10: 25.90% / 44.19% / 52.05%
- Sampling Naturalness / Semantic: 66.67% / 59.00%
- terminal P(EOS): 0.01351; premature EOS Top-1: 0.00%
- Greedy runaway / repetition / median onset: 100.00% / 0.9189 / 14.0
- Full context loss / advantage: 4.8008 / 1.1440
- Middle CE: 6.5773 → 6.5066; Rare CE: 9.7689 → 9.5930
- Corpus exposure: 47.52%

## Historical context comparison

- 10.240M full loss / advantage: 5.0710 / 0.9706
- 15.360M full loss / advantage: 4.8068 / 1.1259
- 15.872M full loss / advantage: 4.8008 / 1.1440

## Operations

- GPU mean throughput: 12986.04 tok/s
- Peak VRAM: 547.30 MiB
- Max temperature: 83C; longest >80C: 12.90s; THERMAL_ATTENTION=YES
- CUDA FP32, AMP OFF, EOS weight 1.5, repetition auxiliary OFF
- Parallel CPU evaluation: DISABLED
- Checkpoint integrity: PASS
- Final Blind: unopened; SHA256 PASS
- pytest: 410 passed, 4 warnings in 89.91s
- Render/Vercel: unchanged
