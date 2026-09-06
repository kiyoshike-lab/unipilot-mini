# Foundation v3.5 Thermal-Aware Short GPU Continuation

## Decision

- Gate 1: **STOP_FREQUENCY_GATE**
- Gate 2 executed: **NO**
- Gate 2: **NOT_EXECUTED**
- Final Gate: **LM_PLATEAU_REVIEW**
- 20M permission: **NO**
- Foundation Base completion: **NO**

## Final metrics

- Validation loss / std: 4.4158 / 0.0054
- Top-1 / Top-5 / Top-10: 26.32% / 44.50% / 52.45%
- Sampling Naturalness / Semantic: 71.33% / 60.33%
- terminal P(EOS): 0.01241; premature EOS Top-1: 0.00%
- Greedy runaway / loop onset / repetition-1: 100.00% / 20.2 / 0.9203
- Attractor: WORSENING ({'label': 'WORSENING', 'weakening_signals': ['later_loop_onset'], 'worsening_signals': ['higher_confidence_margin', 'lower_loop_entropy']})
- Full context loss / advantage: 4.7744 / 1.1858
- Context regression: NO

## Operations

- Rolling 512k: {'validation_loss_change': -0.025457754731178284, 'top1_change': -0.0001220703125}
- Rolling 1.024M: None
- Corpus exposure: 48.28%
- CUDA FP32, EOS weight 1.5, repetition auxiliary OFF, AMP OFF
- Parallel CPU evaluation: DISABLED
- Checkpoint integrity: PASS
- Final Blind: unopened; SHA256 PASS
- pytest: 420 passed, 2 failed (pre-existing Phase 42 checkpoint-path compatibility)
- Render/Vercel: unchanged
