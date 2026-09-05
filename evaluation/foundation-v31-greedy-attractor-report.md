# Foundation v3.1 Greedy attractor research

All five 256k-token seed-42 arms completed from the same official 15.360M checkpoint and optimizer state. Official checkpoints were not modified.

| Arm | EOS weight | lambda | loss | runaway | median onset | repetition-1 | judgment |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 1.0 | 0.0 | 4.428535 | 100% | 18.0 | 0.9252 | SAFE_BUT_NO_EFFECT |
| B | 1.5 | 0.0 | 4.428706 | 100% | 19.0 | 0.9255 | SAFE_BUT_NO_EFFECT |
| C | 1.5 | 0.01 | 4.428706 | 100% | 19.0 | 0.9255 | TOO_WEAK |
| D | 1.5 | 0.03 | 4.428724 | 100% | 19.0 | 0.9250 | SAFE_BUT_NO_EFFECT |
| E | 1.5 | 0.05 | 4.428784 | 100% | 18.0 | 0.9247 | SAFE_BUT_NO_EFFECT |

C/D/E did not improve runaway versus EOS-corrected B (all 100%), and median loop onset remained 19 for B/C/D and regressed to 18 for E. The tiny repetition differences are not sufficient for a SAFE_AND_HELPFUL selection. Three-seed confirmation was therefore not run.

EOS-corrected baseline B retained validation loss 4.428706 and Top-1/Top-5/Top-10 of 26.81%/43.91%/51.90%. Terminal P(EOS) was 0.01302 with Top-1/Top-5/Top-10 of 0%/56.0%/77.8%; non-terminal Top-1 remained 0%. Sampling naturalness/semantic were 71%/66%, so no Japanese-quality regression was found. FIRST_BREAK remained NO.

Experimental checkpoint READY/strict-reload/source-integrity checks passed for 5/5 arms. Full pytest: 396 passed (4 dependency deprecation warnings). Final Blind content was not opened; its SHA-256 matched `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b`.

Gate: **EOS_FIXED_BUT_ATTRACTOR_TRAINING_FIX_FAILED**. Formal 20M permission: **NONE**. Foundation Base completion: **NO**.
