# UniPilot Foundation v2.3 — PHASE 34

Gate: **CONTINUE_1M_TOKEN_LIMITED**

Inference parity: **PASS**

KV-cache parity: **PASS**

640k pilot: **EXECUTED** (seed 42 only)

Proceed to 1M in the next phase: **YES**

## Base prefix completion

| tokens | teacher loss h32 | teacher top-1/5/10 h32 | divergence | greedy 1-gram rep. | sampling t0.7 natural |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256000 | 6.8045 | 0.085 / 0.140 / 0.184 | 1.140 | 0.960 | 8% |
| 512000 | 6.4396 | 0.098 / 0.163 / 0.210 | 1.155 | 0.949 | 12% |
| 640000 | 6.3113 | 0.101 / 0.173 / 0.226 | 1.160 | 0.947 | 48% |

## Diagnosis

Training/inference logits and cached/non-cached logits agree within tolerance. The dominant failure is not an inference implementation bug. Teacher-forced learning continues, but the first free-running error occurs near token 1 and drives a high-frequency newline/token loop. Low corpus exposure and scarce EOS targets are contributors. Sampling reveals better candidates, but decoding results are diagnostic only.

640k checkpoint integrity: **PASS**. Synthetic architecture/EOS-capability evidence remains PASS. Final Blind content was not opened.
Foundation Base is not complete. Architecture, tokenizer, corpus, Campus, production, Render, and Vercel were unchanged.
