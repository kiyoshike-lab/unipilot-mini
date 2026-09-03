# UniPilot Foundation v2.6 — PHASE 37

3.072M intermediate gate: **CONTINUE_TO_5M**.
Final gate: **CONTINUE_10M_GENERATION_LAG**.
Language Emergence: **PARTIAL**.

## Three-seed learning curve

| tokens | validation loss (mean ± std) | Top-1 | Top-5 | Top-10 | corpus exposure |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048000 | 5.7305 ± 0.0343 | 14.38% ± 0.51% | 25.41% ± 0.85% | 32.13% ± 0.67% | 6.1312% |
| 2560000 | 5.6269 ± 0.0145 | 14.99% ± 0.41% | 27.54% ± 0.63% | 34.55% ± 0.31% | 7.6640% |
| 3072000 | 5.4937 ± 0.0154 | 16.45% ± 0.21% | 29.39% ± 0.42% | 36.21% ± 0.19% | 9.1968% |
| 3584000 | 5.3923 ± 0.0234 | 17.16% ± 0.06% | 30.62% ± 0.23% | 37.74% ± 0.45% | 10.7297% |
| 4096000 | 5.3235 ± 0.0409 | 18.15% ± 0.58% | 32.10% ± 0.39% | 38.95% ± 0.52% | 12.2625% |
| 4608000 | 5.2456 ± 0.0060 | 18.65% ± 0.22% | 33.00% ± 0.24% | 40.04% ± 0.25% | 13.7953% |
| 5120000 | 5.1803 ± 0.0231 | 18.62% ± 0.38% | 33.43% ± 0.21% | 41.07% ± 0.28% | 15.3281% |

## Required comparisons

2.048M → 3.072M: loss -0.2368; Top-1/5/10 2.08%/3.99%/4.08%.
3.072M → 5.120M: loss -0.3134; Top-1/5/10 2.17%/4.04%/4.86%.

| tokens | h32 loss / Top-10 | sampling natural / semantic | greedy rep-1 / runaway | full-vs-last-1 advantage |
| ---: | ---: | ---: | ---: | ---: |
| 2048000 | 5.8052 / 30.28% | 54% / 32% | 0.929 / 100% | 0.2131 |
| 2560000 | 5.6923 / 32.17% | 36% / 18% | 0.912 / 100% | 0.4280 |
| 3072000 | 5.5715 / 33.94% | 50% / 28% | 0.913 / 100% | 0.5410 |
| 3584000 | 5.4970 / 35.50% | 38% / 22% | 0.904 / 100% | 0.4065 |
| 4096000 | 5.3956 / 36.69% | 44% / 24% | 0.894 / 100% | 0.4144 |
| 4608000 | 5.3064 / 38.33% | 34% / 12% | 0.913 / 100% | 0.4927 |
| 5120000 | 5.2612 / 38.64% | 68% / 32% | 0.908 / 100% | 0.8697 |

Frequency learning (outside Top-1% Top-10): 17.72% → 28.61%.
Checkpoint integrity: **PASS** (18/18); bitwise resume: **PASS**; synthetic smoke: **PASS**.
Final Blind SHA256: `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b`; content was not opened.

## Decision

Next token budget: **7M intermediate checkpoint toward 10M**. Foundation Base complete: **NO**. Architecture, corpus, tokenizer, Campus, production, Render, and Vercel were unchanged.
