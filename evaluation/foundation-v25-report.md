# UniPilot Foundation v2.5 — PHASE 36

Gate: **CONTINUE_5M_GENERATION_LAG**

Language Emergence: **PARTIAL**

## Three-seed learning curve

| tokens | val loss mean ± std | top-1 mean ± std | top-5 mean ± std | top-10 mean ± std | corpus / epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024000 | 6.0557 ± 0.0253 | 10.93% ± 1.05% | 21.22% ± 0.50% | 27.47% ± 0.35% | 3.0656% / 0.030656 |
| 1280000 | 5.9674 ± 0.0260 | 12.08% ± 1.07% | 22.46% ± 0.28% | 28.80% ± 0.40% | 3.8320% / 0.038320 |
| 1536000 | 5.8816 ± 0.0333 | 13.22% ± 0.30% | 23.11% ± 0.66% | 30.01% ± 0.61% | 4.5984% / 0.045984 |
| 1792000 | 5.7818 ± 0.0255 | 13.68% ± 0.22% | 24.78% ± 0.45% | 31.62% ± 0.54% | 5.3648% / 0.053648 |
| 2048000 | 5.7305 ± 0.0343 | 14.38% ± 0.51% | 25.41% ± 0.85% | 32.13% ± 0.67% | 6.1312% / 0.061312 |

### 2.048M results by seed

| seed | val loss | top-1 | top-5 | top-10 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 5.7473 | 14.40% | 25.21% | 32.13% |
| 123 | 5.6828 | 14.99% | 26.54% | 32.95% |
| 2026 | 5.7615 | 13.73% | 24.48% | 31.31% |

## Improvement rate

| interval | loss / M tokens | Top-1 / M | Top-5 / M | Top-10 / M |
| --- | ---: | ---: | ---: | ---: |
| 512000-1024000 | -0.6306 | 3.67% | 7.77% | 9.07% |
| 1024000-1536000 | -0.3402 | 4.49% | 3.70% | 4.97% |
| 1536000-2048000 | -0.2950 | 2.25% | 4.48% | 4.13% |

## Teacher-forced and free-running generation

| tokens | h32 loss/top-10 | divergence | rep-1/2/3/4 | greedy natural/semantic/runaway | sampling natural/semantic/runaway |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024000 | 6.0951 / 25.30% | 1.145 | 0.940/0.927/0.913/0.895 | 0%/0%/100% | 42%/16%/96% |
| 1280000 | 6.0147 / 26.41% | 1.130 | 0.936/0.919/0.900/0.880 | 0%/0%/100% | 36%/18%/100% |
| 1536000 | 5.9305 / 27.83% | 1.160 | 0.931/0.912/0.891/0.869 | 0%/0%/100% | 44%/18%/98% |
| 1792000 | 5.8617 / 28.83% | 1.130 | 0.921/0.898/0.874/0.849 | 0%/0%/100% | 32%/12%/100% |
| 2048000 | 5.8052 / 30.28% | 1.190 | 0.929/0.917/0.904/0.891 | 0%/0%/100% | 54%/32%/100% |

### Teacher-forced horizon at 2.048M

| horizon | loss | top-1 | top-5 | top-10 | correct-token probability |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.9290 | 12.50% | 20.50% | 30.00% | 0.03775 |
| 2 | 5.8204 | 13.00% | 23.25% | 30.75% | 0.04190 |
| 4 | 5.8389 | 13.50% | 23.75% | 31.13% | 0.04557 |
| 8 | 5.8733 | 13.12% | 23.12% | 29.88% | 0.04473 |
| 16 | 5.8724 | 12.12% | 22.41% | 29.62% | 0.04233 |
| 32 | 5.8052 | 12.78% | 23.03% | 30.28% | 0.04604 |

Repetition trend: **SLOW_IMPROVEMENT**.

## Frequency, punctuation, boundary, and context

| tokens | outside Top-1% Top-1/5/10 | generated Top-1% share | JS divergence |
| ---: | ---: | ---: | ---: |
| 1024000 | 2.28%/7.44%/11.76% | 74.90% | 0.5056 |
| 1280000 | 3.63%/9.27%/13.53% | 75.90% | 0.4986 |
| 1536000 | 4.91%/10.02%/15.03% | 59.36% | 0.4910 |
| 1792000 | 5.33%/11.85%/16.80% | 73.06% | 0.4745 |
| 2048000 | 6.23%/12.71%/17.72% | 80.75% | 0.5000 |

### Validation frequency buckets at 2.048M

| bucket | targets | top-1 | top-5 | top-10 | correct probability | cross entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top_1_percent | 2072 | 38.45% | 62.90% | 74.68% | 0.13646 | 3.3477 |
| top_5_percent_excluding_top_1 | 1558 | 21.78% | 38.90% | 50.32% | 0.08777 | 4.2322 |
| top_20_percent_excluding_top_5 | 2208 | 1.36% | 6.52% | 11.41% | 0.00487 | 6.4964 |
| middle_20_to_80_percent | 2242 | 0.52% | 1.25% | 2.17% | 0.00096 | 7.9843 |
| rare_bottom_20_percent | 112 | 0.00% | 0.00% | 0.00% | 0.00005 | 10.4428 |

### Nine-token punctuation at 2.048M

| token | actual | Top-1 predicted | mean probability | generated |
| --- | ---: | ---: | ---: | ---: |
| 。 | 2.710% | 8.044% | 0.02922 | 0.539% |
| 、 | 3.308% | 17.346% | 0.03886 | 4.188% |
| の | 1.245% | 7.214% | 0.01385 | 0.164% |
| に | 1.331% | 1.990% | 0.01072 | 0.016% |
| は | 1.526% | 2.405% | 0.01275 | 0.547% |
| を | 0.903% | 6.396% | 0.01188 | 0.102% |
| が | 1.501% | 4.797% | 0.01455 | 0.852% |
| と | 0.818% | 0.574% | 0.00689 | 0.172% |
| で | 0.659% | 0.159% | 0.00710 | 0.000% |

| boundary | actual | Top-1 predicted | mean probability | generated |
| --- | ---: | ---: | ---: | ---: |
| 。 | 2.398% | 8.044% | 0.02922 | 0.539% |
| ！ | 0.143% | 0.171% | 0.00157 | 0.008% |
| ？ | 0.006% | 0.000% | 0.00003 | 0.000% |
| newline | 3.108% | 5.164% | 0.02313 | 45.742% |
| <EOS> | 0.031% | 0.000% | 0.00040 | 0.000% |

| tokens | full loss | last-64 | last-16 | last-2 | last-1 | full vs last-1 advantage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024000 | 6.0828 | 5.9464 | 5.9850 | 6.3641 | 6.6711 | 0.5884 |
| 1280000 | 6.2513 | 5.9960 | 6.0425 | 6.2053 | 6.2921 | 0.0408 |
| 1536000 | 6.1061 | 5.8474 | 5.9116 | 6.2849 | 6.4324 | 0.3263 |
| 1792000 | 5.8631 | 5.7078 | 5.7672 | 5.9046 | 6.0902 | 0.2272 |
| 2048000 | 5.8971 | 5.7052 | 5.6819 | 5.8516 | 6.1102 | 0.2131 |

EOS training exposure at 2.048M: seed 42: 618 input / 617 supervised; seed 123: 627 input / 627 supervised; seed 2026: 595 input / 597 supervised.

Checkpoint integrity: **PASS** (12/12); bitwise resume: **PASS**; synthetic smoke: **PASS**.

| tokens | knowledge keyword hit rate | role |
| ---: | ---: | --- |
| 1024000 | 0.00% | observational only |
| 1280000 | 0.00% | observational only |
| 1536000 | 0.00% | observational only |
| 1792000 | 0.00% | observational only |
| 2048000 | 0.00% | observational only |

Final Blind SHA256: `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b`; content was not opened.

## Decision

Next token budget: **3M intermediate checkpoint toward 5M**.
Foundation Base is not complete. Architecture, tokenizer, corpus, Campus, production, Render, and Vercel were unchanged.
