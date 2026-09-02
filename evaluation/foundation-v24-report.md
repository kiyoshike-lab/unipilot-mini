# UniPilot Foundation v2.4 — PHASE 35

Gate: **CONTINUE_2M_GENERATION_LAG**

1.024M: **PASS**

Language Emergence: **PARTIAL**

Formal architecture: **Current** (19,514,880 parameters). Training used the fixed 33,402,759-token clean corpus and vocab 4096 tokenizer.

## Three-seed learning curve

| tokens | val loss mean ± std | top-1 mean ± std | top-5 mean ± std | top-10 mean ± std | corpus / epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512000 | 6.3786 ± 0.0154 | 9.05% ± 0.09% | 17.24% ± 1.11% | 22.83% ± 1.09% | 1.5328% / 0.015328 |
| 640000 | 6.2579 ± 0.0105 | 9.31% ± 0.19% | 18.99% ± 1.52% | 24.62% ± 1.69% | 1.9160% / 0.019160 |
| 768000 | 6.1653 ± 0.0089 | 9.66% ± 0.23% | 19.15% ± 0.95% | 26.75% ± 0.37% | 2.2992% / 0.022992 |
| 896000 | 6.1112 ± 0.0170 | 11.65% ± 1.42% | 20.25% ± 1.60% | 26.55% ± 0.76% | 2.6824% / 0.026824 |
| 1024000 | 6.0557 ± 0.0253 | 10.93% ± 1.05% | 21.22% ± 0.50% | 27.47% ± 0.35% | 3.0656% / 0.030656 |

### 1.024M results by seed

| seed | val loss | top-1 | top-5 | top-10 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 6.0376 | 12.05% | 21.81% | 27.50% |
| 123 | 6.0382 | 11.21% | 21.24% | 27.88% |
| 2026 | 6.0915 | 9.52% | 20.59% | 27.03% |

| tokens | train loss | PPL | LR | grad norm | train tok/s | peak RAM MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512000 | 6.4345 | 589.19 | 1.0e-04 | 2.322 | 880.5 | 1186.3 |
| 640000 | 6.3420 | 522.15 | 1.0e-04 | 2.218 | 880.8 | 1136.1 |
| 768000 | 6.2244 | 475.95 | 1.0e-04 | 2.067 | 879.4 | 1155.5 |
| 896000 | 6.1774 | 450.93 | 1.0e-04 | 1.865 | 880.5 | 1186.2 |
| 1024000 | 6.1258 | 426.69 | 1.0e-04 | 2.458 | 880.6 | 1186.2 |

Validation loss and all aggregate Top-k measures improved from 512k to 1.024M; validation did not regress while training loss improved.

## Teacher-forced horizon at 1.024M

| horizon | loss | top-1 | top-5 | top-10 | correct-token probability |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.1757 | 9.50% | 16.50% | 24.00% | 0.0264 |
| 2 | 6.1101 | 10.50% | 18.25% | 24.75% | 0.0289 |
| 4 | 6.1091 | 12.00% | 19.62% | 25.37% | 0.0342 |
| 8 | 6.1301 | 11.19% | 18.81% | 24.81% | 0.0337 |
| 16 | 6.1498 | 10.44% | 18.16% | 24.62% | 0.0309 |
| 32 | 6.0951 | 10.78% | 18.67% | 25.30% | 0.0335 |

## Free-generation and repetition curve

| tokens | divergence | rep-1/2/3/4 | loop onset / max span | greedy natural / semantic / runaway | sampling t0.7 natural / semantic / runaway |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512000 | 1.155 | 0.949/0.941/0.933/0.926 | 4.51 / 58.28 | 0% / 0% / 100% | 12% / 2% / 100% |
| 640000 | 1.160 | 0.947/0.940/0.933/0.926 | 5.11 / 58.34 | 0% / 0% / 100% | 48% / 28% / 98% |
| 768000 | 1.175 | 0.934/0.911/0.889/0.865 | 11.63 / 48.47 | 0% / 0% / 100% | 58% / 24% / 100% |
| 896000 | 1.155 | 0.942/0.930/0.918/0.904 | 6.89 / 53.80 | 0% / 0% / 100% | 30% / 10% / 100% |
| 1024000 | 1.145 | 0.940/0.927/0.913/0.895 | 9.12 / 49.88 | 0% / 0% / 100% | 42% / 16% / 96% |

Greedy repetition improved only slightly and remains severe. Greedy still diverges near the first token and runs away on every probe; sampling shows partial Japanese structure. This is generation lag rather than an inference-path or training failure.

### Prefix completion at 1.024M

| set | examples | divergence | character validity | Japanese ratio | natural | semantic | boundary | runaway |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 52 | 1.212 | 84.62% | 41.58% | 0.00% | 0.00% | 15.38% | 100.00% |
| validation | 200 | 1.145 | 89.50% | 43.17% | 0.00% | 0.00% | 14.00% | 100.00% |
| sentence | 50 | 1.140 | 90.00% | 49.86% | 0.00% | 0.00% | 14.00% | 100.00% |

## Frequency learning

### Outside the Top-1% frequency bucket

| tokens | top-1 | top-5 | top-10 | generated Top-1% share | distribution JS divergence |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512000 | 0.17% | 3.45% | 6.47% | 99.36% | 0.5519 |
| 640000 | 0.27% | 5.46% | 8.48% | 96.45% | 0.5563 |
| 768000 | 0.40% | 5.01% | 10.57% | 82.91% | 0.5044 |
| 896000 | 3.18% | 6.36% | 10.71% | 87.66% | 0.5139 |
| 1024000 | 2.28% | 7.44% | 11.76% | 74.90% | 0.5056 |

### Validation frequency buckets at 1.024M

| bucket | targets | top-1 | top-5 | top-10 | correct probability | cross entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top_1_percent | 2072 | 36.45% | 61.90% | 73.87% | 0.12683 | 3.5223 |
| top_5_percent_excluding_top_1 | 1558 | 8.79% | 25.78% | 38.55% | 0.01650 | 4.9236 |
| top_20_percent_excluding_top_5 | 2208 | 0.12% | 2.42% | 5.28% | 0.00248 | 6.7772 |
| middle_20_to_80_percent | 2242 | 0.00% | 0.01% | 0.10% | 0.00042 | 8.2411 |
| rare_bottom_20_percent | 112 | 0.00% | 0.00% | 0.00% | 0.00003 | 10.7055 |

## Punctuation and boundary distribution at 1.024M

| token | actual | Top-1 predicted | mean probability | generated |
| --- | ---: | ---: | ---: | ---: |
| 。 | 2.710% | 6.628% | 0.02422 | 0.219% |
| 、 | 3.308% | 18.384% | 0.03025 | 5.906% |
| の | 1.245% | 16.492% | 0.01318 | 6.000% |
| に | 1.331% | 0.464% | 0.01004 | 0.492% |
| は | 1.526% | 2.466% | 0.01167 | 4.039% |
| を | 0.903% | 13.647% | 0.01233 | 0.500% |
| が | 1.501% | 0.378% | 0.01123 | 0.312% |
| と | 0.818% | 0.342% | 0.00602 | 0.000% |
| で | 0.659% | 0.000% | 0.00577 | 0.000% |

| boundary | actual | Top-1 predicted | mean probability | generated |
| --- | ---: | ---: | ---: | ---: |
| 。 | 2.398% | 6.628% | 0.02422 | 0.219% |
| ！ | 0.143% | 0.037% | 0.00179 | 0.000% |
| ？ | 0.006% | 0.000% | 0.00003 | 0.000% |
| newline | 3.108% | 5.505% | 0.01959 | 26.266% |
| <EOS> | 0.031% | 0.000% | 0.00041 | 0.000% |

EOS exposure is measured from the actual training stream:

| seed | input EOS | supervised EOS targets | input BOS | supervised BOS targets |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 330 | 329 | 330 | 330 |
| 123 | 335 | 335 | 335 | 335 |
| 2026 | 294 | 295 | 293 | 294 |

## Context utilization (representative seed 42)

| tokens | full loss | last-64 | last-16 | last-2 | last-1 | full vs last-1 advantage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512000 | 6.3319 | 6.2854 | 6.3420 | 6.5760 | 6.9839 | 0.6520 |
| 640000 | 6.2613 | 6.2113 | 6.2466 | 6.4457 | 7.0281 | 0.7668 |
| 768000 | 6.1424 | 6.0767 | 6.1553 | 6.3558 | 6.5694 | 0.4270 |
| 896000 | 5.8229 | 5.7832 | 5.8015 | 6.1463 | 6.5499 | 0.7270 |
| 1024000 | 6.0828 | 5.9464 | 5.9850 | 6.3641 | 6.6711 | 0.5884 |

## Regression, integrity, and observational probes

Checkpoint integrity: **PASS** (11/11). Bitwise resume reproducibility: **PASS**.
Synthetic smoke: **PASS**. Context maintained: **PASS**. Prior inference/KV-cache parity: **PASS**.

| tokens | knowledge keyword hit rate | role |
| ---: | ---: | --- |
| 512000 | 0.00% | observational only |
| 640000 | 0.00% | observational only |
| 768000 | 0.00% | observational only |
| 896000 | 0.00% | observational only |
| 1024000 | 0.00% | observational only |

Knowledge completion is not a gate before instruction tuning. Human-readable fixed-prefix examples are saved in `evaluation/foundation-v24-generation-examples.json`.
Final Blind SHA256: `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b`. Its content was not opened.

## Decision

Gate: **CONTINUE_2M_GENERATION_LAG**. Formal architecture: **Current**. 1.024M: **PASS**. Language Emergence: **PARTIAL**.
Next token budget: **2M**. Full training continuation: **YES**.
Foundation Base is not complete. Architecture, tokenizer, corpus, Campus, production, Render, and Vercel were unchanged.
