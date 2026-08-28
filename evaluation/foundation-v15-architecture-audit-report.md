# UniPilot Foundation v1.5 Architecture & Learning Capacity Audit

## Final Gate

- Final: **ARCHITECTURE_FIX_FOUND**
- Architecture audit (current unscaled baseline): **FAIL**
- Context Learning: **FAIL**
- 256k run: **NOT RUN — Gate blocked**
- Next token budget: **STOP**
- Corpus addition: **NO**
- Architecture change: **YES** — tied embedding `sqrt(d_model)` scaling is the next isolated candidate
- 46M / Campus / instruction / human feedback / DPO / production: **not executed**

A specific fix was found: scaling resolves the excessive residual-stream/input scale mismatch and clearly improves the 64k-token loss and Top-1. It does not yet solve arbitrary copy/key binding, so this result does not authorize a 256k, 512k, or 1M full-corpus run.

## Current architecture

- Decoder-only Transformer; 10 layers; hidden 384; 6 heads x 64; FFN 1536.
- Pre-LN LayerNorm (epsilon 1e-05); GELU; learned absolute embedding.
- Attention: manual multi-head causal scaled dot-product self-attention; QK^T / sqrt(head_dim) = QK^T / 8; softmax over keys.
- Q/K/V bias True; output projection bias True; embedding scaling `none`.
- Residual: `x + attention(norm1(x))`, then `x + ffn(norm2(x))`.
- Dropout 0.1 on embeddings, attention probabilities/output, and FFN output.
- Bias-free tied LM head; initialization N(0, 0.02); no residual-specific scaling.
- Parameters: 19,514,880. Full diagram: `evaluation/foundation-v15-architecture.md`.

### Parameter breakdown

| Group | Parameters |
|---|---:|
| token_embedding | 1,572,864 |
| position_embedding | 196,608 |
| attention | 5,913,600 |
| normalization | 16,128 |
| feed_forward | 11,815,680 |
| unique_total | 19,514,880 |

## Static implementation audit

Causal mask, `QK^T/sqrt(head_dim)`, key-axis softmax, learned absolute position indices `0..T-1`, and both Pre-LN residual paths passed exact unit tests. Manual diagnostic forward matches the model forward bit-for-bit on the audit input.

## Activation and attention

| Step | Embedding RMS | Layer 0 output RMS | Layer 9 output RMS | Final hidden RMS | Logits RMS | Health |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.0285 | 0.1789 | 0.6605 | 1.0000 | 0.3888 | FAIL |
| 10 | 0.0285 | 0.1823 | 0.9628 | 1.0001 | 0.3946 | FAIL |
| 100 | 0.0288 | 0.2144 | 4.3243 | 1.0092 | 1.4670 | FAIL |

All measured tensors remained finite and final LayerNorm kept final hidden RMS near 1, but the unscaled baseline residual stream grows from 0.0288 to 4.324 by layer 9 at step 100. This is classified as an unhealthy scale mismatch, not NaN/divergence.

| Step | Mean normalized entropy | Min–max | BOS attention | Previous-token attention | Mean max attention |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9986 | 0.9967–0.9993 | 0.0431 | 0.0349 | 0.0482 |
| 10 | 0.9990 | 0.9967–0.9998 | 0.0432 | 0.0349 | 0.0475 |
| 100 | 0.9981 | 0.9841–1.0000 | 0.0421 | 0.0351 | 0.0485 |

Attention audit is **PASS**: no head is fixed on BOS, the previous token, or one key with the configured collapse thresholds. Full layer/head values are stored in the JSON audit.

## Context sensitivity and ablation

Context Sensitivity Score is **8.516** (mean total variation x100); Top-1 changes for 21.88% of same-final-token pairs. A last-token bigram has exactly zero difference for these pairs.

| Context | Loss | PPL | Top-1 | Mean target probability |
|---:|---:|---:|---:|---:|
| 512 | 7.0124 | 1110.3 | 5.47% | 1.32% |
| 64 | 7.0242 | 1123.5 | 6.25% | 1.16% |
| 16 | 7.0842 | 1192.9 | 3.91% | 0.96% |
| 2 | 7.1185 | 1234.6 | 2.34% | 0.33% |
| 1 | 7.0997 | 1211.6 | 4.69% | 0.34% |

Full-context loss advantage is 0.0873 over last-1 and 0.1061 over last-2. The real-corpus model therefore uses more than bigram context, although usage remains weak.

## Synthetic Context Gate

| Task (independent training) | Accuracy | >90% |
|---|---:|---|
| indexed_copy | 22.50% | FAIL |
| previous_key_lookup | 25.00% | FAIL |
| long_range_dependency | 100.00% | PASS |
| pattern_continuation | 57.50% | FAIL |
| context_conditioned | 100.00% | PASS |

Mixed current overall: 62.50%; mixed scaled overall after 1000 updates: 59.90%. Every input ends in token 8; the last-token bigram baseline is 11.2% on the mixed set. Long-range retrieval and simple context conditioning work, but arbitrary query-to-value binding remains near chance (4 candidates) and pattern selection remains near chance (2 candidates). **CONTEXT LEARNING: FAIL**.

## Architecture ablations — identical 65,536-token stream

| Configuration | Params | Δ params | Loss | Top-1 | Top-5 | Context score | tok/s | RAM MB | Activation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| current_preln_gelu_tied | 19,514,880 | +0.00% | 7.1209 | 5.03% | 12.18% | 3.081 | 1275.7 | 755.3 | FAIL |
| tied_embedding_sqrt_scale | 19,514,880 | +0.00% | 6.9953 | 7.08% | 12.72% | 3.904 | 1242.9 | 756.6 | PASS |
| untied_lm_head | 21,087,744 | +8.06% | 7.1149 | 5.32% | 11.84% | 3.851 | 1351.1 | 799.5 | FAIL |
| pre_rmsnorm | 19,506,816 | -0.04% | 7.1221 | 5.03% | 12.21% | 3.062 | 1333.9 | 777.5 | FAIL |
| fewer_layers_wider | 18,965,184 | -2.82% | 7.1003 | 5.66% | 11.96% | 4.351 | 1397.1 | 711.1 | FAIL |
| more_layers_narrower | 18,736,640 | -3.99% | 7.1345 | 5.18% | 11.62% | 3.050 | 1193.0 | 809.5 | FAIL |

Best short ablation: **tied_embedding_sqrt_scale**. Against current it improves loss by 0.1256, Top-1 by 2.05 points, context score by 0.823, and reduces residual/embedding RMS growth from 156.4x to 5.2x.

RMSNorm, untied head, and depth/width alternatives do not clearly beat scaled tying. GELU and 6x64 heads were not expanded into extra ablations because activation nonlinearity is finite, head dimension is standard, and exact attention tests pass.

## Token-frequency collapse and calibration (128k baseline)

| Frequency bucket | Targets | Accuracy | Mean target probability | Cross entropy |
|---|---:|---:|---:|---:|
| top_1_percent | 2072 | 26.98% | 7.24% | 4.2281 |
| top_5_percent_excluding_top_1 | 1558 | 0.00% | 0.14% | 6.7235 |
| top_20_percent_excluding_top_5 | 2208 | 0.00% | 0.06% | 7.6480 |
| middle_20_to_80_percent | 2242 | 0.00% | 0.02% | 8.9065 |
| rare_bottom_20_percent | 112 | 0.00% | 0.00% | 10.6740 |

| Token | Actual validation | Top-1 predicted | Train frequency | Embedding norm | Norm percentile |
|---|---:|---:|---:|---:|---:|
| 。 | 2.71% | 43.55% | 2.43% | 0.4725 | 89.23% |
| 、 | 3.31% | 31.41% | 3.50% | 0.4623 | 87.55% |
| の | 1.25% | 9.61% | 1.14% | 0.4254 | 69.56% |
| に | 1.33% | 0.00% | 1.05% | 0.4133 | 55.69% |
| は | 1.53% | 0.05% | 1.26% | 0.4248 | 69.12% |
| を | 0.90% | 0.49% | 1.09% | 0.4146 | 57.25% |
| が | 1.50% | 0.00% | 1.19% | 0.4318 | 74.66% |
| <EOS> | 0.01% | 0.00% | 0.03% | 0.4116 | 52.91% |

The LM head has no bias and is weight-tied. `。` is only 2.71% of validation targets but 43.55% of Top-1 predictions; `、` is 3.31% actual but 31.41% predicted. Punctuation vectors are high-norm and aligned with the frequent-token centroid. The collapse is therefore encoded in the tied embedding/hidden geometry and residual scale, not an output-bias parameter.

## Controlled short Japanese corpus

Built 5,000 deterministic 20–197-token sentence segments from the existing clean train split, with source/license/category metadata. A separate 90/10 diagnostic split is used; it is **not** added to the Foundation corpus.

At 65,536 tokens with the scaled candidate: loss 6.7599, Top-1 10.11%, Top-5 15.30%. Natural/Semantic/EOS remain 0%, runaway remains 100%; short clean sentences alone do not fix language emergence at this token budget.

## Bigram fairness

**PASS.** Train and validation packed hashes and document IDs are disjoint (0 overlap). Counts use only the 128k sampled training macroblocks; validation is scoring-only. Add-alpha smoothing uses alpha=0.1, UNK remains smoothed, packed BOS/EOS are included, and tokenizer/vocab are identical. The audited token-matched Bigram remains loss 6.6169 / Top-1 11.67% / Top-5 28.14%.

## 256k and scaling trend

The required Synthetic Gate did not pass, so 256k was not run. Consequently no 128k→256k trend is claimed. The next full-corpus budget is **STOP**, not 512k or 1M.

## Integrity and protection

- Final Blind content was not opened; SHA256 only: `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b` (PASS).
- Diagnostic checkpoint strict reload passed; bitwise interrupted/resumed unit test passed.
- v0.4, Campus v2.3, Render, Vercel, Release, external AI/API, push, and deploy were untouched.
