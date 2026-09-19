# PHASE60 / Foundation v4.9 — Continuation instability audit

Gate: **MIXED_CONTINUATION_INSTABILITY**. Zero new training, zero inference, zero optimizer steps. Six existing checkpoint SHA and strict reload checks passed.

All three seeds are clipped at every step and show validation improvement with Core/normal CE worsening. Exposure-associated forgetting replicates within seeds; seed123 normal failures persist under every within-family leave-one-out. These streams support a mixed short-continuation instability signature, but do not isolate data order, parent weights, moments, or clipping causally.

This is an observational comparison of three seeds. Clipping, data exposure, parent weights and moments were not independently randomized. No dominant causal mechanism is established.

## Failure matrix

| Metric delta (candidate − own parent) | seed42 | seed123 | seed2026 |
|---|---:|---:|---:|
| validation.ce | -0.00067371 | -0.00197846 | -0.00661671 |
| validation.top1 | +0.00067907 | +0.00037488 | +0.00110750 |
| validation.top5 | +0.00024849 | +0.00076904 | +0.00079903 |
| validation.top10 | +0.00109036 | +0.00095969 | +0.00162590 |
| context.128.ce | -0.00869244 | +0.00958221 | -0.01311825 |
| context.512.ce | -0.00493910 | -0.00460949 | -0.01521169 |
| eos.terminal_probability | +0.00371696 | +0.00881132 | -0.00094050 |
| eos.terminal_top1 | +0.00000000 | +0.00000000 | +0.00000000 |
| eos.nonterminal_probability | +0.00004932 | +0.00020065 | -0.00003578 |
| eos.premature_argmax | +0.00000000 | +0.00000000 | +0.00000000 |
| core.micro_ce | +0.10640996 | +0.00154863 | +0.17651043 |
| core.macro_ce | +0.12327609 | +0.01879182 | +0.15695181 |
| core.top1 | +0.00045086 | -0.00180343 | +0.00022543 |
| core.top5 | +0.00022543 | -0.00247971 | -0.00811542 |
| core.top10 | +0.00338142 | -0.00180343 | -0.00653742 |
| supported_tail.micro_ce | +0.09414037 | +0.03691404 | +0.16303213 |
| supported_tail.macro_ce | +0.08368771 | +0.03373076 | +0.17800171 |
| supported_tail.top1 | +0.00130890 | +0.00065445 | +0.00065445 |
| supported_tail.top5 | +0.00327225 | +0.00196335 | -0.00065445 |
| supported_tail.top10 | +0.00327225 | -0.00196335 | -0.00130890 |
| normal.mean_ce | +0.04186515 | +0.12068934 | +0.03388760 |
| normal.terminal_probability | +0.00017774 | +0.00034193 | +0.00031635 |
| normal.math.ce | +0.05195177 | -0.06456327 | +0.15056646 |
| normal.code.ce | -0.00840944 | +0.31767452 | +0.08898020 |
| normal.lists.ce | -0.01847768 | +0.13207960 | +0.05631101 |
| normal.definitions.ce | +0.09190452 | +0.01671875 | -0.02725399 |
| normal.terminology.ce | +0.09235656 | +0.20153713 | -0.09916568 |
| context.long_benefit | -0.00375335 | +0.01419170 | +0.00209343 |

All 34 existing comparisons, parent/candidate values, and 10,000 paired-document bootstrap intervals are in `phase60/three-seed-failure-matrix.json`. BORDERLINE means exact equality to the original inclusive margin; no additional tolerance exists.
Seed42 calculations are descriptive diagnostics only. PHASE57 stays EXPERIMENT_INVALID and Arm A was not rescored. PHASE59 stays CONTROL_STABILITY_MIXED.

- seed42 failed checks: core.micro_ce, core.macro_ce, core.paired_ce_ci95_upper.
  Core bootstrap: {'mean': 0.10640995980998731, 'lower': 0.05227659745522194, 'upper': 0.16010384429745586}.

- seed123 failed checks: normal.mean_ce, normal.code.ce, normal.lists.ce, normal.terminology.ce.
  Core bootstrap: {'mean': 0.001548634134938418, 'lower': -0.057131482412248545, 'upper': 0.060703928889190635}.

- seed2026 failed checks: eos.terminal_probability, core.micro_ce, core.macro_ce, core.paired_ce_ci95_upper, normal.math.ce.
  Core bootstrap: {'mean': 0.17651043120499557, 'lower': 0.13066227658526477, 'upper': 0.22267614135722386}.

## Gradients and endpoint update vectors

| Seed | Clip rate | Mean | Median | P90 | P95 | Max | Net update norm | Relative norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 100% | 4.028544 | 3.959335 | 4.658792 | 4.944962 | 5.398579 | 2.098723 | 0.012917 |
| 123 | 100% | 4.068165 | 3.943645 | 4.554118 | 4.824681 | 7.577708 | 2.110997 | 0.012831 |
| 2026 | 100% | 4.120650 | 3.905410 | 4.747050 | 5.076167 | 14.615757 | 2.123238 | 0.012877 |

Post-clip norms and scales are derived from historical raw norms using min(1,1/(norm+1e-6)); the old runs did not log measured post-clip norms. Endpoint displacement/122 is not mean per-step path length. Tied head is counted once.

| Component net update norm | seed42 | seed123 | seed2026 |
|---|---:|---:|---:|
| embedding_tied_head | 0.666593 | 0.684116 | 0.730816 |
| attention | 1.130201 | 1.133696 | 1.128585 |
| FFN | 1.621348 | 1.627435 | 1.626648 |
| LayerNorm | 0.069466 | 0.069147 | 0.067789 |
| position_embedding | 0.222130 | 0.222944 | 0.223078 |

| Transformer layer net update norm | seed42 | seed123 | seed2026 |
|---|---:|---:|---:|
| 0 | 0.614937 | 0.616369 | 0.614527 |
| 1 | 0.612682 | 0.612163 | 0.603818 |
| 2 | 0.600470 | 0.598871 | 0.593569 |
| 3 | 0.599853 | 0.616592 | 0.610975 |
| 4 | 0.621579 | 0.630782 | 0.625380 |
| 5 | 0.635703 | 0.638001 | 0.640498 |
| 6 | 0.642898 | 0.642873 | 0.642322 |
| 7 | 0.649206 | 0.647890 | 0.649813 |
| 8 | 0.647390 | 0.646273 | 0.654107 |
| 9 | 0.624802 | 0.622285 | 0.624725 |

- Update cosine 42:123: -0.00156344; norm ratio 0.99418546.

- Update cosine 42:2026: 0.00355259; norm ratio 0.98845382.

- Update cosine 123:2026: 0.00020151; norm ratio 0.99423484.

Full per-layer cosines, first/second moment norms before/after, and parameter-relative adaptive directions are in `phase60/optimizer-state-audit.json`. Coordinates belong to different parent weights; these cosines are local direction similarities only.

Optimizer classification: OPTIMIZER_EVIDENCE_INSUFFICIENT. Parent/candidate global adaptive relative-direction max/min ratios are 1.0315/1.0091. Position embeddings have the largest component-relative displacement (~3.4%), consistently across all seeds; a seed-specific optimizer explosion is not demonstrated.

Historical scheduler learning_rate and peak_learning_rate metadata remain 1e-4, while all six actual optimizer groups hold 5e-5. The frozen loops never call a scheduler. This shared stale metadata does not explain between-seed failures; future execution must assert the actual group LR at every update.

## Exposure and forgetting

| Seed | Core occurrences | Tail occurrences | EOS density | Rare share | Core zero exposure | Worsened tokens | Spearman | Zero/nonzero mean CE delta |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 393 | 336 | 0.000336 | 0.013272 | 32.20% | 73.31% | -0.419989 | +0.248874 / +0.063617 |
| 123 | 429 | 417 | 0.000416 | 0.015145 | 27.54% | 51.69% | -0.365184 | +0.124014 / -0.021205 |
| 2026 | 322 | 345 | 0.000192 | 0.011735 | 39.83% | 72.88% | -0.371063 | +0.265508 / +0.085091 |

FREQUENCY_FORGETTING_SIGNATURE: YES (observational). Spearman, zero/nonzero split and exposure quartiles reuse the same token-level evidence and do not count as independent dominance proof. All three token CE distributions, full4096-ID target/input histograms, per-token Core exposure, EOS/special/repetition density and BOS document representation are retained in the audit artifacts; token raw is on Z:.

Consumed122-block sets share zero blocks across every pair of seeds. BOS-derived document counts are 142/146/129; category labels are unavailable. Family exposure proxies count literal substrings only and are not authoritative categories.

Seed2026 worsened 72.88% of 236 Core types; CE delta median/P75/P90 = 0.111279 / 0.263374 / 0.480230.

## Normal controls and EOS/generation

| Family delta | seed42 | seed123 | seed2026 |
|---|---:|---:|---:|
| math | +0.051952 | -0.064563 | +0.150566 |
| code | -0.008409 | +0.317675 | +0.088980 |
| lists | -0.018478 | +0.132080 | +0.056311 |
| definitions | +0.091905 | +0.016719 | -0.027254 |
| terminology | +0.092357 | +0.201537 | -0.099166 |

seed42 has no normal-control safeguard failure. seed123 fails mean/code/lists/terminology; each failed family remains failed after removal of every individual example (NORMAL_FAILURE_BROAD). seed2026 math remains failed in every leave-one-out (NORMAL_FAILURE_FAMILY_SPECIFIC). The full 20 paired examples, unchanged margins, individual positive mass shares and leave-one-out deltas are in `phase60/normal-control-audit.json`.

| Candidate generation | seed42 | seed123 | seed2026 |
|---|---:|---:|---:|
| Sampling runaway_rate | 0.996667 | 0.990000 | 0.996667 |
| Sampling eos_completion_rate | 0.003333 | 0.010000 | 0.003333 |
| Sampling naturalness_rate | 0.723333 | 0.646667 | 0.713333 |
| Sampling semantic_rate | 0.660000 | 0.546667 | 0.623333 |
| Sampling topic_retention_rate | 0.186313 | 0.206559 | 0.192083 |
| Sampling japanese_validity_rate | 0.786667 | 0.786667 | 0.806667 |

Greedy runaway remains 100% for all three candidates. Terminal/nonterminal EOS deltas are in the failure matrix above. Generation did not become safe and this audit does not measure anti-loop efficacy.

## PHASE61 preregistration

CREATED: fresh Control 5e-5 plus exactly one half-LR continuation arm at 2.5e-5. Three seeds42/123/2026; each arm starts from its unchanged16.384M parent,122 updates,62,464 LM gradient tokens+15 matched no-grad slots=63,904 conservative positions. Six runs total:374,784 LM tokens;383,424 conservative positions. No extension.
The mechanistic rule is one fixed half-scale LR intervention before any candidate outcome. Preserve moments/permutation/RNG, EOS1.5, clip1.0, CUDA FP32, no anti-loop; no sweep. Success requires every half-LR seed to pass all34 unchanged safeguards against both own parent and fresh matched control. All-six execution validity is required; failed seeds cannot be averaged away.
Schema SHA256: 6576428940779e2908a8aacc67c4fb6bd2ca8bdc945e6202b88d5116ec701c6b. Full thresholds, identities, run order, runtime LR authority and decision rules are frozen in `phase60/phase61-continuation-stability-preregistration.json`. **training_authorized=false**. No PHASE61 training was run.

## Test isolation and final checks

The mutating Campus test now reads/writes temporary outputs via a pytest-only monkeypatch. Session checksums guard all six repository output files, including the report. The five current dirty JSONs were not restored, edited or staged. Production evaluator paths/model logic are unchanged.

- targeted pytest: exit0, passed229, failed0, errors0, skipped0, warnings3. Log and JUnit SHA are in `phase60/pytest-targeted.json`.

- full pytest: exit0, passed755, failed0, errors0, skipped4, warnings4. Log and JUnit SHA are in `phase60/pytest-full.json`.

All current dirty files (179 expanded paths;151 collapsed status entries), protected4 and READY5, old PHASE57/58/59 outputs, and six checkpoints match their starting SHA. Raw/checkpoint binaries are excluded from Git. No main, Render or Vercel production operation was performed.

Approved LR remains FORMAL_LR_APPROVED_5E5; Generation Policy UNSAFE; PHASE57 EXPERIMENT_INVALID; PHASE59 CONTROL_STABILITY_MIXED. Canonical promotion/20M/Foundation Base: NO. Reserve2 SEALED/UNSCORED; Final Blind SHA ONLY.
