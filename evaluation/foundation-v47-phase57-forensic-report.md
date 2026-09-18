# PHASE58 — Evaluation contract and PHASE57 forensic audit

PHASE57 remains **EXPERIMENT_INVALID**. Its automatic gate remains `CONTROL_DRIFT_REVIEW_REQUIRED`; 128k was not run. Intervention efficacy is **UNDETERMINED / NO VALID CLAIM**. No old output was passed through the new safety wrapper to revive a formal PHASE57 decision. PHASE58 performed no training.

## Evaluation contract

`generation-eval-v2` defines runaway, sentence completion, naturalness, semantic, topic retention, Japanese validity, EOS completion, mean EOS probability, repetition1–4, unique-token ratio, loop onset and cycle length. Rates and probabilities are in [0,1], never percentages; loop lengths/indices use tokens. Sentence completion is an automatic proxy and is separate from EOS completion.

The explicit `phase51-metrics-v1` adapter requires the producer's exact keys. Unknown/missing fields, old aliases, wrong types, NaN/Inf, invalid ranges, mismatched sample counts, and zero ratio denominators fail closed. The wrapper consumes canonical sampling outputs only. Twenty registered threshold families expand to 34 concrete comparisons, each with producer field, consumer field, unit, comparison and failure behavior. All margins are unchanged. Future comparisons use explicit decimal arithmetic for inclusive boundaries; no tolerance is applied and old gates are never changed.

Actual unchanged-parent CUDA FP32 inference on two preregistered non-secret prompts passed runner → adapter → schema → wrapper → Gate. Other quality values in this tiny plumbing test are explicitly synthetic, so it makes no model-quality claim. Renaming naturalness causes the preflight to fail. Contract gate: `EVALUATOR_CONTRACT_READY`.

Contract SHA256: `6576428940779e2908a8aacc67c4fb6bd2ca8bdc945e6202b88d5116ec701c6b`.

## Additional PHASE57 implementation findings

The original naturalness/semantic alias mismatch is reproduced. The old wrapper also preferred greedy metrics when a requested field existed there, despite labeling topic retention/Japanese validity comparisons as sampling. A binary-float subtraction produced `0.010000000000000009` at the 0.01 boundary. Normal-control math/code/list regular expressions were double-escaped; the old descriptive shape count is not reliable capability evidence. The absolute-update auxiliary cache cursor selected episodes32–46; the registration did not explicitly state its initial cursor. These are append-only PHASE58 findings, not changes to old artifacts or decisions.

## Control drift — diagnostic only

The original Core token membership, sorted positions and packed512 block geometry match the frozen v39 calculation. Stored occurrence-level CE values reproduce both stored micro and macro values exactly; no inference or corrected efficacy gate was run for this audit.

Control minus parent Core micro CE is **+0.10640996**; macro-per-token CE is **+0.12327609**. Both exceed the unchanged +0.1 margin. Paired document bootstrap (10,000, seed5701), diagnostic only:

- micro 95% interval: **[+0.05227660, +0.16010384]**;
- macro 95% interval: **[+0.09008991, +0.13703810]**. Macro resamples whole documents, then averages token types represented in each draw.

CE increases at 62.53% of Core occurrences and 73.31% of Core token types. Top20 tokens account for 40.16% of positive token contribution mass; top5 documents account for 34.04% of positive document contribution mass. The descriptive concentration rule classifies this as **CONTROL_DRIFT_BROAD within the fixed Core population**. This does not establish broad language degradation or a causal mechanism. Token/document IDs, counts, frequency ranks, micro/macro contributions and support are recorded without body dumps.

Parent→Control parameter-update norm is **2.09872302**, 1.2917% of parent norm. Embeddings, attention, FFN, LayerNorm and each block are decomposed in the audit. Tied embedding/LM-head parameters are counted once.

All122 Control gradient norms exceeded clip1.0 (mean4.02854, max5.39858), none exceeded10. LM mean loss was4.52195; first16 mean4.61066 versus last16 mean4.46049. This is an optimization signature, not evidence that clipping caused drift. Model/optimizer/scheduler/permutation/RNG initial fingerprints match the parent; Control and A final RNG/permutation match each other.

The actual122 blocks contain62,464 targets; EOS density0.03362% versus whole-train0.02997%, adjacent repetition1.14369%, and bottom20%-frequency tokens1.32716%. Core exposure is393 occurrences across160 of236 types. Exposure versus per-token CE delta correlation is −0.3270, descriptive only. Reliable semantic category metadata was unavailable, so no category claim is made.

Classification: `CONTROL_DRIFT_BROAD`, `CONTROL_DRIFT_SEED_LOCAL_POSSIBLE`, `CONTROL_DRIFT_CAUSE_UNRESOLVED`. Only seed42 has this continuation; seed-local versus systematic cannot yet be determined. No optimizer or data-order causal experiment was performed.

## Intervention dose — diagnostic only

There were15 auxiliary slots and15 unique cache episodes (indices32–46);104 negative events/masked positions were used,6.9333 per auxiliary update. Seven slots had an empty mask (46.67%). Among107 eligible nonspecial cycle candidates,3 were vetoed (2.8037%); this denominator excludes EOS/short episodes and is not a new training mask.

Raw auxiliary loss averaged0.28157575 on auxiliary updates,0.03461997 across all122 updates. Coefficient remained0.05. Weighted contribution averaged0.01407879 on auxiliary updates and0.00173100 overall. Weighted/LM ratio averaged0.33611% on auxiliary updates; total weighted auxiliary divided by total LM loss was **0.0382789%**.

Separate Arm A LM gradients, auxiliary gradient norms and their cosine were not logged: **NOT_AVAILABLE**. They were not recreated with training. Parent-relative Control and Arm A update vectors have cosine0.99776245; their parameter difference norm is0.14037898. Thus an intervention-associated weight difference is present; loss size alone does not establish gradient dose or efficacy.

| Descriptive observation | Parent | Control64k | Arm A64k |
| --- | ---: | ---: | ---: |
| Greedy runaway | 100% | 100% | 100% |
| Sampling runaway, three RNG mean | 98.333% | 99.667% | 99.333% |
| Greedy loop onset | 29.23 | 28.28 | 26.40 |
| Greedy repetition1 | .91641 | .90938 | .91000 |
| Terminal EOS probability | .01056 | .01427 | .01587 |
| Sampling naturalness proxy | .68000 | .72333 | .70667 |
| Sampling semantic proxy | .61000 | .66000 | .63333 |

These directly stored observations are not registered valid decision metrics. The apparent0.33 percentage-point sampling difference is not an efficacy claim. Dose classifications: `INTERVENTION_CONFOUNDED_BY_CONTROL_DRIFT`, `INSUFFICIENT_DOSE_EVIDENCE`. No coefficient was tuned.

## PHASE59 registration

Created `evaluation/phase58/phase59-control-stability-preregistration.json` after all PHASE58 prerequisite checks. It specifies Control only, seeds123/2026, approved LR5e-5, EOS1.5, CUDA FP32,122 updates per seed,62,464 LM tokens and1,440 conservative matching-forward positions (63,904 accounted). No intervention gradients, budget extension or seed42 confirmation rerun is allowed.

Parent SHAs:

- seed123: `78df96b70016dda180f4529833d904e2d448e11469a87278d4881ed09c93dc20`;
- seed2026: `7dcf5fa2da58777040cf9c03f1f513d707e6866f2eaf237bf203bd1b0135f069`.

Both parent checkpoints passed strict model/optimizer/scheduler/sampler/RNG checks, including RNG restore roundtrip. Per-seed next-permutation hashes are frozen. Matching no-grad eval forwards retain the historical fixed cache/cursor; the dry run showed no RNG/state mutation, and PHASE57 endpoints retained identical RNG streams. Future execution must assert this around every matching forward.

All34 quality comparisons are frozen. Both seeds must pass all safeguards for `CONTROL_STABLE_MULTISEED`. Both must fail at least one primary Core mean margin for `CONTROL_DRIFT_REPLICATED`. Other valid combinations yield `CONTROL_STABILITY_MIXED`; incomplete or invalid evaluation yields `EXPERIMENT_INVALID`. Neither a positive nor a negative result revisits the PHASE55 LR selection question.

PHASE59 training is **NOT_AUTHORIZED / NOT_RUN**; it requires a new explicit user request, a new runner, and runtime integrity plus contract dry-run checks before any optimizer step.

## Safety and validation

PHASE57 artifacts/checkpoints and existing dirty files are frozen by SHA in the PHASE58 preflight. Protected4/READY5 and sealed fingerprints are rechecked. Final Blind is hash-only; Reserve2 is sealed/unscored; PHASE53 Reserve was not opened. Checkpoint COPY/MOVE/DELETE/OVERWRITE are all0. Raw contributions/dose/parameter diagnostics and test XML are on Z: outside Git. Web/product code is unchanged.

Contract/objective/forensic tests and full pytest results are recorded in `evaluation/phase58/final-qa.json`. Only PHASE58 additions are eligible for the separated research commits. Approved LR remains5e-5; Generation Policy remainsUNSAFE; canonical promotion,20M permission, Foundation Base and production deployment remainNO.
