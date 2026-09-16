# Foundation v4.4 — one-shot confirmatory report

Formal LR Gate: **FORMAL_LR_APPROVED_5E5**. Approved LR: 5e-5 for future research use only; no training or model promotion.

## Custody and holdout

PHASE53 Reserve27 permanently RETIRED_UNSCORABLE, unopened/unscored; near relation to supplemental remains UNKNOWN. These sets must never be claimed independent. No retired Reserve use for validation, selection, thresholding, hyperparameters or quality assessment.
PHASE54 supplemental591 /1,078,706 tokens reused, acquisition window unchanged. All historical comparison source hashes verified. Internal174,345 unordered pairs: exact0, normalized0, near0 at full normalized5-character Jaccard>=.8. Historical reuse retains the narrower SimHash<=3 AND Jaccard>=.8 rule and its lexical/coverage limitations.
**FRESH_HOLDOUT_V2_READY**. SHA256, normalized SHA and SHA256-64 bottom256 KMV fingerprints computed before split. Future sketch comparison is approximate (review/exclude at estimatedJaccard>=.75); it cannot certify absence of paraphrases. No raw or normalized text is in the fingerprints.

| Split | Documents | Tokens | State |
|---|---:|---:|---|
| diagnostic | 299 | 556546 | pipeline-only; not primary LR evidence |
| confirmatory | 172 | 296912 | consumed for model selection |
| future-reserve2 | 120 | 225248 | SEALED_WITH_FINGERPRINTS / no inference |

Category-stratified SHA ordering, seed phase55-v2-5501, memberships frozen before scoring. Categories<6 assigned entirely to Diagnostic (math3, language5, procedure1); no claim of broad subject coverage. Reserve2 contents were not reopened after sealing. Source metadata/license references and per-split distributions are in the manifest/fingerprint artifacts.

## Preregistered comparison

Original frozen SHA: `d9e93c9854c2a44abd4dc4a1c6cd76022a79f40a036c3add345c4add36fe16c2`.
Original local preregistration includes tokenized fresh prefixes and is intentionally NOT staged. The public derivative retains all metric/threshold/RNG rules and prompt IDs, with this original digest; no rule was amended after observing results.
Six existing512k checkpoints C/B×seeds42/123/2026: SHA and12 integrity checks each PASS. CUDA FP32, TF32 OFF, no gradients/optimizer updates. Both LRs retain EOS1.5 and auxiliaryOFF. Per-checkpoint complete raw inference saved on Z, not Git. Thermal maximum82°C across confirmatory runs; no hardware throttle; monitored cooldown pauses used.
Equal-document CE, all tokens including EOS target excluding BOS. Fixed128-target chunks with up to384-token overlap for max512 context; max128 short blocks score identical targets. These are bounded-context curves, not exactly512 history tokens at document starts. Seed-average paired document C−B, percentile95% bootstrap,10,000 replicates, RNG550055. Seeds fixed; CIs do not imply generalization over all training seeds.

| Metric (equal3-seed mean) | C /5e-5 | B /7.5e-5 |
|---|---:|---:|
| CE full | 4.386036 | 4.416194 |
| Perplexity exp(mean CE) | 80.321385 | 82.780636 |
| CE short | 4.470720 | 4.500896 |
| Top1 rate | 0.248022 | 0.243531 |
| Top5 rate | 0.438345 | 0.433769 |
| Top10 rate | 0.524431 | 0.520385 |
| Terminal P(EOS) | 0.009942 | 0.009582 |
| Nonterminal P(EOS) | 0.000446 | 0.000437 |
| Premature EOS argmax rate | 0.000000 | 0.000000 |
| runaway_rate | 0.989583 | 0.989583 |
| eos_rate | 0.010417 | 0.010417 |
| sentence_completion_rate | 0.083333 | 0.057292 |
| japanese_validity | 0.776042 | 0.778646 |
| topic_retention_proxy | 0.209377 | 0.209838 |
| naturalness_rate | 0.723958 | 0.710938 |
| semantic_rate | 0.664062 | 0.627604 |
| repetition_1 | 0.323304 | 0.336589 |
| repetition_2 | 0.113266 | 0.124959 |
| repetition_3 | 0.046449 | 0.057082 |
| repetition_4 | 0.017489 | 0.024804 |

Primary paired CE: -0.03015828;95% CI [-0.03113266,-0.02922173]. All3 seed-specific CIs also below0 (JSON).

| Arm /seed | EOS probability | Mean rank | Top1 | Top5 | Top10 |
|---|---:|---:|---:|---:|---:|
| C/42 | 0.010058 | 26.13 | 0.0000 | 0.4709 | 0.6802 |
| C/123 | 0.011151 | 19.23 | 0.0000 | 0.4767 | 0.6628 |
| C/2026 | 0.008618 | 23.79 | 0.0000 | 0.5058 | 0.6977 |
| B/42 | 0.010345 | 25.82 | 0.0000 | 0.4360 | 0.6802 |
| B/123 | 0.010241 | 22.21 | 0.0000 | 0.4244 | 0.6105 |
| B/2026 | 0.008162 | 20.68 | 0.0000 | 0.5116 | 0.7151 |

## Relative safety, not generation safety

Registered clear-material-regression rule requires the entire paired95% interval to exceed a worsening boundary: context CE +.03; loss of long-context benefit+.03; terminal EOS relative drop20%; premature EOS+1pp; sampling runaway+8pp or naturalness/semantic/Japanese-validity drop8pp. All8 flags are false. Full uncertainty and per-seed distributions remain in JSON; absence of a flag is not a proof of no harm.
64 checkpoint-independent fresh prompts ×2 fixed RNG ×3 seeds per LR. Sampling decoder temperature.7, max64, no forced-stop credit. Both LR runaway98.9583%, EOS completion1.0417%: **GENERATION_POLICY_UNSAFE** remains. Naturalness/semantic/topic/completion are automatic proxies, not human correctness or actual relevance validation.
Confirmatory consumed_for_model_selection=true from scoring start and confirmed complete here; never reuse as blind. Diagnostic is not a replacement primary set. PHASE53 Reserve and Reserve2 remain unavailable to model scoring.

## Outcome

Approved5e-5 research LR does NOT authorize training start, Foundation Base completion, generation safety, canonical,20M or Production deployment. New training NO, canonical NO,20M NO, Foundation Base NO.
Attractor audit: INSUFFICIENT_EVIDENCE / MORE_ROOT_CAUSE_WORK_REQUIRED. No PHASE56 arms registered; see foundation-v44-attractor-root-cause.md.
