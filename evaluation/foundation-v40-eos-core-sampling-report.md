# PHASE51 — EOS / Core / Sampling diagnostic

LR Recommendation Gate: **MEASUREMENT_UNCERTAINTY_REMAINS**. Recommended LR: none. Comparative LM preference: 5e-5, not a safety approval.

Formal training permission: NO. New GPU training: NO. Next canonical target: N/A until PHASE52. 20M: NO. Foundation Base: NO.

Preregistration fixed before inference. CUDA FP32, no concurrent heavy CPU evaluation. Existing PHASE50 metrics/gates reused, no threshold or population changes. Nine checkpoints passed strict reload/metadata and before/after SHA checks. No checkpoint COPY/MOVE/DELETE/overwrite.

## EOS

All146 independent document ends measured once, plus500 distinct nonterminal positions excluding EOS/BOS. Historical500 EOS observations repeated146 documents; they were not500 independent ends. Historical gate uses own-LR256k control, not the formal baseline.

| Arm | Seed | Baseline P(EOS) | Candidate P(EOS) | Delta 95% CI | Old EOS gate |
|---|---|---|---|---|---|
| C | 42 | 0.013882 | 0.010558 | [-0.004676, -0.002106] | True |
| C | 123 | 0.008107 | 0.011482 | [0.002275, 0.004644] | True |
| C | 2026 | 0.019147 | 0.008187 | [-0.013825, -0.008601] | False |
| B | 42 | 0.013882 | 0.010606 | [-0.004647, -0.002016] | True |
| B | 123 | 0.008107 | 0.011249 | [0.001872, 0.004621] | True |
| B | 2026 | 0.019147 | 0.007681 | [-0.014421, -0.009015] | False |

EOS classifications: {'C': 'SEED_LOCAL_EOS_VARIANCE', 'B': 'SEED_LOCAL_EOS_VARIANCE'}. Per-document ranks, top-k, competitors, margins, lengths and entropy are in the EOS artifact.

## Core seed2026

Core236 types /4436 occurrences /146 documents unchanged;10000 paired document resamples with RNG4900,5100,2026. LOO is influence diagnosis only, not a replacement gate.
- C: TOKEN_CONCENTRATED; CE delta 0.062993; CI upper range [0.17312611315452311, 0.17357143014178222]; top5 documents positive contribution 49.3%; top20 tokens 50.8%; LOO upper range [0.07689261636879152, 0.18672007268744403]. 1/146 omissions would cross .10, without authorizing any omission.
- B: TOKEN_CONCENTRATED; CE delta 0.102466; CI upper range [0.2358501925529789, 0.23820243899815707]; top5 documents positive contribution 49.0%; top20 tokens 51.2%; LOO upper range [0.1188147757813057, 0.2522998369173304]. 0/146 omissions would cross .10, without authorizing any omission.

## Sampling and evaluator audit

Same100 fixed prompts ×3 CUDA sampling RNG bases per checkpoint; temperature.7, no top-k/top-p filtering, max64 unchanged. Base44000 repeated exactly for all9 checkpoints. All2700 generated rows and paired prompt distributions are retained. Naturalness/semantic/completion are deterministic automatic proxies, not human quality. Semantic proxy checks surface structure, not prompt relevance.

Historical generation used CPU (`evaluation_execution.device=cpu`); numeric seeds are not equivalent CPU/CUDA random streams. New comparisons are CUDA versus CUDA. Differences from the historical single-seed aggregate cannot clear its gate.

- C classification: PROMPT_LOCAL_REGRESSION.
- B classification: PROMPT_LOCAL_REGRESSION.

This preregistered label describes concentrated negative prompt contributions, not a statistically established aggregate regression. For seed123 C, naturalness is73%/65%/64% across the three CUDA RNG bases (mean67.33%); semantic proxy64%/56%/56% (mean58.67%). Paired baseline deltas are+1.00pp (95% CI[-5.33,+7.33]) and+4.00pp ([-2.00,+10.00]). The top10 prompts account for56.7%/51.9% of negative contributions. Both prompt-local effects and RNG variation are present; no human-quality conclusion follows.

For seed123 B, naturalness71%/65%/69% (mean68.33%); semantic proxy61%/57%/58% (mean58.67%). Paired baseline deltas+2.00pp ([-5.67,+9.33]) and+4.00pp ([-3.33,+11.33]). Replicated own-LR256k controls were not measured, so the old relative gate remains unresolved.

EOS's SEED_LOCAL label likewise does not dismiss harm: seed2026 P(EOS) fell substantially, with a negative paired CI for both candidates; seed42 also declined while seed123 improved. The Core concentration labels are close to the preregistered50% cutoff (not a robust causal boundary); bootstrap CI upper estimates are stable and still exceed the unchanged.10 threshold.

## Attractor and LR

C versus B attractor: MIXED; C-minus-B mean loop onset -0.413, repetition1 -0.001458. Greedy runaway remains100% in all arms. Entropy/probability margin here average all steps; old summary measured loop-onset steps.

5e-5 leads 7.5e-5 in loss, top-k, Middle, Core, Supported Tail and full-context CE. Existing failures remain: C123 Sampling; C2026 EOS/Core; B123 Middle/Sampling; B2026 EOS/Sampling/Core. Diagnostics do not establish a safe canonical LR. No automatic promotion.

## Artifacts and reproducibility

See phase51/preregistration.json, resolver-gate.json and the four detailed diagnostic JSONs. Raw full-step traces remain local under phase51/raw; SHA256 inventory is in the summary. Compact per-prompt/per-document evidence is versioned. FinalBlind content was not opened (hash only). Preflight pytest480 PASS; final QA is recorded separately after ML/Web completion.
