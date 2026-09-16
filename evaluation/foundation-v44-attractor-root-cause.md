# Foundation v4.4 — attractor root-cause audit

**INSUFFICIENT_EVIDENCE** for a dominant cause. **MORE_ROOT_CAUSE_WORK_REQUIRED**. Research LR5e-5 is approved separately. No training, canonical or20M; no executable PHASE56 arm preregistration was created.

## Integrated observations

Reused PHASE51 fixed900 greedy traces from baseline/C/B×3seeds and PHASE52 source-hash-verified time series. PHASE53 policy study remains GENERATION_POLICY_UNSAFE; decoder mitigation did not demonstrate safe natural termination. New PHASE55 normal controls are separate from fresh LR selection.
Analyzed up to32 steps strictly before and after loop onset, retaining prompt taxonomy, prefix length, cycle length, repetition/unique-token ratio, entropy, top1/top2 margin and EOS. Early onsets have only14.83–19.69 pre-onset steps on average by checkpoint; post windows31.70–32.00. No invented missing steps. Source series contain cumulative repetition1–4; detailed repeated-ngrams retained in the hash-bound PHASE52 raw evidence.
Across all9 checkpoints after-onset top1 increases and unique-token ratio falls; EOS decreases. For C2026: top1 .28091→.33632, entropy4.78123→4.45774, margin.21972→.26912, EOS.000612→.000378. This supports a descriptive self-copy attractor with probability concentration and weak stopping, but does not identify which factor causes another.
Full logit-vector/hidden-state cosine: NOT_MEASURED. Historical records retain top5 probabilities, not full vectors; reconstructing cosine would be invalid. No architecture hooks or model changes introduced.

## Copy signature versus training

Top20 generated post-onset ngrams per width, chosen by occurrence count then token-ID tie break. All33,402,759 packed training tokens examined, cross-document hits excluded. All10,012 JSONL documents were retokenized and matched exactly to packed boundaries before joining category labels. Per-pattern frequency, document support and category support are in phase55/attractor-root-cause.json. No source prose is quoted.

| n | Selected | Zero training frequency | Max training frequency |
|---|---:|---:|---:|
| 3 | 20 | 2 | 276052 |
| 4 | 20 | 11 | 209 |
| 5 | 20 | 16 | 36 |
| 6 | 20 | 17 | 34 |

Many frequent longer generated patterns are absent from training, weakening a simple direct-copy explanation. It does NOT rule out learned data-induced repetition. Frequent3grams may reflect ordinary grammar. No calibration, randomized position control or causal objective ablation distinguishes overconfidence, EOS suppression, self-copy and exposure effects.

## Normal repetition control risk

Reused all20 frozen PHASE54 templates: math, lists, definitions, code and repeated terminology ×n2/3/4/5. Concrete prompts and rules frozen before new greedy inference; baseline42/C42/B42. All60 outputs exhausted128 tokens without EOS; observable shape proxy0/20 for each model. This is not a human semantic score. The pretraining model cannot currently satisfy the finite instructional templates, so this control set has a floor effect: unchanged0 does not establish non-regression. Independent boundary review remains pending.

## Candidate comparison (hypotheses, not measured benefits)

| Candidate | Expected benefit | Failure risk | Normal-repeat risk | Complexity | Reversibility |
|---|---|---|---|---|---|
| Approved LR continuation control | Isolate objective change at5e-5 | Shared loops likely remain | Existing failure floor | Low | New experimental checkpoint only |
| Generated-prefix ngram unlikelihood | Target repeated self-generated contexts absent in teacher forcing | Bad negative mining; probability mass diversion | High: math/list/code repetition | Medium; exact negative labels needed | Disable loss, retain parent checkpoint |
| Sequence-level anti-loop | Penalize pathological completed trajectories | High-variance/sparse signal; force-shortening shortcut | High: legitimate repeated procedures | High; trajectory scoring and masks | Separate arm, no canonical change |
| Data repeated-pattern reweighting | Reduce objectively excessive source patterns if implicated | Remove useful language; longer generated grams often absent | High: textbook/code/definitions | Medium; corpus provenance audit | New weights only, original corpus immutable |
| EOS-aware auxiliary | Improve stopping where endings are justified | Premature termination, apparent runaway improvement without quality | Medium/high on long explanations | Medium; terminal/nonterminal supervision | Disable auxiliary, preserve baseline |

Unlikelihood is a published objective family with token/sequence variants; its reported success elsewhere does not prove suitability here. [Welleck et al., Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319). The benefit/risk judgments above are hypotheses from this audit, not replicated paper results.
The exact failed PHASE42 teacher-forced repeated3/4gram alternative-token auxiliary (weights.01/.03/.05, runaway100%) is explicitly excluded. A differently supervised generated-prefix proposal is not yet registered or authorized.

## Gate and next evidence required

No PHASE56 arms: LR condition passes, sufficient-cause/safe-control condition does not. Do not create foundation-v44-training-fix-preregistration.json. Next bounded non-training research should separate EOS/position/self-prefix effects on fixed historical prompts and establish a normal-repeat control with measurable nonfloor correctness; never reuse fresh Confirmatory as blind or open either Reserve.
If a later phase establishes both conditions, a future design should use control+max2 interventions, initial64–128k/max256k, greedy runaway<=50% minimum (<=25% desired), substantial sampling decline, and preregistered LM/Core/SupportedTail/EOS/context/normal-control non-degradation. These are planning constraints, not executable arms or permission to start.
