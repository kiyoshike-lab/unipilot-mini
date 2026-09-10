# PHASE52 — Attractor mechanism audit

Root-cause classification: **MIXED_OR_UNKNOWN**. Attractor Gate: **ATTRACTOR_CAUSE_STILL_UNKNOWN**. Greedy runaway900/900 (100%) remains a major Foundation Base blocker. No causal repair claim or training permission.

## Scope and reproducibility

Reused PHASE51 CUDA FP32 traces: baseline15.872M, C5e-5 and B7.5e-5 at16.384M, each seeds42/123/2026 and the same100 prompts. Raw artifact and checkpoint SHA256 values were verified before use. No fresh model inference, decoder grid search or training was run. An observational analysis plan fixed family mapping/window/counting before producing these new summaries.

Existing metadata taxonomy and prefix lengths are retained. Labels such as “factual” or “procedural” were not invented from outputs. Prefix length and content are confounded, so these groups are not a controlled position experiment.

Full local time series save token ID, repeated n-grams1–4, entropy, Top1/Top2 probabilities, margin, EOS probability and unique-token ratio at each generated step. Each example has a detected onset/cycle and up to32 preceding generated tokens. Early onsets have fewer than32 observed steps; no prefix logits were fabricated. The compact versioned summary includes per-prompt early-window/onset endpoints and all nine aggregates. Full windows/128-step series are in `phase52/raw/attractor-time-series.json` (SHA in summary), deliberately not staged.

## Observations, not causal proof

Relative logit dynamics are also retained for every available pre-onset window and onset: `log(P(top1)/P(top2))` and `log(P(top1)/P(EOS))` recover the corresponding logit differences from the saved softmax probabilities (within their recorded numerical precision). Absolute logits cannot be recovered. Zero-probability cases are null, not fabricated finite gaps. The separate local `phase52/raw/pre-onset-logit-dynamics.json` has its SHA and nine onset aggregates in `relative_logit_dynamics` in the summary; no new inference was needed.

| Candidate mechanism | Evidence / limitation |
|---|---|
| LOCAL_NGRAM_ATTRACTOR | Repeated exact cycles observed in all900 examples. This describes the behavior, not the training cause. |
| EOS_SUPPRESSION | Mean EOS probability at onset is low: C2026 .000585, B2026 .000960, baseline2026 .000293. Low probability alone does not isolate the cause; values vary by seed. |
| OVERCONFIDENT_TOP1 | C2026 onset Top1 .3332 / Top2 .0506, margin .2826, entropy4.636; B2026 .3218/.0413, margin .2804, entropy4.694. Baseline2026 .2392/.0537, margin .1854, entropy5.183. Increased concentration is associated with loops but is not a calibrated causal-overconfidence finding. |
| POSITION_CONTEXT_EFFECT | Different prefix lengths/families available, but no randomized content-matched position intervention. Cause unresolved. |
| DATA_REPETITION_SIGNATURE | Top20 generated cycles counted exactly in the frozen train corpus. Token205 alone occurs854789 times and appears as114 generated loop cycles; cycle[205,32] occurs16737 times in train and15 generated examples. Frequency is not evidence that repeated generation was directly memorized. |

N-gram counts include occurrence rates per million eligible train windows. No large source passages are displayed. The existing Foundation v1.1 source/cleaning manifest and Wikimedia/Wikibooks licensing/provenance records are preserved. No corpus redistribution or license change occurs here.

## Decoder mitigation and next decision

Existing temperature.7 multi-RNG sampling metrics are included for comparison with greedy. They are historical diagnostic evidence, not a newly preregistered mitigation trial. No new top-k/top-p/repetition-penalty/hard-stop experiment was performed; those were optional. No decoder was modified, no Production setting changed, and no claim is made that the model defect is repaired or Beta safety established.

A future authorized experiment would need controlled EOS/position/decoding interventions, fixed outcomes and held-out quality validation. Current evidence cannot distinguish a single cause. Formal LR remains unresolved; candidate preference5e-5 is not canonical approval. New training:NO; canonical:NO;20M:NO; Foundation Base:NO.
