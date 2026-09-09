# PHASE50 / Foundation v3.9 — Frequency Gate v3

Formal LR Gate: **FORMAL_LR_STILL_UNRESOLVED**. Approved LR: NO. Selected LR: NONE.

Preflight: 443 passed, 0 failed. Final pytest: 480 passed, 0 failed (5 warnings); web framing contract separately PASS.

## Checkpoint-independent preregistration

Train rank >=3277 (bottom 20% of unchanged 4096 vocabulary), validation occurrences 2–9 inclusive, at least two independent validation documents per token. Select every qualifying ID and occurrence, excluding position zero. Core and Supported Tail IDs are disjoint. Model loss/probability/top-k were not used for membership.
Preregistered on 2026-09-07T12:57:31.296812+00:00. Definition file SHA256: 7d4a890faf0b4d366ad147eb01fc7c2f4cfb7a4e610b305f616471e720cd2c85.

| Population | Types | Occurrences | Documents | Role |
|---|---:|---:|---:|---|
| core | 236 | 4436 | 146 | formal frequency evidence |
| supported_tail | 290 | 1528 | 141 | formal frequency evidence |
| legacy | 58 | 112 | 2 | WARNING ONLY |

Supported membership SHA256: a73d20341ee27b2533693b126a888c0903b56a5bc57d601d7e95833ac2d23cf4.
Legacy SHA256: dba4f22a4cc8d9ce52287228e4fbd5759d584daff7579abf9bd310e0144f68ca. Its 58 / 112 / 2 membership is unchanged and cannot veto LR approval.
Maximum observed non-Core rare support before the repeated-support filter: {'tokens': 379, 'occurrences': 1688, 'documents': 141}.
Targets of >=30 documents and >=300 occurrences are met without adding any non-rare ID. Classification: DOCUMENT_SUPPORTED_EVALUATOR_VALID.

## Existing 512k checkpoints, three seeds; no new training

| LR | Validation ± SD | Top1 / 5 / 10 | Middle CE | Core micro / macro CE | Supported micro / macro CE | Legacy CE | Natural / Semantic |
|---|---|---|---:|---|---|---:|---|
| 5e-05 | 4.347283 ± 0.008019 | 27.29% / 45.13% / 53.08% | 6.461308 | 9.250460 / 9.269114 | 9.784955 / 10.010819 | 9.568183 | 69.33% / 60.33% |
| 7.5e-05 | 4.374239 ± 0.006253 | 26.97% / 44.69% / 52.70% | 6.490935 | 9.272481 / 9.285930 | 9.800815 / 10.024438 | 9.560458 | 70.00% / 63.00% |

Paired C minus B, seed-averaged before document resampling:
- Core CE 95% CI: [-0.03165040356034702, -0.011250247390852192].
- Supported Tail CE 95% CI: [-0.024783321496225606, -0.006986016969177991].

2000 document-cluster replicates (RNG 4900). Per-seed micro CE, macro token CE, Top1/5/10, mean/median/geometric correct probability and intervals are in `phase50/frequency-statistics.json`; seed mean/std are in the summary. Head combines old rank 0–819 sub-buckets with occurrence weighting; Middle uses ranks 820–3276. Core and legacy results reuse exact PHASE49 SHA-matched checkpoints. Supported Tail uses CUDA FP32 inference on fixed packed 512 contexts.

## Why no formal approval

Supported Tail passes every per-seed noninferiority margin (paired CE CI upper <=0.25). Core keeps its pre-existing <=0.10 margin and nonfrequency checks remain unchanged. Legacy has no vote. Improving evaluator support does not erase independent EOS, Core or sampling failures.

| LR | Seed | Supported delta CE [95% CI] | Remaining failed checks |
|---|---|---|---|
| 5e-05 | 42 | -0.272223 [-0.3178031334691244, -0.22562087261275285] | none |
| 5e-05 | 123 | -0.224149 [-0.27847276681641475, -0.17489474659271376] | sampling |
| 5e-05 | 2026 | +0.024042 [-0.03112002902682265, 0.07903194833619573] | eos, core_ci |
| 7.5e-05 | 42 | -0.256767 [-0.3079652689736102, -0.20285317952306706] | none |
| 7.5e-05 | 123 | -0.221599 [-0.2870324090011461, -0.16122577731035617] | middle, sampling |
| 7.5e-05 | 2026 | +0.053615 [-0.007481255172032833, 0.11737143809267374] | eos, sampling, core_ci |

B has +0.67pp naturalness and +2.67pp semantic mean, but C improves LM, top-k, Middle, Core and supported rare CE. Neither qualitative preference nor minimum validation loss can override the failed per-seed safety checks.

## Context, EOS, attractor and integrity

| LR | Full context CE / advantage | Terminal P(EOS) / premature | Greedy runaway / loop onset |
|---|---|---|---|
| 5e-05 | 4.711762 / 1.184105 | 0.0100058 / 0.00% | 100% / 18.17 |
| 7.5e-05 | 4.743163 / 1.194590 | 0.0097378 / 0.00% | 100% / 15.67 |

Context regression: NO on all existing checks. Premature EOS: 0%, but seed 2026 terminal EOS fails the retained relative safety threshold. Greedy runaway remains 100%, a major separate research issue.
Inference maximum GPU temperature: 79.0°C. Each evaluation records cooldown/start telemetry. GPU training tok/s: N/A (no training). No overclock, undervolt, fan or power-limit changes. CPU parallel evaluation DISABLED.
Checkpoint SHA256, strict reload and resume-state integrity: 19/19 PASS after COPY to the configured Z root. Original D files retained. Protected 4 files and relocated READY 5: unchanged. Other pre-existing infrastructure edits were retained separately; the historical 18-file preflight is not a claim that those infra files remain unchanged. Official 15.872M and old LR1e-4 16.128M unchanged. Final Blind was hashed only: fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b.
Canonical 16.128M training: NO. Candidate metrics: N/A. No lineage promotion. 20M permission: NO. Foundation Base complete: NO.

## Limitations and next gate

All intervals are conditional on the fixed validation corpus, not evidence for all-language generalization.
Head/Middle, Core and legacy exact PHASE49 results reused with identical checkpoint SHA; only new supported-tail inference was run.
Legacy Tail is warning only. Existing per-seed Core/EOS/sampling margins are not loosened after seeing outcomes.
No new GPU training, no candidate, no canonical promotion. Both 512k arms remain experimental.

Next: preregister a focused EOS/Core/sampling investigation before requesting any new training. No training permission is implied by this report.
