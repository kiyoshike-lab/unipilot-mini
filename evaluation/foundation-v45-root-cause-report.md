# Foundation v4.5 — Generation Collapse Observatory

PHASE56. Root Cause Gate: **MIXED_CAUSE_WITH_ACTIONABLE_TARGET**. Training Readiness Gate: **PHASE57_TRAINING_READY**, restricted to the registered seed42 mechanistic screen after a future runtime preflight. No training was run. A single dominant cause has **not** been established; the actionable target is generated-prefix self-feedback, not a proven architectural defect.

## Scope and custody

- Approved research LR stays5e-5. No new7.5e-5 comparison or LR selection; PHASE55 one-shot Confirmatory172 is consumed and was not rerun.
- Z resolver PASS; explicit `UNIPILOT_CHECKPOINT_ROOT=Z:\AI\unipilot-mini\checkpoints`. Four checkpoints SHA/strict model and optimizer/resume-state checks passed; same pre/post checkpoint hashes, exact hooked/unhooked logits and unchanged model tensors. COPY/MOVE/DELETE/OVERWRITE all0.
- Primary C16.384M processed-token checkpoints, seeds42/123/2026; model has19,514,880 parameters (19.5M denotes parameter count, not processed tokens). Minimal15.872M seed42 reference uses only the first8 selected documents.
- PHASE55 Diagnostic299 is marked consumed_for_diagnostics=true; the rule selected24 IDs before outcomes and scored these24, not all299. Diagnostic content was read for this bounded study, never for gradients or formal LR selection. This is not fresh confirmatory evidence.
- PHASE53 Reserve stays RETIRED_UNSCORABLE; PHASE55 Reserve2 stays SEALED_WITH_FINGERPRINTS / UNSCORED; both hash-only. Final Blind SHA-only remains `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b`.
- Observability preregistration SHA `6ead2d2d1a6385e43752b6b2a980a37c23531c8629cba555293d928a289199e9` precedes all observations. Its original rules and source hashes were not edited after results.

## Collected evidence

622 CUDA FP32 trajectories: primary216 free/teacher/short-context runs,306 one-token perturbations,24 legacy generation controls,60 normal teacher-forced controls and16 baseline free/teacher runs. These are **not622 independent samples**. Main inference is a24-document sample repeated across three checkpoints. Hidden-state and attention hooks collected all10 layers without modifying model code; selected full4096 logits/hidden vectors and per-step raw text are Z-only, never Git. Git contains body-free derived metrics and hash manifests.

Each step includes token ID/text, absolute last-input position, selected-token probability, top1/top2/margin, entropy, EOS probability/rank/logit/margin, hidden/logit norms, previous/cycle cosines and attention mass on recent1–4 keys, previous cycle, earlier context, BOS and expected key position. The state at a step predicts the next token: it is not the representation of that newly selected token. Cycle comparison uses the free trajectory's period for both free and teacher controls. Onset windows use up to32 steps strictly before/after the unchanged loop onset, with boundary counts retained.

| Seed | Free runaway | Free−teacher repetition4 | Post-onset layer0 cosine: free / teacher | Post-onset layer9 cosine: free / teacher | Earliest registered cycle-lock layer |
|---|---:|---:|---:|---:|---|
| 42 | 100% | +0.81933 | 0.91386 / 0.12400 | 0.93656 / 0.22633 | NOT_ESTABLISHED |
| 123 | 100% | +0.79300 | 0.84444 / 0.12807 | 0.87279 / 0.23570 | NOT_ESTABLISHED |
| 2026 | 100% | +0.80733 | 0.81613 / 0.13987 | 0.85176 / 0.23828 | NOT_ESTABLISHED |

All10 layers in every seed exceed the registered free-minus-teacher cosine difference0.05. However, none crosses the separate mean absolute cycle-lock cosine0.95 threshold. Thus layer0 already shows regime-sensitive recurrence, but **no causal origin layer or registered lock layer can be claimed**. Seed42 baseline8-doc runaway is8/8, repetition4=.900 versus teacher=.034; this shows the failure predates the selected checkpoint, not a fresh LR comparison.

Perturbation rule was onset−4/−8/−16, top2 allowed token or aligned ground-truth token, with the original prefix replayed until exactly that step.306 runs,303 actually changed a token,300 changed runs retained at least16 suffix tokens. All300/300 eligible suffixes contain a loop by the **unchanged** periodic-span definition. Three short EOS outcomes are excluded from that eligibility denominator but disclosed: all-run EOS completion3/306=.9804%. Thus the assay supports a persistent basin, not proof that every perturbation can never escape or that every suffix repeats the identical original cycle. Perturbations from the same document are correlated; no300-independent-trial confidence claim.

## Confidence and EOS

| Seed | Entropy before → after | Top1 before → after | Margin before → after | P(EOS) before → after | EOS rank before → after |
|---|---:|---:|---:|---:|---:|
| 42 | 4.88366 → 4.52748 | .24442 → .31131 | .16600 → .25010 | .00037058 → .00028769 | 909.98 → 902.00 |
| 123 | 4.49922 → 4.37402 | .27764 → .31255 | .19890 → .24126 | .00033230 → .00025374 | 943.58 → 858.14 |
| 2026 | 4.80086 → 4.69655 | .25540 → .29198 | .17486 → .22386 | .00038419 → .00043611 | 736.91 → 561.53 |

Free after-onset entropy is below paired teacher entropy (4.84435/4.74579/4.74003); top1 margins are above teacher(.19028/.19359/.20188). Confidence sharpens in all three seeds but probabilities around.29–.31 are not near-certain collapse; an independent confidence intervention is still needed to establish causation. EOS rank improves after onset and P(EOS) rises in seed2026, contradicting a universal onset-triggered EOS suppression story. Low EOS on nonterminal Wikipedia continuations is not sufficient evidence of incorrect terminal learning. Full EOS logits/margins and per-document windows are in the derived summary.

Attention previous-cycle mass is distributed, typically peaking around layers3–4 at.14–.16; earlier context retains roughly.45–.91 mass across layers. This is not exclusive local attention fixation. Attention weights are descriptive, not causal attribution. Shorter natural prefixes shift mean repetition4 by−.03433/+ .01967/+ .03833 across seeds: no consistent direction, and content loss is confounded with absolute position. No artificial padding or positional-embedding rewrite was used.

## Training-pattern support

Complete33,402,759-token train corpus and10,012-document category mapping are hash-bound to prior evidence. Matching excludes cross-document patterns. Occurrence counts, independent document support and category support for every observed pattern are saved in the Z raw support table; the Git summary carries its SHA and aggregate rates, not corpus bodies.

| ngram | Unique observed patterns | Absent from train | Novel type rate | Occurrence-weighted novel rate |
|---|---:|---:|---:|---:|
| 3 | 404 | 42 | 10.3960% | 14.5906% |
| 4 | 503 | 204 | 40.5567% | 45.9923% |
| 5 | 587 | 414 | 70.5281% | 74.0433% |
| 6 | 658 | 590 | 89.6657% | 93.4830% |
| 8 | 764 | 754 | 98.6911% | 99.6894% |

These are ngrams in the post-onset region, not an independent sample of memorized semantic facts. Long exact loops mostly unseen in train weaken direct long-phrase copying as the sole cause; frequent shorter pieces and learned distributional patterns can still contribute. The first8 legacy prompts also run away24/24 across seeds, a small descriptive transfer check.

## Hypothesis table

| Hypothesis | Supports | Contradicts / limits | Unknown | Confidence |
|---|---|---|---|---|
| H1 SELF_FEEDBACK_ATTRACTOR | Free–teacher repetition gap.793–.819, large all-layer cycle gap;300/300 eligible perturbed suffixes reloop | Teacher forcing also changes content; some replacements end early | Whether generated-prefix training fixes it; basin identity | Moderate mechanistic target; not dominant proof |
| H2 MODEL_OVERCONFIDENCE | Entropy falls, top1/margin rise in all seeds; centered-logit cycle cosine.953–.982 | Top1 only.29–.31; teacher also confident; no confidence-only intervention | Necessary cause versus consequence of repeated input | Low-to-moderate association |
| H3 EOS_SUPPRESSION | Low loop EOS probability;100% free budget exhaustion | Seed2026 EOS probability rises; EOS rank improves all seeds; windows are not natural document ends | Counterfactual terminal supervision effect | Low as universal dominant cause |
| H4 TRAINING_PATTERN_MEMORIZATION | Most generated3grams have train support; category/doc support available |89.67%6gram types and98.69%8gram types absent | Indirect memorization of short pieces/distributional bias | Low for exact long-pattern dominance |
| H5 POSITIONAL_INTERACTION | Natural shortening changes trajectories | Direction inconsistent; prefix meaning changes too | Pure absolute-position effect | Low / unresolved |
| H6 UNDERTRAINED_BASE_INSTABILITY | Failure in15.872M baseline and16.384M C; weak normal control performance | No training-budget/architecture controlled comparison | More data versus objective versus architecture | Low causal confidence |
| H7 MIXED | Preregistered feedback + persistent basin + novel6 signals all PASS | Signals are related; no unique causal separation | Relative contributions and successful safe remedy | Moderate for actionable mixed target, not established complete explanation |

The root gate follows the frozen conjunction, not a post-hoc dominant label. The next experiment tests a hypothesis; it does not certify that root cause is completely solved.

## Normal controls and registered intervention

20 frozen normal-control concretizations ×3 seeds were teacher-scored. Mean CE4.25157/4.29567/4.28590; terminal P(EOS).007318/.004912/.002855. Prior output-shape floor0/60 is not reinterpreted as safety. Non-degradation is tested with teacher-forced mean and family CE plus terminal EOS; actual output shape remains descriptive.

Only **one intervention A** was designed after the root gate: generated-prefix contiguous-cycle unlikelihood with aligned-reference and reference-snippet veto. Control is the unchanged approved5e-5/EOS1.5 objective. A differs from failed PHASE42's teacher-forced3/4gram auxiliary in conditioning history, four-cycle mask, reference veto and one-in-eight-update dose; the mathematical unlikelihood family is acknowledged, not renamed as a novel loss. Prototype fixture tests show pathological loops receive positive loss with the correct descent direction while required math/code/list/definition/term repetition has zero negative loss.

`phase56/phase57-training-preregistration.json` freezes parent SHA, equations, coefficient.05, seed42, untouched optimizer/scheduler/permutation/RNG continuity, data selection, evaluation contracts and numeric thresholds. First ceiling64k actually charges63,904 gradient-input positions; maximum128k charges127,808, including auxiliary slots.122/244 updates contribute62,464/124,928 main LM tokens respectively. Every arm uses identical accounting, data order and auxiliary forward slots. Training-only cache generation is separately logged inference and must be frozen before either arm, never from any holdout.

Meaningful success requires greedy and sampling runaway<=50% (each fixed sampling RNG also), preferred<=25%, and>=10 percentage-point reduction versus matched control. Quality margins include validation CE+.05, Top1−.01/Top5/10−.02, context CE+.05/long-benefit loss.02, terminal EOS ratio>=.90, Core CE+.10 and SupportedTail CE+.25 including paired upper confidence bounds, normal mean CE+.05/family+.10 and further EOS/proxy safeguards. All compare against both parent and matched control. Any failing or missing safeguard stops the arm; no thresholds are adjusted afterward.

Readiness means the **design and synthetic contract** are ready for a short PHASE57 screen. It is not a universal semantic-safety proof. Future execution still requires runtime preflight, minimum20 training-only negative events, safe output/disk/thermal checks and validated evaluators. A normal repetition absent from the reference could be falsely penalized, so dose and non-degradation gates are essential. No ArmB, multi-seed training, canonical run,20M continuation, Foundation Base completion or deployment is authorized by this report alone.

## Artifacts

- `phase56/observability-preregistration.json`, original freeze and safety record.
- `phase56/generation-observatory-summary.json`: body-free per-document windows and aggregate signals.
- `foundation-v45-root-cause-summary.json`: compact gate/readiness.
- `phase56/run-index.json`:622 raw JSON/vector hash references under Z.
- `phase56/phase57-training-preregistration.json`, `training-design-review.json`, objective prototype and11 passing targeted tests.
- Full local QA and Git preservation: `../docs/PHASE56_HANDOFF.md`.

New training **NO**; Canonical **NO**;20M **NO**; Foundation Base **NO**. Generation Policy remains unsafe pending a successful separate experiment and independent validation.
