# PHASE54 training-side attractor design

Gate: **MORE_ROOT_CAUSE_WORK_REQUIRED**. Formal LR remains unresolved, so PHASE55 experimental arms are **not instantiated**. This is a design and outcome-threshold registration; it is not an executable training recipe or training permission. Architecture, tokenizer, context512 and EOS weight1.5 remain unchanged.

## Evidence integration

PHASE51–52 contain matched historical prompts for baseline/C/B and three seeds. PHASE52 found loops in900/900 greedy outputs. At onset C2026 had entropy4.636, Top1/Top2 .3332/.0506, margin .2826 and EOS probability .000585; B2026 had4.694, .3218/.0413, .2804 and .000960. These are associations, not proof of a single cause. Prefix length and prompt family are confounded. Token205 occurs854,789 times in training and appears as114 generated cycles; high frequency does not prove memorized-loop causation. Sources: `foundation-v41-attractor-summary.json` and its report, derived from PHASE51 traces.

PHASE53's fixed24-prompt comparison reproduced greedy runaway100% for C/B. Median onset was13/12 and repetition1 .857/.834. Sampling lowered token repetition but best runaway was95.83%/97.92%. Forced stop replay lowered cap exhaustion yet every triggered stop lacked EOS. No safe decoder or normal-repeat false-positive estimate exists. The goal is natural termination with retained useful content, not a smaller reported budget-exhaustion rate.

## Candidate review

| Candidate | Rationale and disposition |
|---|---|
| Lower-LR continuation only | Necessary future control only after formal approval; historical C/B both100% runaway gives no evidence that LR alone repairs loops. No LR chosen now. |
| Sequence-level repetition regularization | Candidate concept: operate on model rollouts rather than only teacher-forced prefixes. Needs separately labelled pathological spans, frozen normal-repeat exemptions, gradient-scale audit and LM noninferiority. A repeated sequence alone is not enough to label it pathological. |
| Repeated-ngram unlikelihood | The exact PHASE42 method is excluded. A rollout-conditioned variant would be a different proposal requiring independent controls and loss-support evidence; it is not yet a safe intervention. |
| EOS-aware adjustment | PHASE41/42 improved EOS learning without eliminating attractors. Keep weight1.5; any new weighting must be its own arm, with premature EOS and language regression controls. Not selected here. |
| Separate instruction/SFT | Useful product-stage objective but cannot establish base-LM stability. Defer until base gate passes; do not mix academic instruction loss into continuation. |
| Data dedup / repeated-pattern weighting | Audit actual pathological training spans and data mass first. Normal formulae, code, list labels and repeated terminology must not be globally downweighted. No corpus edits in this phase. |

[Welleck et al., Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319) studies token/sequence unlikelihood objectives as a way to reduce repetition. Its results motivate a hypothesis, not evidence that UniPilot will improve. The [authors' implementation](https://github.com/facebookresearch/unlikelihood_training) distinguishes training-side changes from decoder changes. No external model/API is used.

PHASE42 `foundation_v31_objective.py` mines repeated3/4-gram next-token alternatives from teacher-forced input and excludes the true next token; loss is mean -log(1-p) over these candidates. Lambdas .01/.03/.05 left runaway100%; onset stayed19 for .01/.03 and18 for .05 versus19 for EOS1.5 control. Small repetition differences were not useful. Candidate sparsity and teacher-forcing/rollout mismatch are plausible explanations, **not established causes**: no gradient-support evidence was saved that isolates them. Reusing that same method with a renamed arm is prohibited.

## Registered future outcome screen

The JSON freezes success targets now: each seed greedy runaway≤25%, sampling≤10%, ≥50 percentage-point greedy improvement versus a matched control, no forced-stop credit, normal-repeat correctness≥95% with no newly introduced severe error in any of math/lists/definitions/code/terminology. LM CE deterioration≤.03 and Top1/5/10 degradation≤1 percentage point; supported Core/Tail CE≤.03; no EOS or context safety regression. Automatic naturalness/semantic proxies may not drop more than5 points and cannot replace human quality review. Paired document/prompt bootstrap95% intervals and per-seed results are required; small sample/insufficient support means inconclusive, never an automatic pass.

Normal controls are checkpoint-independent templates:20 items across5 families, with expected repetition, exact-output requirements and pathological excess rules in the JSON. These are design fixtures, not new generation tests and not source corpus text. Future binding must freeze concrete tokenizer prefixes and IDs before training; boundary labels need independent review first.

Only after valid Formal LR approval and intervention/control review may a PHASE55 spec instantiate approved-LR control plus at most2 intervention arms, using matched parents/seeds/sampler/RNG and separate nonoverwriting outputs. Budget proposal:65,536-token initial gate, maximum262,144 after an explicit interim gate; not several million tokens. No arms, loss weights, training run or canonical continuation have been launched. Architecture review is not yet justified by these confounded observations alone.
