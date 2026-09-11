# PHASE53 — Track C generation-policy study

Track C is **DONE**. Best safe decoder candidate: **NONE**. Attractor Gate: **GENERATION_POLICY_UNSAFE**. This is a paired decoder study, not Formal LR evidence: `FORMAL_LR_UNRESOLVED` remains unchanged. It does not repair the model, permit staging or authorize training.

## Fixed scope and integrity

Only24 existing PHASE51 prompts were used, in their frozen sorted-ID order; ordered prompt SHA256 is `82e394d70cf85f25b193610dca8502779ee444fe23e2a01da0ec631cb34e6c28`. C=5e-5 seed42 and B=7.5e-5 seed42 checkpoint bytes matched their preregistered SHA256 values. CUDA FP32 ran with TF32 disabled, two fixed sampling bases (53000/53100),64 new-token cap, and no training/save. The registered modes were greedy,temperature0.7,top-k50,top-p0.90 and repetition penalty1.10. No grid, prompt, seed, metric or threshold was added after results.

Fresh diagnostic/confirmatory data and Future Reserve were not opened, read, scored or used. Final Blind remains hash-only. Raw per-prompt generation artifacts retain arm/mode/RNG/loop/completion/repetition/proxies locally and are not staged. Results aggregate all20 completed raw runs; thermal maximum was58C, below the registered80C stop.

## Greedy reproduction and registered alternatives

Greedy budget-exhaustion runaway reproduced at **100%** for both arms (C median loop onset13, repetition-1 .857; B median12, repetition-1 .834). Thus the historical greedy issue did not silently disappear under the fixed scope.

| Arm / mode | Runaway | Completion proxy | Character valid | Topic-overlap proxy | Repetition-1 |
|---|---:|---:|---:|---:|---:|
| C greedy | 1.000 | .000 | 1.000 | .423 | .857 |
| C temperature .7 | .958 | .042 | .813 | .174 | .337 |
| C top-k50 | .979 | .063 | .917 | .218 | .468 |
| C top-p.90 | 1.000 | .042 | .917 | .208 | .423 |
| C repetition1.10 | 1.000 | .000 | 1.000 | .314 | .790 |
| B greedy | 1.000 | .000 | .958 | .418 | .834 |
| B temperature .7 | 1.000 | .063 | .813 | .167 | .340 |
| B top-k50 | .979 | .042 | .979 | .199 | .449 |
| B top-p.90 | 1.000 | .146 | .979 | .184 | .398 |
| B repetition1.10 | 1.000 | .000 | .917 | .308 | .803 |

The sampling settings reduce repetition and raise automatic naturalness/semantic proxies in some cells, but no mode satisfies every preregistered screen: runaway≤.20, at least.50 absolute runaway reduction, completion≥.50, semantic≥.50, character-valid≥.95 and no material decline versus temperature on its comparison checks. The automatic proxies are not human-quality evidence. Repetition penalty reduces repetition modestly but leaves runaway at100% and is decoder mitigation only, never `MODEL_FIXED`.

## Loop-stop replay safety

The fixed replay detector is first three adjacent identical cycles (width1–4) after at least8 generated tokens. It is post-hoc **replay**, not natural EOS. Greedy replay reduces budget-exhaustion runaway C:1.000→.250 and B:1.000→.333, but triggers on75%/66.7% of outputs and saves34.0/28.3 tokens per output. Every triggered result is a non-EOS truncation; triggered EOS completions are0. Temperature replay still leaves runaway .771/.813.

No labelled normal-repeat math/list/definition controls were preregistered, so a false-positive rate cannot be estimated. The high trigger rates plus non-EOS truncation mean replay is **not** a safe completion policy or staging candidate. It cannot be represented as quality improvement merely because it stops a loop early.

## Boundaries and handoff

No safe decoder candidate exists in the frozen grid. Do not search new temperatures, penalties, thresholds or stop rules to manufacture one. A future policy study needs preregistered normal-repeat controls, independent human quality/safety assessment and a separate authority decision. Fresh Holdout remains `FRESH_HOLDOUT_TOO_SMALL`; Confirmatory Gate stays `CONFIRMATORY_DATA_INVALID`. New training:NO; canonical:NO;20M:NO; Foundation Base:NO.
