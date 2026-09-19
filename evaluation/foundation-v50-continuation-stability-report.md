# PHASE61 / Foundation v5.0 — Continuation Stability Experiment

**EXPERIMENT_INVALID**

RESOURCE_ISOLATION_VIOLATION: two CUDA Python process trees overlapped after the execution host returned before a foreground training command had completed. The study must fail closed; no retry, extension, evaluation, checkpoint deletion, overwrite, or promotion is allowed.

Three checkpoints were atomically saved before the stop: seed42 Control, seed42 Half-LR, and seed123 Control. All passed strict model/optimizer/scheduler/sampler/RNG reload checks. The seed123 Control run receipt was not produced before the abort, so it remains incomplete. Every saved checkpoint remains EXPERIMENTAL, NOT_CANONICAL and NOT_PRODUCTION. None is evaluated or can support any efficacy, LR-selection, or generation claim.

The remaining registered runs were not started after the abort. The three parents were re-hashed and strict-reloaded unchanged. No checkpoint was copied, moved, deleted, overwritten, renamed, staged, or promoted.

Approved research LR remains 5e-5. The 2.5e-5 arm remains an unvalidated stabilization candidate only. Formal LR changed: NO. Generation Policy: UNSAFE. PHASE57 remains EXPERIMENT_INVALID; PHASE59 remains CONTROL_STABILITY_MIXED. Foundation Base, 20M and Production: NO.
