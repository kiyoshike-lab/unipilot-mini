# PHASE63M runtime binding enforcement

Final gate: `PHASE63M_RUNTIME_BINDING_CPU_VALIDATED`.

The worker and supervisor share `training/phase63_binding_contract.py`. Before any CUDA API, they require a current source manifest, a strict unarmed binding, and a separate SHA-pinned, single-action external authorization record. The source manifest is checked again immediately before CUDA initialization.

CPU-only evidence confirms old binding rejection, PHASE63L candidate rejection, unarmed execution rejection, source-manifest enforcement, binding-SHA enforcement, authorization-purpose enforcement, PID-reuse rejection, and TOCTOU rejection. The authorization record must bind one candidate SHA, one phase, one action, one run, validity interval, scope, target GPU, and exact process identities.

Timeline: the first full pytest run found one compatibility-identifier failure; the identifier was restored without changing enforcement behavior. The post-fix targeted suite passed 38 tests. The post-fix full suite passed 852 tests with 4 skipped and no failures/errors.

The final candidate remains unarmed: `execution_authorized=false`, `no_train_gpu_authorized=false`, `training_authorized=false`, and `approved_processes=[]`.

GPU execution, CUDA context, GPU lock, NO_TRAIN worker, training, optimizer steps, and PHASE63 checkpoint creation were not performed. Cooling remains `PHASE63_COOLING_REVIEW_REQUIRED`; GPU Guard remains `PHASE63_GPU_GUARD_CPU_VALIDATED`.
