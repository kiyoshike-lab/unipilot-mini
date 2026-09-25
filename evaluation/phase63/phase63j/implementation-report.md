# PHASE63J — GPU Guard Revision, CPU-only validation

Final gate: **PHASE63_GPU_GUARD_CPU_VALIDATED**
Cooling gate: **PHASE63_COOLING_REVIEW_REQUIRED**

## Scope and result

This implementation is CPU-only. No GPU inventory query, GPU lock acquisition for a CUDA run, CUDA context, CUDA tensor, model CUDA load, worker launch, training, `backward()`, optimizer step, checkpoint creation, or generation evaluation was performed.

The revision adds a provider-injected external-compute monitor, exact instance approval validation, a worker-side live lock/parent/Job assertion before CUDA code, stream rehashing for the generic receipt, and a strict versioned NO_TRAIN receipt validator. Missing or ambiguous data is a block. C+G has no automatic exception; a future exception needs one current, hash-bound, expiring, single-use `no-train-gpu-supervisor` approval. Pure C is never exception-exempt.

The execution binding at `evaluation/phase63/execution-binding.json` was deliberately not edited. It is now stale because the worker, supervisor and guard source hashes changed; any later GPU test needs a fresh binding, cooling resolution, fresh process review and separate explicit approval.

## Tests

- Targeted CPU-only pytest: **48 passed**, exit 0.
- Full pytest in a persistent terminal session with `CUDA_VISIBLE_DEVICES=""`: **823 passed, 4 warnings**, exit 0.
- The initial detached full-suite attempt is not used for the gate: its parent shell disappeared before Job Object tests finished. The resulting missing-parent path is now explicit `HOST_PARENT_EXITED_LOCK_RETAINED`; the persistent-session rerun passed.

`tests/test_gpu_migration.py` performs a collection-time `torch.cuda.is_available()` check with CUDA hidden and has a CPU monkeypatch test. No PHASE63J test invokes a CUDA workload, and no CUDA context was created.

## Source binding impact

| File | Before SHA256 | After SHA256 |
|---|---|---|
| `training/gpu_execution_guard.py` | not present | `38e35c472c25d22c43e4dcaaedef59b64d17eb42b82e356af86b26abd80c6874` |
| `training/gpu_execution_lock.py` | `1a1b508701cdc45b3a87e8ec6e0c1bcf177db20e27e9fdd60f1c2206e806439e` | `5d95fb38def0bc6588eb13777b20b0e942db17927aecb8dbc7b9c444713f46c9` |
| `training/exclusive_cuda_runner.py` | `62698095a30f508fce9e1224d96a1c5600607e127b5d98347a6167f97d8f9b12` | `31cf49b804bceb7c6d54ead7ac6993b5d4059de948da9c0433c67d3082ea994c` |
| `training/phase63_worker.py` | `e9bba5b18a313973285a3708dde1709ece3b2fa90f7cb341e8c5cdd046279613` | `008867aecd6d870c124df4c9bc40a828f8a4e41783bf7a073e6f9f9b0bcf5c00` |
| `scripts/run_phase63_study.py` (validator) | `499e159d74455b2224bcf70a35a2c739fe20d7ed9f0d356055109ea6151dff6c` | unchanged |

## Preservation

The PHASE63I baseline manifest (48 pre-existing evidence/source files) is unchanged except for the three authorized guard source files above. The protected4, READY5, dirty JSON5, original dirty179, PHASE61 frozen evidence and PHASE63 preregistration checks pass. PHASE63 checkpoint count remains 0 and no runtime GPU lock exists.

Branch: `foundation-research`; start HEAD: `af81a92071b14a7adececea82f5809f1fe0be6d5`; origin: `44d4caf132094ac1ba6567ebd3b8d963a4b2cfd8`. The existing two unpushed commits are retained. This report does not authorize a GPU test or a scientific run.
