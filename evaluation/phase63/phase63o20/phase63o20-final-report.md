# PHASE63O20

Final Gate: `PHASE63O20_PRE_CHILD_ABORT_RECOVERY_CPU_VALIDATED`

O19 showed that the existing O5L proof unconditionally required `child-created.json`. O20 adds the isolated `PRE_CHILD_ABORTED` lifecycle. New executions persist `CHILD_NOT_CREATED_PROVEN` only while `proc is None`; a missing child record is never sufficient. Historical O-RETRY-9 can use the stricter legacy-equivalent proof only when its LOCKED, prelaunch and ABORTED evidence agrees and every possible post-child artifact is absent.

Starting HEAD: `e66157ec689db41a49ecf1fbf29e653c9cb09b04`.

Changed production files: `training/exclusive_cuda_runner.py`, `training/gpu_execution_lock.py`.

Added test: `tests/test_phase63o20_pre_child_lock_recovery.py`.

Old O5L child-exited path: PASS. New pre-child path: PASS. Unknown/unreadable child state, live owner, Job not clear, mutation guard held, CUDA-created state, nonzero forward/optimizer/checkpoint state and training side effects: REJECT.

The CPU-heavy guard is unchanged. It does not measure CPU utilization: it blocks on one process-table command-line substring scan for `-m pytest`, `npm run build`, or `playwright`.

Targeted CPU tests: 47 passed. New closure: 44 files. New manifest: `830865c15f530dd2ef54c7136768476e8360a49ade57785dfa21edfd6400f347`. The prior binding is `STALE_AFTER_O20`.

The actual O-RETRY-9 stale lock is unchanged. A read-only proof derives `PRE_CHILD_ABORTED` from its strict legacy lifecycle equivalent; no recovery authorization or no-train authorization was created. CUDA was not created; forward and optimizer steps are zero; training did not run.
