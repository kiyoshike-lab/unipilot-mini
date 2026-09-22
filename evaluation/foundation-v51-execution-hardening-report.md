# PHASE62 / Foundation v5.1 — Single-GPU execution hardening

PHASE61 remains **EXPERIMENT_INVALID**. The 2.5e-5 efficacy question is **UNDETERMINED**. PHASE62 performed zero training, zero optimizer updates and no generation/quality evaluation.

## Forensic findings

Gate: MIXED_PROCESS_LIFECYCLE_FAILURE. The host returned partial command output while its OS process remained alive. The assistant wrapper printed only result.output, dropping session/exit information, and the assistant launched two duplicate seed42 Control calls without waiting. This was an orchestration error, not evidence that the host killed training at 30 seconds.
Later execution records show both duplicate calls completed the fixed122-update loop and then failed at the existing checkpoint path. This proves244 additional duplicate optimizer updates and at least610 total updates across the three saved original runs plus two duplicates. Exact total in-memory progress after the third saved checkpoint cannot be established. These duplicate runs were missing from the PHASE61 final report.

| Command | Launch (UTC) | Model-visible partial return (UTC) | Actual host completion (UTC) |
|---|---|---|---|
| Original six-run command | 14:11:04.144 | 14:11:34.374 | 14:13:55.927; exit1 after stop |
| Duplicate seed42 Control #1 | 14:12:04.425 | 14:12:34.603 | 14:13:05.973; Python exit1 |
| Duplicate seed42 Control #2 | 14:12:47.855 | 14:13:18.057 | 14:13:48.469; Python exit1 masked by shell exit0 |

All dates are2026-09-19. The recorded process tree A was shell13160 → venv launcher15136 → Python2792; tree B was shell7960 → launcher18700 → Python15076. Tree B is associated with duplicate#2 by creation time; its CIM command line was unavailable. Duplicate#1 OS PID and exact OS process exit timestamps were not recorded. Exec session IDs are not OS PIDs.
Original seed42 Half-LR and seed123 Control overlapped duplicate seed42 Control execution. Per-update wall timestamps and per-PID CUDA-device inventories were not captured, so exact optimizer-step intersections cannot be reconstructed. Stdout completion logs, command bounds and filesystem save times are retained as evidence, not invented precision.

## Checkpoint custody

| Run | SHA256 | Frozen status |
|---|---|---|
| 42 control | 83cbafa7e3aba0850cba5bc08d8e236ca79c73aa839014699fe2e92ff2b8b0ee | COMPLETED_BEFORE_ABORT |
| 42 half-lr | f86e69df68f964c8375185d3810cf8fd8a30a6d4cd6c0f6cfa9fbddeedd97edd | COMPLETED_BEFORE_ABORT |
| 123 control | 2ba2e6b38d92b2b3f6fc348437b8e98714afd5823e0e12c0cf0bbb2bbd4ac1b4 | INCOMPLETE_STUDY_RUN |

All three pass strict model/optimizer reload, finite state,16,446,464 tokens,32122 optimizer/scheduler step and unchanged full parent permutation. Python/NumPy/Torch CPU RNG loading passes. CUDA RNG byte structure was validated on CPU; no live CUDA RNG roundtrip or model inference is claimed. Seed123 Control remains incomplete because its run receipt is absent. Labels EXPERIMENTAL / NOT_CANONICAL / NOT_PRODUCTION / INVALID_STUDY_CONTEXT live in the custody manifest; binaries and PHASE61 receipts were not modified.

## Execution contract

All four new GPU execution kinds share one lock under checkpoint_root.parent/runtime. Atomic exclusive creation, PID+creation time+hostname ownership, and an OS mutation guard prevent double starts and wrong-owner removal. A suspended child is assigned to a Windows Job Object before it runs. Job accounting includes descendants; receipt publication waits for exit0, empty child tree, stdout/stderr EOF, released CUDA contexts, and a strict verified artifact hash.
The state sequence is PREPARED → LOCKED → RUNNING → CHECKPOINT_SAVED → PROCESS_EXITED → RECEIPT_WRITTEN → RELEASED. The checkpoint state records a verified durable artifact observation; actual child save precedes its exit. A receipt without RELEASED, or a checkpoint without a receipt, is incomplete. There is no cross-file transaction claim.
Crash/timeout retains the ownership lock and an INCOMPLETE/ABORTED record, without killing any live process. Stale cleanup requires a new explicit verification, owner PID absence, creation-time mismatch, clear GPU inventory, clear job tree and persisted child exit evidence (or no child ever started). An unresolved orphan-tree history blocks cleanup.
Live NVIDIA inspection on this WDDM host returns C+G/N/A for display applications. Pure graphics rows are excluded; unknown compute rows block. C+G exceptions require exact current process identity and executable review. This phase creates no broad allowlist and does not claim the current ambiguous list is a clean training preflight.

## PHASE63 registration

CREATED, training_authorized=false, training_executed=false. Original PHASE48 parents; fresh six runs in order42 Control/Half-LR,123 Control/Half-LR,2026 Control/Half-LR. LR5e-5 versus2.5e-5;122 updates each. Scientific keys, evaluation sets,34 safeguards and decision logic match the immutable PHASE60 registration exactly. Only execution controls/output phase change. New roots are experimental/phase63/continuation-stability and Z:/AI/unipilot-mini/evaluation/phase63.
Registration freezes the tested lock and supervisor SHA. Future execution still requires explicit authorization and a bounded one-run worker/strict validator integrated with this supervisor and SHA-bound before optimizer step. Frozen PHASE61 train/train_one entry points are not safe launch paths and must not be reused directly.

## QA and preservation

- targeted: exit0, passed26, failed0, errors0, skipped0, warnings0. Logs/JUnit hashes are in the QA receipts.
- full: exit0, passed782, failed0, errors0, skipped4, warnings4. Logs/JUnit hashes are in the QA receipts.

Windows dummy-process tests verify concurrent rejection, PID reuse, stale/live owner checks, child and grandchild wait, no completion merely from checkpoint existence, crash/deadline handling, receipt requirement, inventory parsing and state transitions. They do not train on GPU.
The intermediate targeted-final receipt records a test-only NameError; it was fixed before targeted-verified and full QA. Earlier receipts are retained rather than overwritten.
Protected4, READY5, dirty JSON5, all179 existing dirty paths, PHASE61 evidence and all checkpoint bytes remain unchanged. Raw logs and checkpoint binaries remain outside Git staging. Main and Render/Vercel Production are unchanged. Web was not edited; npm build/browser QA were not required.
Approved LR5e-5;2.5e-5 UNVALIDATED; Generation Policy UNSAFE; PHASE57 EXPERIMENT_INVALID; PHASE59 CONTROL_STABILITY_MIXED; PHASE61 EXPERIMENT_INVALID. Canonical/20M/Foundation Base/Production:NO.
