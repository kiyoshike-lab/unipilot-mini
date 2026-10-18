# PHASE63O3 CPU-only closure

Gate: `PHASE63O3_EXTERNAL_PROCESS_EXCEPTION_CONTRACT_CPU_VALIDATED`.

The NO_TRAIN path now accepts only exception bundles contained in the separate, SHA-bound external authorization. Each bundle is exact-instance bound to PID, creation time, host, canonical executable path and executable SHA-256; it also binds the authorization ID, binding SHA, implementation HEAD, source manifest, GPU UUID, purpose, phase, expiry and single-use lifecycle.

The runner rejects the legacy display/PID exemption mechanism. It monitors before lock-protected child launch, throughout the child lifetime and after exit. A missing formerly-approved external process is permissive; an unapproved, restarted, changed or newly appearing C/C+G process fails closed. Training does not receive this exception bundle.

CPU fixture checks passed: 104 tests. Full pytest was not run because CUDA non-initialization cannot be guaranteed. No CUDA context, GPU lock, GPU workload, NO_TRAIN execution, training, optimizer step, data consumption or PHASE63 checkpoint occurred.

The regenerated source closure has 43 files and SHA-256 `ee3e79f50aa2edbcd164603e0b56150c4ca50df7cc98d6a9afdd761e564f8c2f`; N-RERUN-3 binding `9cea350f1e57e22612e9df2337c5bfd8a0ebd084dbb11f23fdc09237a9d0836b` is stale. A fresh N-RERUN-4, fresh inventory, and separate current-instance user approval are required before any future O execution.
