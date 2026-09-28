# PHASE63K6 thermal runtime guard

Result: `PHASE63K6_THERMAL_RUNTIME_CPU_VALIDATED`.

`training/phase63_thermal_guard.py` is a CPU-testable, injected telemetry state
machine. It validates timestamped samples, records a strict receipt, pauses at
the registered 80°C boundary, resumes at or below 65°C, and aborts at 85°C,
hardware thermal slowdown, or any telemetry failure. A sample older than six
seconds, a timestamp that does not advance, or a timestamp regression is a
fail-closed abort. The six-second limit is an execution-safety value derived
from the existing five-second query bound plus scheduling jitter; it does not
modify the scientific protocol.

The worker now performs binding validation, authorization, lock ownership, and
source-hash validation before thermal precheck, and thermal precheck before any
CUDA/model action. Each training update asks the guard for permission before
permutation/macro-batch/data/RNG/optimizer work. Disk reserve and thermal work
are separate. The old exception-swallowing monitor is not on the PHASE63 live
path; the new watchdog shares an explicit abort state with the worker.

All K6 validation was CPU-only. It did not query actual GPU inventory, acquire
a GPU lock, construct a CUDA context, launch NO_TRAIN or training, call an
optimizer step, create a checkpoint, or run generation evaluation. The old M
candidate is intentionally unmodified and `STALE_AFTER_K6`; it must be replaced
only by a later fresh, unarmed binding phase.

Cooling remains `PHASE63_COOLING_REVIEW_REQUIRED`. This implementation closes
the K5 runtime-control gap but does not reclassify physical cooling evidence.
