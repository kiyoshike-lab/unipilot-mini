# Generic GPA Simulator — Stage4 Beta

`/gpa` is deterministic TypeScript arithmetic, never LLM calculation. Defaults are editable generic examples, not a university policy. Beta means tested generic functionality, not a Validated/Complete university-specific system.

`GradingPolicy`: named grade IDs/labels/points/inclusion, explicit retake mode, optional policy rounding. `Course`: ID/name, decimal credits, grade ID, inclusion and optional retakeOf. Grade points0–100, per-course credits0–1000, maximum100 courses; invalid/nonfinite inputs, missing grades, duplicate IDs, cyclic/missing/branching retake references are rejected.

Raw GPA = sum(included grade points × included credits) / included credits.0 credits, all excluded or no courses =>undefined (—), not0. Included failures with grade point0 remain in the denominator. Grade-level exclusions and per-course exclusions are separate. Accumulation uses JavaScript double precision without early rounding. Display rounding (3 decimals) and optional policy rounding (UI2 decimals, nearest/floor) are distinct; raw quality points/credits feed subsequent target calculations. This is not arbitrary-precision decimal arithmetic or an official institution's rounding implementation.

Retake options are explicit generic assumptions:

- Count both: count every included attempt.
- Replace old grade: exclude referenced prior attempts and count latest attempts, requiring equal credits per replacement edge.
- Exclude old: exclude prior attempts and count new attempts using their own credits, permitting different credit counts.

Chains are supported; branches/cycles are rejected. A manually excluded latest attempt does not restore the old one. Users must verify whether these assumptions match their actual university. No implicit transcript equivalence or university-specific exception rule exists.

What-if overrides selected grades without changing current grades. Target calculation uses actual current quality points Q and GPA credits C, future GPA credits F and target T: `(T*(C+F)-Q)/F`. A negative requirement is displayed as0; above maximum included policy points is “Impossible under current policy.” With F=0, no division occurs: either target already achieved or future credits required. Already-funded and currently-achieved states are distinct. Future retakes that change the denominator are outside this target scenario; use explicit course/retake inputs to model them.

Explicit per-tab Save, Clear with confirmation and local JSON export. No automatic server save or external data transfer. Stored structures/numerics are validated; invalid data is not silently overwritten. Closing the tab can lose unsaved data. Export includes policy, actual/scenario inputs and computed results; it is a local user download.

Tests cover weighted/decimal/excluded/min/max/zero GPA, retakes, malformed references, targets/impossibility, rounding separation, storage/export and browser save/reload/Clear. Screenshots and browser outputs are local Demo evidence only.
