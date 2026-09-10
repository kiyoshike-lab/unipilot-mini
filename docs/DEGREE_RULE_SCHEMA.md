# Course/Credit — Stage4 Foundation

`/degree` is an explicitly fictional **TEST UNIVERSITY** prototype, not an official graduation audit. Feature7 remains Foundation. No real university rules or personal transcript are imported or generated.

`DegreeRule` fields: university, faculty, department, curriculumYear, ruleVersion, effectiveFrom/effectiveTo, category, requiredCredits, courseConstraints, groupConstraints and source metadata. A source requires URL or document ID, verified date, matching year/version and independent verification evidence for real-rule use. Metadata typed by a frontend is not independent verification. The `independentlyVerified` flag is a trusted-boundary contract for a future authenticated source verifier, not a browser-toggle feature; never trust an unvalidated client JSON assertion. The current UI exposes only fixture/Unknown/conflict scenarios.

`CreditCourse`: ID/name, category, decimal credits, completed and included. Deterministic totals/category totals use completed included rows only. Remaining credits = max(0,required−earned). Named-course and group-credit constraints are separate outputs, never replaced by a total-credit-only graduation claim. Duplicate IDs and invalid/nonfinite credits are rejected.

Version safety: university/faculty/department/year/version/effective period must agree across rules; duplicate category definitions or conflicting versions return VERSION_CONFLICT. No blending or silent “latest version wins.” Invalid dates/rule numbers return INVALID_RULE. Missing/unverified source returns UNKNOWN and no required/remaining-credit claims; raw user-credit totals remain available. The future caller must additionally select and authenticate the correct effective curriculum for the student before invoking this domain engine.

Only explicitly fixture-marked TEST UNIVERSITY rules can calculate without official evidence, and the result is always FIXTURE_ONLY. Example TOTAL12/general4/specialist8 are fabricated test values, prominently labeled, never described as actual degree requirements. Even trusted-rule arithmetic does not determine all exceptions or final graduation eligibility; advisers remain responsible.

Inputs stay in component memory; reload/Clear discards them. No server save, rule fetch, LLM arithmetic, official university search or external AI API is enabled. Tests cover category/remaining/excluded/decimal credits, missing sources, year/version/date conflicts and course/group constraints.
