# Feature15 Career Foundation

Route `/career`, Academic OS Stage8. Status **Foundation**, not Beta: there is no real-student quality/fairness evaluation, company retrieval or model integration.

The profile is manually entered: interests, actual courses, projects, research, optional work/club activities, goals and self-declared skills. Evidence facts have source kind, title, detail and source identifier. Skills may link to a fact; missing evidence remains visibly unverified. No qualifications, GPA, internships, roles, awards or research outcomes are inferred.

Memory/Planner are not read merely by opening Career. “候補を表示” explicitly reads their saved local records; “Import selected items” copies only checked records into an unsaved Career draft. Showing candidates is not importing them. Imported coursework is registered, not proven completed. Import never invents skills or changes originals.

- Self-PR displays Evidence / Strength / Example / Result. Missing Result remains missing; the system does not embellish an accomplishment.
- ES displays the user's question and facts with a Unicode-codepoint count, limit and over-limit notice. It does not automatically shorten or rewrite facts. The destination's own counting rules may differ.
- Interview preparation uses generic questions tailored by quoting the student's supplied goal/example, preserves answer notes and asks evidence/next-step follow-ups. It does not predict a company's selection process.
- All outputs are labelled **Template / structured draft**. Corporate culture, salary, hiring patterns and selection details are not asserted. Official/verified source integration is **NOT_IMPLEMENTED**.
- Save uses only the versioned Career localStorage key. Export downloads current validated Career data, excluding unrelated fields. Clear removes only Career data, including imported copies. Original Memory/Planner records are retained. Automatic applications, email submission and external AI API are OFF.

Validation caps100 facts/skills and6000-character strings, checks unique IDs/evidence references, and fails closed on corrupt saved records. There is no automatic overwrite on restore error. Removing evidence unlinks dependent skill/PR references. Exports strip unknown top-level and nested fields. This validates shape, not factual truth: student statements stay unverified.

Local unit tests cover manual roundtrip, selection/consent, missing evidence/results, evidence deletion, ES limits, interview notes and corrupt/foreign-field rejection. Browser QA covers explicit save/import/export/clear and source-storage isolation, absent/present result, no hidden POST, all16 routes, five widths and accessibility essentials. Live Preview and student outcome evaluation remain NOT_TESTED.
