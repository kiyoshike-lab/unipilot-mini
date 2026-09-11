# Student Planner / Today's UniPilot — Stage5

Feature9 is **Beta**, scoped to tested local deterministic records, not university rule validation. `/planner` separates Today / Schedule / Assignments / Attendance. Home shows only explicitly saved user records. There are no generated schedules, classrooms, deadlines, external calendars, LLM arithmetic or server writes. Empty state is empty; Demo fixtures exist only in tests.

## Records and arithmetic

`web/lib/planner.ts` owns schema `unipilot-planner-v1`. Timetable: ID, name, weekday0–6, start/end HH:mm, optional location/instructor. Start must precede end within one day; overlapping classes on the same weekday are rejected. Adjacent endpoints are allowed. Weekly recurring slots do not imply semester/holiday knowledge. Deleting a class requires confirmation, removes linked attendance and detaches (does not delete) assignments.

Attendance stores one record per class/civil date, status present/late/absent/excused. All four counts are shown. The explicitly disclosed generic percentage is present/(present+late+absent); excused is excluded and late is not silently converted to present. Zero denominator displays undefined. The optional user-defined positive absence-count threshold emits a reminder, never “you will fail the course.” No institution's policy is inferred.

Assignments store title, optional registered course, due civil datetime **and its own IANA timezone**, status not-started/in-progress/done, optional priority/note. Current clock determines remaining milliseconds and overdue (strictly past due, unless done). Display-zone edits do not change stored assignment instants. Explicit changing of an assignment's due zone reinterprets its civil deadline and is visible in the editor.

## Timezone and Today

Display/class timezone initially uses the browser IANA zone, with Asia/Tokyo as initial render fallback. Due inputs show their own explicit editable zone. Civil times resolve through Intl timezone data over modern15-minute offsets (years2000–2099). Invalid dates/zones, DST gaps and ambiguous repeated times are rejected instead of choosing an undocumented offset. For recurring classes only, ambiguous/nonexistent DST occurrences are skipped with a visible notice; they are not silently rescheduled. Updating the clock every30 seconds and on focus refreshes Today/overdue. It is not a notification scheduler.

Priority is deterministic: overdue assignments, then next24 hours, then next72 hours, then next class within7 days. Within assignments: earliest due instant first; equal instants use high→normal→low, then stable ID. Done tasks are excluded. “Today's assignments” uses the display timezone's civil date. Attendance warnings are derived solely from explicitly configured thresholds. Recommendations are reminders, not a personalized AI optimization claim.

## Persistence and privacy

Explicit Save writes localStorage key `unipilot-planner-v1`; unsaved inputs do not affect Home. Other-tab storage events refresh saved records. No encryption/account sync; anyone with access to this browser profile may access these records. Do not store unnecessary sensitive personal details.

Clear all removes the Planner key and resets editor state after confirmation. It does not touch GPA/Report/Research data. JSON Export is generated locally. Import validates schema, shape, finite bounds, IDs/references, schedule overlap, attendance uniqueness and timezones, then asks before replacing current editor data; Save is still required to persist. Invalid imports leave existing storage unchanged. Limits:2MB input file,100 classes,500 assignments,2000 attendance records; strings are bounded. A corrupt saved record is not automatically overwritten.

## Evidence / limitations

Planner unit tests cover empty state, add/edit/delete, overlap/invalid times, attendance counts/percentage/threshold, timezone/DST, due/overdue, priority, serialization and rejected inputs. `planner-email-qa.cjs` covers browser persistence, import/export/Clear, Today, error handling, keyboard/focus and five responsive widths. Fixtures are labelled Demo. Official attendance eligibility, holiday/semester logic, reminders, calendar integrations and authenticated student Live QA remain outside the completed scope.
