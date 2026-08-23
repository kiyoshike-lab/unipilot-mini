# University knowledge registry

UniPilot Campus reads university-specific records only from a university subdirectory and only when every record has `enabled: true`, an official `source_url`, and a `retrieved_at` date.

Suggested layout:

```text
knowledge/universities/<university-slug>/
  regulations.jsonl
  syllabus_help.jsonl
  student_handbook.jsonl
```

Do not store copied handbooks or scraped page bodies. Store a short project-authored factual summary, the official page title, URL, retrieval date, applicable academic year, and a quotation-free verification note. Records remain disabled until a human has checked the official source and its permitted use.
