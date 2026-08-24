# UniPilot Campus v2.2 knowledge report

- Evaluation: deterministic local source-linked benchmark (1,000 questions)
- Hallucination test: 500 questions
- External AI/LLM judge: not used
- Technical Gate: **PASS**
- Human Gate: **PENDING** (production promotion prohibited)

## Knowledge

- External documents: 610
- Wikipedia: 496
- Government: 105
- University official: 9
- Reviewed FAQ: 1000
- Duplicates removed: 0
- Stale documents: 0
- Fetch failures: 0

The highest-use publisher in the benchmark was **Wikimedia Foundation**. University counts stay below the candidate
target because pages without an explicit reusable license are deliberately excluded. Failed and unsupported
sources are listed in `campus-v22-knowledge-report.json`.

## Evaluation summary

- v2.1 correctness / relevance / grounding / hallucination: 0.803 / 0.209 / 0.000 / 0.500
- v2.2 correctness / relevance / grounding: 0.991 / 0.991 / 0.984
- Unsupported / hallucination: 0.000 / 0.000
- Normal / detailed coverage: 4.737 / 4.984
- Normal / detailed average characters: 469.3 / 989.7
- Retrieval P95: 46.4 ms
- Standalone v2.1 / v2.2 RSS: 323.922 / 431.215 MB

## Remaining risks and decision

- Knowledge gaps: gpa_credit、ai_pc_programming
- Freshness risk: scholarships, tuition, employment, registration and institutional rules require periodic verification.
- RAG failures must be corrected in topics, source coverage or retrieval before considering model training.
- Standard 50M remains stopped; current evidence does not authorize resuming long training.
- Human knowledge review is unscored, so v2.2 cannot be promoted even if the Technical Gate passes.
