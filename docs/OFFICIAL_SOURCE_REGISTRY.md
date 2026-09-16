# Official University Information — Foundation

PHASE55 / Stage7 / Feature14. `/official-search` is **fictional local fixture search only**, not live retrieval. TEST UNIVERSITY and `official.unipilot.example` are not a real source of university policy. No network, external LLM, background refresh, persistence or automatic Today/Memory/Plan insertion.

The application-controlled OfficialSourceRegistry records university, domain, title/type/URL, academic year, effective date, last verified date, terms note, registration state and fixture provenance. Domain trust is scoped to a university. HTTPS only, no credentials, nondefault ports or trailing-dot hosts. Exact domains by default; explicit subdomain grants require a dot boundary. Similar suffixes and user-controlled URLs cannot grant official status. Production registry onboarding requires independent domain ownership/authority review; a domain suffix alone is not authority.

RetrievalEvidence binds source ID to the exact final URL, body, exact supporting span, claim key and assertion. Redirect chains are not implemented and changed URLs are refused. A future normal HTTP adapter must validate each redirect and DNS/public-network target, response size/type, retrieval timestamp and source identity; no server fetch/SSRF surface is currently exposed. This interface is not permission to scrape or bypass access controls.

States: Verified official (in Demo, only the fixture contract), Official but stale (>90 days), Official year mismatch, Unofficial, Unknown, Conflict. Invalid/missing/future dates, missing terms or evidence refuse verification. Same university/year/claim with different assertions produces Conflict, keeps both sources, and recommends latest-source/contact confirmation. Cross-year documents retain their mismatch rather than being silently merged. Structured claim extraction is a fixture contract, not validated semantic conflict detection on arbitrary prose.

Citation mapping preserves metadata/span and verification notes. Even a fixture's Verified official maps at most to **Span confirmed**, never Citation Engine's reserved Verified authenticity state. Year and last_verified_at are visible even when unknown. There is no synthesized answer when evidence is absent.

Real-university Live retrieval, current-year freshness QA, terms review, independent source authenticity and student workflows are NOT_TESTED. Foundation remains Foundation; no Beta promotion or Production deployment.
