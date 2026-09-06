# Academic Constellation / Web v1

Deep academic ink (#090f19), ice cyan (#a2e9f5), mint (#9fdec5), soft amber (#efcb89), restrained indigo accents. System Japanese font stack; no font CDN, WebGL, large animation library, new math dependency, generated hero media, or background video.

## Functional visual language

| Motif | Function |
|---|---|
| Knowledge Rail | Persistent desktop feature navigation; compact mobile bottom navigation |
| Focus Dock | Current Ask/Study mode and honest availability badge |
| Academic Nodes | Study, Report and Research entrances with one-sentence purpose |
| Source Trace | Actual response-associated source metadata, empty/unknown/stale states |
| Study Pulse | Currently selected subject, explanation level and teaching method, not fabricated progress |

The command center explains its university purpose before the first scroll. Chat is central but not the only surface. Source inspection and workspaces form a three-column desktop information hierarchy. Mobile keeps a single main column and bottom navigation. The small constellation is a lightweight decorative SVG, not a simulated knowledge graph.

## Accessibility and state design

Semantic main/headings/navigation, skip link, explicit labels, aria-current, aria-live status/log, visible cyan focus ring, 44px primary mobile controls, form error/retry state and reduced-motion override. Main copy and generated answers use readable system fonts. Empty sources do not masquerade as verified sources. Report/Research retain Coming next labels and explicit shell limitations.

States: first visit, no conversation, no sources, pending connection, incremental response, empty response, stream unavailable with fallback, API error with explicit retry, clipboard denied, storage unavailable, unknown metadata and stale metadata.

## Review

The design uses workspaces, a knowledge rail and source context rather than a centered chat clone. Future-facing styling is limited to a small constellation, thin borders and functional color accents. The hero explicitly names students, study, reports and research. No fake GPA, courses, deadlines, sources or learning progress appear.

Responsive QA covers 360, 390, 768, 1024 and 1440px across the five new/main routes. Automated DOM overflow checks complement desktop/mobile screenshot review. See `web/qa/results.json` for functional checks. Browser fixtures use local Demo data only; live backend availability is not asserted.
