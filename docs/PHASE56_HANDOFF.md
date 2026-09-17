# PHASE56 handoff — Foundation v4.5 / Academic OS Stage8

Local completion:2026-09-17 JST. Start branch `foundation-research`, HEAD and remote both `d3df3f4eb9155294d2b615792870d7ab8fa8af13`, verified before changes. Repository: `C:\Users\nlgid\Documents\Codex\2026-08-15\files-pasted-by-the-user-unipilot\outputs\unipilot-mini`. The desktop task's default directory is different: always set the repo workdir explicitly.

## ML result

| Item | Result |
|---|---|
| Resolver / checkpoint root | PASS; explicit per-process `UNIPILOT_CHECKPOINT_ROOT=Z:\AI\unipilot-mini\checkpoints` |
| Approved LR |5e-5, FORMAL_LR_APPROVED_5E5 unchanged; no new7.5e-5 selection |
| Confirmatory172 | Not rerun; remains consumed for model selection |
| Reserve protection | PHASE53 RETIRED_UNSCORABLE; PHASE55 Reserve2 SEALED_WITH_FINGERPRINTS / UNSCORED; hash-only |
| Final Blind | Hash-only, expected `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b` |
| Diagnostic consumption |299-document source marked consumed;24 preregistered documents actually scored; never gradient data |
| Runs / seeds |622 total including teacher/perturb/control; C42/123/2026 and minimal baseline42 |
| Hidden / attention | PASS,10 layers; exact hooked/plain logits equal and unchanged tensors |
| Registered loop-lock layer | NOT_ESTABLISHED in all3 seeds; no causal origin-layer claim |
| Free versus teacher | Repetition4 gap+.81933/+.79300/+.80733; all10 layers show larger periodic cosine in free mode |
| Perturbation |300/300 eligible changed suffixes loop again;306 total,303 changed;3 short EOS outcomes disclosed |
| EOS | Low probability, but suppression not consistent: seed2026 probability rises and rank improves all seeds |
| Overconfidence | Entropy falls and top1/margin rise; association, not proof of sole cause |
| Novel loop ngrams |6gram types89.6657% absent from complete train; occurrence-weighted93.4830% |
| Root Cause Gate | MIXED_CAUSE_WITH_ACTIONABLE_TARGET |
| Training Readiness | PHASE57_TRAINING_READY: design/synthetic contract, subject to future runtime preflight |
| PHASE57 arms | Control + A generated-prefix contiguous-cycle unlikelihood with reference veto; no B |
| PHASE57 budget | Seed42 only;64k/128k total gradient-input ceilings, actual63,904/127,808 including auxiliary slots |
| Training / Canonical /20M / Foundation Base | NO / NO / NO / NO |

Main supporting evidence is the three preregistered signals: feedback gap, persistent perturbed basin, mostly novel long loops. Counterevidence/limits: EOS is not uniformly suppressed, hidden cycle-lock threshold.95 is not met, attention still covers earlier context, position is confounded with content, and no controlled architecture/undertraining study exists. Small correlated trajectories are not independent confirmatory evidence. Generation Policy remains unsafe.

See `evaluation/foundation-v45-root-cause-report.md` for the seven-hypothesis supports/contradicts/unknown/confidence table. `evaluation/phase56/phase57-training-preregistration.json` fixes parent SHA, data/equations/.05 coefficient/normal-control safeguards/numeric margins/stop criteria. Synthetic tests do not establish universal semantic safety. Future runtime must reject insufficient training-only negative samples, any missing evaluator, failed state continuity, unsafe normal metrics or partial outputs. PHASE42 teacher-forced repetition auxiliary remains excluded.

## Product and QA

Feature15 `/career` is **Foundation**: manual profile, explicit selected Memory/Planner copies, evidence-linked skills, missing-result-safe Self-PR, fact-only ES length check, generic interview questions/notes, own-key Save/export/Clear. Fixture fake achievements0; no inferred qualification/GPA/role/award/internship/result. Template label is explicit. Company verified-source integration NOT_IMPLEMENTED. Automatic applications/sending and External AI API OFF.

All15 statuses/dependencies/data flows and deterministic/AI/retrieval/source responsibilities are audited in `ACADEMIC_OS_V1_ARCHITECTURE.md`. Feature9–15 storage/API/unsaved boundaries are in `ACADEMIC_OS_PRIVACY_BOUNDARIES.md`. Note the existing shared health GET and server-RAM chat session boundary: local-only Career does not mean all Academic OS chat is device-local. Office Hours and Official Search stay Foundation; no feature is Complete and no UI-only status promotion occurred.

| Local check | Result |
|---|---|
| Repository pytest |526 passed,5 warnings; preexisting5 generated evaluation files restored to identical SHA |
| Observatory/objective targeted tests |11 passed, including five normal-repeat families and pathological-loop gradient direction |
| Web unit |83 passed; strengthened Career foreign nested-field test separately8/8 PASS |
| npm lint / npm build | PASS / PASS; production build26 static pages |
| Browser Demo | All6 suites PASS; core55 route checks +10 populated Evidence; calculators10; Planner/Email15; Memory/Plan75; Office/Official10; Career/integration85 |
| Feature1–14 regression | PASS local client contracts, not live model quality |
| Mobile |360/390/768/1024/1440, no horizontal overflow; all16 routes reachable |
| Accessibility essentials | Labels, keyboard menu/Escape/focus return, focus-visible, aria-live, errors,44px controls, semantic main and reduced motion PASS; not full assistive-technology certification |
| Privacy / isolation | No unexpected feature POST; selected-only copies and unrelated source keys preserved after Career Save/export/Clear |
| Live authenticated Preview | NOT_TESTED; no bypass or Production deployment |

Browser evidence and isolated fictional screenshots: `web/qa/phase56/`. A resumed supplemental Career test initially found the local server stopped (connection refused); restarting the same loopback production-build server resolved it, and the full Career/integration suite passed again. No production service was involved. Node module-type warnings and Next's out-of-repository package-lock warning are non-blocking; dependency/config changes were intentionally avoided.

## Custody, Git and next task

`evaluation/phase56/final-qa.json` records post-run SHA/strict reload/resume-state PASS for all4 observed checkpoints, protected4/READY5 hashes, no incomplete artifacts,622 raw JSON/vector hash pairs, temperature maxima and QA receipts. Existing research checkpoint COPY/MOVE/DELETE/OVERWRITE counts stay0. Raw hidden/logit/text artifacts remain solely under `Z:\AI\unipilot-mini\evaluation\phase56`, not Git. No fresh or Reserve bodies are staged. Sealed sets were never opened for content. Existing dirty checkpoint metadata deletions/resolver edits/raw directories are unrelated user changes and remain unstaged.

Scoped commits:

1. `f14b4502c47161a59ab2468a1a4c46753aec3e63` — research: instrument generation collapse observatory
2. `b5503a5cb28170f75ae6062b9566c2ed92466687` — research: identify attractor cause and register phase57 experiment
3. `6bddbdef14ac0181564aa9b52b76950dda1baeb1` — web: add career foundation
4. The commit containing this handoff — docs: audit academic os v1 integration

Only `git push origin foundation-research` is allowed after local QA; verify final HEAD against remote as the final receipt. `main` remains `b4c21da8976e2c2f95fbcca0499cd915aed8a9df`; no main merge/push, Render Production or Vercel Production action, or model promotion was performed. The final chat records the fourth SHA and push verification.

Do not rerun completed observability/Confirmatory jobs. The observatory/registration receipts are exclusive-write and byte-hash frozen: line-ending changes are source changes and should fail gates, not be silently accepted. `verify_foundation_v45_final.py` is the one-shot precommit verifier bound to this phase's start HEAD; use its receipt for later audits rather than weakening that check. Any PHASE57 training still needs an explicit new task request plus the registered execution preflight; this handoff is not permission to train now.
