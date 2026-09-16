# PHASE55 handoff — Foundation v4.4 / Academic OS Stage7

QA completed September15 JST; Git finalization resumed September16 JST. This phase completed the new holdout, one-shot LR decision and local product Foundation work. It did **not** resolve the dominant cause of pathological generation or approve a training intervention.

## Research outcome

| Item | Outcome |
|---|---|
| Resolver / root | PROCESS_ENV_Z_ROOT_PASS; process env, resolver and existing paths all `Z:\AI\unipilot-mini\checkpoints` |
| Checkpoint integrity | Six existing C/B×42/123/2026 checkpoints: matching SHA, strict model/optimizer and scheduler/sampler/RNG/metadata checks PASS; detailed migration evidence reusable by same SHA |
| Checkpoint operations | COPY0 / MOVE0 / DELETE0 / OVERWRITE0; no binaries staged |
| Final Blind | SHA ONLY PASS; `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b`; no content access |
| PHASE53 Reserve |27 docs permanently RETIRED_UNSCORABLE; openedNO/scoredNO/fingerprint reconstructionNO |
| Retired Reserve versus supplemental | Near relation UNKNOWN, explicitly no future independence claim |
| Supplemental | Reused591 docs /1,078,706 tokens; no acquisition/window change |
| Internal duplicates |174,345 pairs; exact0/normalized0/near0; preregistered full5-character Jaccard>=.8 |
| Fresh gate | FRESH_HOLDOUT_V2_READY |
| Diagnostic |299 docs /556,546 tokens; no primary LR evidence |
| Confirmatory |172 docs /296,912 tokens; one-shot complete, consumed_for_model_selection=true; never blind again |
| Reserve2 |120 docs /225,248 tokens; SEALED_WITH_FINGERPRINTS; no post-seal body reads, prompts or inference |
| Fingerprints | SHA256, normalized SHA, one-way bottom256 KMV stored before split; metadata-only future approximate comparison |
| Split | Category-stratified SHA order, seed phase55-v2-5501, frozen memberships; tiny categories Diagnostic-only |
| Prereg SHA | `d9e93c9854c2a44abd4dc4a1c6cd76022a79f40a036c3add345c4add36fe16c2` |
| Formal LR | FORMAL_LR_APPROVED_5E5, future research LR use only |
| Paired CE C−B |−0.03015828;95% CI [−0.03113266,−0.02922173]; all3 seed-specific intervals favor C |
| Previous/shared generation policy | GENERATION_POLICY_UNSAFE; both LRs sampling runaway98.9583% |
| Attractor classification | INSUFFICIENT_EVIDENCE for dominance; observational self-copy/EOS/concentration signature |
| Training Fix Gate | MORE_ROOT_CAUSE_WORK_REQUIRED |
| PHASE56 arms | None registered; conditional training-fix preregistration intentionally absent |
| New training / canonical /20M /Foundation Base | NO /NO /NO /NO |

| Equal3-seed metric |5e-5 C |7.5e-5 B |
|---|---:|---:|
| Full-context CE |4.386036 |4.416194 |
| Short-context CE |4.470720 |4.500896 |
| Top1 /5 /10 |24.802% /43.835% /52.443% |24.353% /43.377% /52.038% |
| Terminal P(EOS) |0.009942 |0.009582 |
| Nonterminal P(EOS) |0.000446 |0.000437 |
| Premature EOS argmax |0% |0% |
| Sampling EOS completion |1.042% |1.042% |
| Naturalness /semantic proxies |72.396% /66.406% |71.094% /62.760% |

All8 preregistered clear-material relative safety flags are false; this is not a general safety/noninferiority guarantee. Full seed distributions, EOS rank/Top-k, sampling completion/topic/Japanese/repetition metrics, uncertainty and exact bounded-context method are in `evaluation/foundation-v44-confirmatory-{report.md,summary.json}`. GPU inference used CUDA FP32, no CPU-heavy QA concurrently; maximum confirmatory temperature82°C. New inference ended before all final QA.

Root-cause audit examined900 historical traces and exact3–6gram training frequency/document/category support. Training mapping was verified for all10,012 documents. Of20 frequent generated6grams,17 have zero training occurrences: direct copying alone is inadequate, but causal dominance remains unestablished. All60 normal-control generations ran away and observable shape proxy was0/60, so a non-degradation control has a floor effect. No full hidden/logit-vector cosine was fabricated from top5 trace summaries. Candidate intervention benefits/risks/reversibility and the excluded failed PHASE42 auxiliary are documented in `evaluation/foundation-v44-attractor-root-cause.md`.

## Product and QA

Feature13 `/office-hours`: **Foundation**, local template only. Five modes, exact supplied excerpts separated from general learning steps, hints, reasoning self-check, student professor-question draft, policy/evidence escalation. No professor impersonation, model competence claim, persistence or sending.

Feature14 `/official-search`: **Foundation**, fictional TEST UNIVERSITY fixture/contract only. OfficialSourceRegistry, exact-domain/explicit-subdomain rules, URL binding, last_verified_at/year/effective date/terms, stale/year mismatch/Conflict/Unofficial/Unknown states and missing-evidence refusal. Citation mapping never upgrades fixture authenticity to Verified. Live retrieval is not implemented or claimed.

Memory/Plan remain local-only; Today receives no automatic new-feature inserts. Feature1–12 regression PASS. Mobile widths360/390/768/1024/1440 PASS. Labels, keyboard/focus-visible, error association, aria-live,44px targets, semantic structure and reduced motion checked (not full assistive-technology certification).

- pytest:515 PASS /5 warnings,0 failures. Preexisting generated5 restored with exact SHA.
- Web unit:75 PASS. npm lint/build PASS;25 static outputs.
- Local production-build Chromium Demo:5 suites PASS,175 responsive checks including repeated/populated routes. Artifacts in `web/qa/phase55/`.
- Live Preview: NOT_TESTED. External AI API: OFF. No main/Render/Vercel Production changes/deployments or model promotion.

## Frozen custody / do not rerun

The original `evaluation/phase55/lr-confirmatory-preregistration.json` remains immutable and **untracked** because it contains tokenized fresh prefixes. Its body-free `...-public.json` derivative retains prompt IDs, construction rules, all thresholds and RNG plus the original SHA. This is not a post-hoc amended preregistration. Keep the original local artifact for audit; do not stage it or rerun `freeze`/scoring. Completed inference raw is at `Z:\AI\unipilot-mini\evaluation\phase55`, never Git.

Fresh split bodies are on `Z:\AI\unipilot-mini\data\phase55-fresh-v2`. Reserve2 is sealed; do not open it or the old PHASE54 complete pool after sealing. PHASE53 retired Reserve remains sealed permanently. Only metadata/fingerprints/hashes are usable. Four protected dirty files and five PHASE42 READY records must remain untouched/unstaged. All unrelated preexisting dirty migrations/checkpoint deletions remain the user's state.

## Commit separation

1. `708e141574f7a789db1c6b97e9bd92ebb58c4669` — research: establish clean fresh holdout v2
2. `fb0653539e68b5ae9049ffa701685312114a4225` — research: run one shot lr confirmation and root cause audit
3. `f56704b8bb7b621563109e1b745f47d746298b0b` — web: add ai office hours foundation
4. This handoff/Stage7 integration and QA belong to `web: add official university source foundation` (resolve exact SHA from Git).

Branch is `foundation-research`; push only `origin/foundation-research` after final gates. Original/unchanged main SHA: `b4c21da8976e2c2f95fbcca0499cd915aed8a9df`. Push result and final HEAD/origin match are reported in the task after the command completes.
