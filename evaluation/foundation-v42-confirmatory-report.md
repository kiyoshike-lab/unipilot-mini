# PHASE53 — Fresh acquisition and confirmatory gate

Fresh Holdout Gate: **FRESH_HOLDOUT_TOO_SMALL**. Confirmatory Gate: **CONFIRMATORY_DATA_INVALID** (inadmissible population, not an observed model failure). LR: **FORMAL_LR_UNRESOLVED**. No confirmatory inference or new training.

## Evidence and acquisition scope

The Foundation v1.1 manifests use `latest` dump URLs, so the dump snapshot date itself is **UNKNOWN**, not guessed. Individual training/validation/test revision metadata range from2005-09-03 to2026-08-24T10:02:56Z. The maximum recorded retrieval timestamp is2026-08-27T13:21:24.057451+00:00. The original dump SHA256 values, URLs and retrieval evidence are preserved in `phase53/acquisition-plan.json`.

A checkpoint-independent plan was frozen before acquisition. It requested the most recent500 namespace0 **new page initial revisions** in the fixed August28–September10 window from the official Japanese Wikipedia Action API. This is a bounded500-candidate sample, **not an exhaustive crawl of every page in that window or proof that no larger pool could be obtained**. No model output was used to choose documents. Content lives under `Z:\AI\unipilot-mini\data\phase53-fresh`, separate from checkpoints, outside Git. Every admitted page retains source/project/page ID/revision ID/timestamp/retrieval/license/attribution/cleaning notice and content hashes. Text license: CC BY-SA4.0; source and history attribution are retained under [Wikimedia's terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use). No source prose is displayed in this report.

Existing Foundation cleaning and600..12000 cleaned-character bounds admitted129 documents /174885 tokenizer tokens. The admitted initial revisions are September8–10, strictly later than the recorded corpus retrieval bound. Exclusions:290 character bounds,71 list/navigation-heavy,6 low-Japanese ratio,4 residual markup. No lower-quality replacement material was added to meet a count.

Categories: general92, education13, history10, science9, society/economics3, computing2; mathematics and language0. This is materially unbalanced. Category support cannot be treated as representative confirmation. Future acquisition needs a broader, checkpoint-independent source/category design and adequate document count before scoring.

## Leakage / sealed splits

Known comparison population:11916 distinct historical text fields from all Foundation v1.1 train/validation/test documents plus PHASE52-listed JSON evaluation banks. Checks: cleaned exact SHA256, NFKC/whitespace/casefold hash, known Wikipedia source-page identity, existing32-bit5-character SimHash Hamming<=3 followed by normalized5-gram Jaccard>=.8. Admitted candidates are also added to the index to reject intra-pool duplicates. Exact/normalized duplicate exclusions0; near-duplicate exclusions0; known source-page exclusions0. Near-duplicate screening is not proof against paraphrases, missing external history or transformed short passages. Final Blind was excluded from parsing, hash only.

Deterministic ordering: SHA256(`phase53-v1|page_id`). Diagnostic77 /100094 tokens; Confirmatory25 /36960; Future Reserve27 /37831. Raw content stays in the local corpus store; only metadata/hashes are versioned. The future reserve is sealed against model inference/scoring and discretionary human selection. Do not reopen or resplit it to improve results. All three splits have `consumed_for_model_selection:false`; none has been scored.

The250-document requirement was frozen before acquisition.129 does not pass, even though total tokens exceed100000. We do not relax that requirement or use the25-document confirmatory split anyway. The placeholder preregistration explicitly records why no metric/decision registration was activated. EOS/Core/frequency/context/sampling metrics for both LR arms are NOT_RUN, not zeros or failures. PHASE51's5e-5 comparative preference remains historical, not independent confirmation.

## PHASE54 handoff

Do not promote or retrain a canonical checkpoint on these results. Obtain a sufficiently large, independently selected licensed pool, retaining the sealed reserve and access history; preregister support, safety margins, decoder/RNG, paired-document analysis and one-shot consumption before scoring. Only after an LR candidate is supported should lineage/parent SHA/optimizer/scheduler/sampler/RNG reproducibility determine whether existing experiments are promotable or a canonical continuation needs retraining. No promotion,20M or Foundation Base permission is issued here.
