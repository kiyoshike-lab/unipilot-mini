# PHASE52 — Fresh confirmatory evidence audit

Availability: **PROVENANCE_UNCERTAIN**. Confirmatory Gate: **CONFIRMATORY_DATA_INVALID** (no admissible fresh population established; not a measured model failure). Formal LR Candidate Gate: **FORMAL_LR_UNRESOLVED**.

No confirmatory inference, new training, canonical continuation or20M permission. Foundation Base remains NO. FinalBlind SHA256 matched `fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b`; its content was not opened. It is prohibited for this gate.

## Provenance evidence

The inventory covers75 candidate files: validation/test documents and packed splits, fixed prompt banks, human sets and evaluation/holdout/benchmark families. Each row records path, SHA256, count/count basis, first tracked commit, known first-use phase, references, influence and eligibility. Counts are null only where a packed artifact has no audited document count or a manifest has no item population. UNKNOWN influence is not silently interpreted as unused; no candidate is marked eligible on the basis of its filename.

- Foundation v1.1 validation:146 documents, repeatedly used for training progress, architecture/learning-rate selection, frequency thresholds and EOS/Core/Sampling failure diagnosis through PHASE51. Ineligible.
- Foundation v1.1 test:157 documents. `evaluate_foundation_v11_completion.py:58` reads test documents to create the50-item completion bank. This evaluator was introduced in commit `defa0da` (Foundation v1.1); the bank is reused in `investigate_foundation_v14.py:1038` for language-emergence diagnosis. The completion bank mixes manual and test-derived prompts. The split is exposed, even though the file is called test. Do not select its apparently unused remainder after seeing previous failures.
- Other Foundation/Campus/generic prompt and human banks: local code/report references and all local Git refs were searched. First tracked commit is not a first-access log; off-repository use, source overlap and clean custody cannot be established. No verified independent fresh set was found. Historical human-approved items are not automatically held-out human evaluation.
- FinalBlind: excluded by instruction, hash only. Its availability does not authorize opening it.

An all-ref history search for `documents/test.jsonl.gz`, `base-completion-50.json`, `validation.bin` and `fixed_prompts` is saved in the table. Ordinary non-protected dataset structures were counted locally without displaying source prose. Dataset contents/checkpoints were not modified.

## Gate outcome and future evidence

`phase52/confirmatory-preregistration.json` records why no gate was registered, rather than inventing thresholds or laundering an old test split into a fresh one. EOS, Core and Sampling confirmatory evaluation are all **NOT_RUN**. There is no confirmatory set path/SHA to report.

Future design: an independent custodian should collect licensed/consented Japanese documents and student-authored prompts, record immutable IDs/hashes/access logs, and deduplicate against all training and historical evaluation material before any candidate output. Prespecify sampling strata, precision/power-based sample size, blinded multiple-rater rubric, metrics/safety margins, multiplicity, missing-data policy, decoder/RNG and paired document/prompt bootstrap. No new collection or human recruitment was performed in this phase. Human naturalness/relevance judgments must remain separate from automatic surface proxies.

PHASE51 remains comparative evidence only:5e-5 outperforms7.5e-5 in LM/top-k/Middle/Core/Supported Tail/Context, but C123 Sampling, C2026 EOS/Core and additional B failures remain. No independent confirmation, formal LR promotion or training permission follows from this audit.

## Safety

Process environment explicitly Z; Python resolver and all19 manifest destinations agree. Nine prior diagnostic checkpoint SHA256 hashes match; PHASE51 strict/model/optimizer/scheduler/sampler/RNG checks are reused with matching content hashes. COPY/MOVE/DELETE/OVERWRITE all0. Protected4 and READY5 remain untouched. Huge PHASE51/52 raw traces remain local, not staged. No GPU inference was needed for the fresh-data audit or existing-trace analysis.
