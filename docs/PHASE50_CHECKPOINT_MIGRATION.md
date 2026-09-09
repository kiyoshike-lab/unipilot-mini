# PHASE50 — Z Checkpoint Migration Gate

The checkpoint save/reference root is configured by `UNIPILOT_CHECKPOINT_ROOT`, now set to `Z:\AI\unipilot-mini\checkpoints` for this user's environment. Scripts do not need per-phase drive literals. Already-running shells may retain the old D value; use the explicit environment setting shown below or launch a fresh shell.

```powershell
$env:UNIPILOT_CHECKPOINT_ROOT = 'Z:\AI\unipilot-mini\checkpoints'
```

This changes routing, not checkpoint content. Existing `checkpoints/...` metadata is retained and resolved via the configured root. Absolute historical source metadata is not rewritten. The pre-existing storage infrastructure and its unrelated uncommitted edits are retained; this phase does not stage all those edits or the pre-existing removed manifests/logs.

## Verified inventory

All 19 required files were absent from the target and found SHA-identical to PHASE50 preflight records under `D:\UniPilot\active\checkpoints`. They were copied to corresponding relative paths on Z; all D originals remain. Total copied payload is approximately 4.16 GiB. Exact sizes, source/destination paths, hashes and per-component checks are in `evaluation/phase50/z-migration.json`.

| Relative lineage | Seeds | Tokens | Count |
|---|---|---:|---:|
| foundation-v33-context-gate / gate-2 | 42, 123, 2026 | 15,872,000 | 3 |
| foundation-v35-thermal-short-gate / gate-1 | 42, 123, 2026 | 16,128,000 | 3 |
| experimental / phase47 / arm-A | 42 | 16,128,000 | 1 |
| experimental / phase47 / arm-B | 42 | 16,128,000 | 1 |
| experimental / phase47 / arm-C | 42, 123, 2026 | 16,128,000 | 3 |
| experimental / phase48 / arm-B | 123, 2026 | 16,128,000 | 2 |
| experimental / phase48 / arm-C | 42, 123, 2026 | 16,384,000 | 3 |
| experimental / phase49 / arm-B | 42, 123, 2026 | 16,384,000 | 3 |

The historical 16.384M files are existing experimental comparisons, not permission to train above the PHASE50 candidate limit. No new training or canonical promotion occurred.

## Gate evidence

`Z_CHECKPOINT_MIGRATION_PASS`: all 19 source/destination size and SHA256 pairs match; strict model and optimizer reload pass; finite FP32 model/optimizer, scheduler step, optimizer step, sampler/permutation and seed/LR/processed-token metadata pass. Python, NumPy, CPU PyTorch and CUDA RNG states were restored into independent generators and compared without taking a training step. Model/optimizer/scheduler/permutation/RNG fingerprints match before and after copying. Parent metadata is retained and its recorded SHA was matched to the existing source parent file, including D archive where applicable.

No checkpoint was deleted, overwritten, moved, regenerated or staged. Copy destination is opened exclusively (`xb`); any collision or incomplete destination blocks the gate instead of replacing it. Existing smoke-test files on Z were inventoried and left alone. No zero-byte/.partial/.tmp/.incomplete artifact was found in the initial target inventory. No required checkpoint is missing after migration.

The four protected files remain at their original repository paths with original hashes. READY 5 files were already relocated by prior storage work: arm-B is in D active, arms A/C/D/E in D archive. Their original hashes all match; this phase does not change or copy/stage them. Check the recorded paths in the migration evidence.

Initial Z free space: 83,948,449,792 bytes. After migration: 79,478,538,240 bytes. The 19 pairs and nine protected artifacts were hash-rechecked on resumption. The persistent user environment was then updated from D active to the approved Z root. Machine-wide environment and GPU settings were not changed.

## Reuse and safety boundaries

The existing ML results, fixed supported-tail membership and `FORMAL_LR_STILL_UNRESOLVED` decision are reused. No GPU training. Final reporting only summarizes saved evaluations. Report readiness is separate from formal LR approval.

`scripts/migrate_phase50_checkpoints.py` is a manually invoked, copy-only gated migration utility, not a startup hook. It requires an explicitly configured destination and source roots. Do not rerun a completed migration to recreate artifacts. If a destination is partial or mismatched, stop for investigation; there is no delete/overwrite repair path.

Web final QA runs only after the migration gate. Push is restricted to `foundation-research` after final tests. No main merge/push, Render Production deployment or Vercel Production deployment is part of this work.
