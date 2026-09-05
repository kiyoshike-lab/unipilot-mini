# UniPilot Foundation GPU Training (Windows Git Bash)

PHASE 39 uses a separate `.venv-gpu` environment. Do not install CUDA PyTorch into the CPU Python environment. Formal continuation remains FP32; AMP is diagnostic only and is not adopted.

## 1. Activate the GPU environment

```bash
cd /c/Users/nlgid/Documents/Codex/2026-08-15/files-pasted-by-the-user-unipilot/outputs/unipilot-mini
source .venv-gpu/Scripts/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_device_properties(0).total_memory)"
```

Expected essentials are `torch.cuda.is_available() == True`, CUDA runtime `13.0`, and `NVIDIA GeForce RTX 2070 SUPER`.

## 2. Run the non-training preflight

```bash
python -m training.validate_gpu_migration --preflight-only
python -m training.train_foundation_continue --device cuda --target-tokens 15360000 --preflight-only
```

These commands verify the PHASE 38 Gate, all three 10.240M checkpoints, architecture, optimizer, scheduler, RNG/sampler continuity, token budget, and the Final Blind SHA256 without opening its content.

## 3. Re-run the migration and speed gates if needed

```bash
python -m training.validate_gpu_migration --benchmark-steps 50
```

Formal training is blocked unless both `GPU_MIGRATION_PASS` and Speed Gate `PASS` are recorded in `evaluation/gpu-migration-report.json`.

## 4. Continue training in FP32

Run seed 42 first, then the remaining fixed seeds:

```bash
python -m training.train_foundation_continue --device cuda --seed 42 --resume checkpoints/foundation-v26-current/current/seed-42/checkpoint-tokens-10240000.pt --target-tokens 15360000
python -m training.train_foundation_continue --device cuda --seed 123 --resume checkpoints/foundation-v26-current/current/seed-123/checkpoint-tokens-10240000.pt --target-tokens 15360000
python -m training.train_foundation_continue --device cuda --seed 2026 --resume checkpoints/foundation-v26-current/current/seed-2026/checkpoint-tokens-10240000.pt --target-tokens 15360000
```

The runner refuses a target other than the PHASE 38-authorized 15,360,000 tokens, refuses to overwrite completed result JSON, preserves the 10.240M source checkpoints, and never restarts warmup.

## 5. Verify and summarize the completed run

```bash
python -m evaluation.verify_foundation_v28_checkpoints
python -m evaluation.report_foundation_v28
python -m pytest -q
```

Outputs are written under `evaluation/`. Checkpoints are written under `checkpoints/foundation-v28-current/` and remain ignored by Git.

If CUDA OOM, NaN/Inf, a driver error, checkpoint error, or an unexpected regression occurs, stop. Do not silently continue on CPU; resume again from the unchanged 10.240M PHASE 38 checkpoint only after the cause is understood.
