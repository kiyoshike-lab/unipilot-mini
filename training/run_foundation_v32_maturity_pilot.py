"""PHASE 43 isolated standard-continuation maturity pilot.

The pilot always starts from the formal 15.360M seed-42 checkpoint, keeps the
official checkpoint immutable, uses the validated EOS weight, and does not use
the rejected repetition auxiliary objective.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from training.device import NvidiaSmiMonitor
from training.foundation_v31_objective import weighted_lm_loss
from training.optimizer import create_optimizer
from training.run_foundation_v30_eos_experiment import load
from training.train_foundation_v15_controlled import macro_batch
from training.train_foundation_v21_ab import file_sha256, random_state


TOKENS_PER_UPDATE = 512
START_TOKENS = 15_360_000
DEFAULT_BUDGET = 256_000
EOS_WEIGHT = 1.5
EXPECTED_SOURCE_SHA256 = "5f2364a42814e40f1f14237db61296176ae63976db43be12c7d834ffb0292ee9"
SOURCE = (
    ROOT
    / "checkpoints/foundation-v28-current/current/seed-42"
    / "checkpoint-tokens-15360000.pt"
)
OUTPUT_ROOT = ROOT / "checkpoints/experimental/phase43/standard-eos-1.5/seed-42"
RESULT_PATH = ROOT / "evaluation/foundation-v32-pilot-training.json"


def checkpoint_path(budget: int = DEFAULT_BUDGET) -> Path:
    return OUTPUT_ROOT / f"checkpoint-tokens-{START_TOKENS + budget}.pt"


def _scheduler_state(source_state: dict, update: int) -> dict:
    state = dict(source_state)
    state["global_step"] = int(update)
    return state


def verify_payload(payload: dict, *, end_update: int, end_tokens: int) -> dict:
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
    checks = {
        "optimizer": bool(payload.get("optimizer_state")),
        "scheduler": payload.get("scheduler_state", {}).get("global_step") == end_update,
        "rng": required_rng.issubset(payload.get("random_state", {})),
        "cuda_rng": "torch_cuda" in payload.get("random_state", {}),
        "sampler": torch.is_tensor(payload.get("permutation")),
        "tokens": int(payload.get("tokens_processed", -1)) == end_tokens,
        "device_metadata": str(payload.get("device_metadata", {}).get("device", "")).startswith("cuda"),
        "precision_fp32": payload.get("precision_mode") == "fp32",
        "experimental": payload.get("experimental") is True,
        "phase": payload.get("phase") == 43,
    }
    return {"checks": checks, "pass": all(checks.values())}


def run(budget: int = DEFAULT_BUDGET) -> dict:
    if budget <= 0 or budget > 512_000 or budget % TOKENS_PER_UPDATE:
        raise ValueError("budget must be a positive multiple of 512 and no more than 512k")
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE 43 pilot requires CUDA")

    device = torch.device("cuda")
    source_sha = file_sha256(SOURCE)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("formal seed-42 checkpoint SHA256 mismatch")

    source_payload, model, optimizer = load(SOURCE, device)
    start_update = int(source_payload["update"])
    updates = budget // TOKENS_PER_UPDATE
    end_update = start_update + updates
    end_tokens = START_TOKENS + budget
    train = np.memmap(
        ROOT / "data/foundation_v11/packed/vocab-4096/train.bin",
        dtype=np.uint16,
        mode="r",
    )

    losses: list[float] = []
    eos_losses: list[float] = []
    non_eos_losses: list[float] = []
    gradient_norms: list[float] = []
    finite = True
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    monitor = NvidiaSmiMonitor(True)
    monitor.start()
    started = time.perf_counter()
    try:
        for update in range(start_update + 1, end_update + 1):
            inputs, targets = macro_batch(
                train,
                int(source_payload["permutation"][update - 1]),
                TOKENS_PER_UPDATE,
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss, eos_loss, non_eos_loss = weighted_lm_loss(
                logits, targets, eos_id=2, eos_weight=EOS_WEIGHT
            )
            if not torch.isfinite(loss):
                finite = False
                raise RuntimeError("non-finite pilot loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                finite = False
                raise RuntimeError("non-finite pilot gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            eos_losses.append(float(eos_loss.detach()))
            non_eos_losses.append(float(non_eos_loss.detach()))
            gradient_norms.append(float(norm))
        torch.cuda.synchronize(device)
    finally:
        elapsed = time.perf_counter() - started
        telemetry = monitor.stop()

    target = checkpoint_path(budget)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.tmp")
    saved = {
        **source_payload,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": _scheduler_state(source_payload["scheduler_state"], end_update),
        "random_state": random_state(device),
        "update": end_update,
        "tokens_processed": end_tokens,
        "experimental": True,
        "phase": 43,
        "pilot_kind": "standard_lm_continuation",
        "eos_loss_weight": EOS_WEIGHT,
        "repetition_auxiliary": False,
        "source_checkpoint": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": source_sha,
        "precision_mode": "fp32",
    }
    torch.save(saved, temporary)

    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    strict_model = DiagnosticTransformerV17(DiagnosticConfigV17(**loaded["config"]))
    strict_model.load_state_dict(loaded["model_state"], strict=True)
    strict_optimizer = create_optimizer(strict_model, 1e-4, 0.1)
    strict_optimizer.load_state_dict(loaded["optimizer_state"])
    integrity = verify_payload(loaded, end_update=end_update, end_tokens=end_tokens)
    integrity["strict_model_reload"] = True
    integrity["strict_optimizer_reload"] = True
    integrity["pass"] = integrity["pass"] and file_sha256(SOURCE) == source_sha
    if not integrity["pass"]:
        raise RuntimeError("experimental pilot checkpoint integrity verification failed")
    temporary.replace(target)

    result = {
        "schema": "foundation-v32-pilot-training-v1",
        "phase": 43,
        "experimental_only": True,
        "device": torch.cuda.get_device_name(device),
        "precision": "fp32",
        "amp": False,
        "seed": 42,
        "start_tokens": START_TOKENS,
        "budget_tokens": budget,
        "end_tokens": end_tokens,
        "updates": updates,
        "objective": {
            "standard_lm": True,
            "eos_weight": EOS_WEIGHT,
            "repetition_auxiliary": False,
        },
        "training": {
            "seconds": elapsed,
            "tokens_per_second": budget / elapsed,
            "mean_weighted_loss": float(np.mean(losses)),
            "mean_eos_loss": float(np.mean(eos_losses)),
            "mean_non_eos_loss": float(np.mean(non_eos_losses)),
            "mean_gradient_norm": float(np.mean(gradient_norms)),
            "max_gradient_norm": float(np.max(gradient_norms)),
            "all_finite": finite,
            "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1_048_576,
            "telemetry": telemetry,
        },
        "checkpoint": {
            "path": str(target.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_sha256(target),
            "source_sha256": source_sha,
            "source_unchanged": file_sha256(SOURCE) == source_sha,
            "integrity": integrity,
        },
        "parallel_cpu_evaluation": "DISABLED",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = parser.parse_args()
    print(json.dumps(run(args.budget), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
