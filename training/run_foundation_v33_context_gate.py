"""PHASE 44 staged three-seed CUDA FP32 context-gate continuation."""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from training.foundation_v31_objective import weighted_lm_loss
from training.optimizer import create_optimizer
from training.run_foundation_v30_eos_experiment import load
from training.train_foundation_v15_controlled import macro_batch
from training.train_foundation_v21_ab import file_sha256, random_state


SEEDS = (42, 123, 2026)
TOKENS_PER_UPDATE = 512
GATE_TOKENS = 256_000
FORMAL_START_TOKENS = 15_360_000
EOS_WEIGHT = 1.5
EXPECTED_PARAMETERS = 19_514_880
EXPECTED_FORMAL_SHA256 = {
    42: "5f2364a42814e40f1f14237db61296176ae63976db43be12c7d834ffb0292ee9",
    123: "aaf8cf8b5a70301f74043438c04f76a16bebd2fd8d8b54f8a4c3872aae43155c",
    2026: "a0556b738db8aa191a0501159701e4ff891789b85ad7e1873b0a60c753741891",
}
FORMAL_ROOT = ROOT / "checkpoints/foundation-v28-current/current"
OUTPUT_ROOT = ROOT / "checkpoints/foundation-v33-context-gate"


class DurationAwareGpuMonitor:
    """Lightweight nvidia-smi sampler with an explicit sustained-heat duration."""

    def __init__(self, interval_seconds: float = 0.25, sustained_seconds: float = 5.0):
        self.interval_seconds = interval_seconds
        self.sustained_seconds = sustained_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, float, float, float, float]] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        command = (
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw "
            "--format=csv,noheader,nounits"
        )
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                values = tuple(
                    float(value.strip())
                    for value in completed.stdout.splitlines()[0].split(",")
                )
                if len(values) == 4:
                    self._samples.append((time.monotonic(), *values))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self._samples:
            return {"samples": 0, "thermal_attention": False}
        times = [sample[0] for sample in self._samples]
        utilization = [sample[1] for sample in self._samples]
        memory = [sample[2] for sample in self._samples]
        temperature = [sample[3] for sample in self._samples]
        power = [sample[4] for sample in self._samples]
        longest = 0.0
        run_start: float | None = None
        previous: float | None = None
        maximum_gap = self.interval_seconds * 4
        for stamp, value in zip(times, temperature):
            if value > 80:
                if run_start is None or (previous is not None and stamp - previous > maximum_gap):
                    run_start = stamp
                longest = max(longest, stamp - run_start + self.interval_seconds)
                previous = stamp
            else:
                run_start = None
                previous = None
        return {
            "samples": len(self._samples),
            "gpu_utilization_percent_mean": float(np.mean(utilization)),
            "gpu_utilization_percent_max": max(utilization),
            "gpu_memory_used_mib_max": max(memory),
            "gpu_temperature_c_max": max(temperature),
            "gpu_power_w_max": max(power),
            "longest_above_80_seconds": longest,
            "sustained_threshold_seconds": self.sustained_seconds,
            "thermal_attention": longest >= self.sustained_seconds,
        }


def formal_checkpoint(seed: int) -> Path:
    return FORMAL_ROOT / f"seed-{seed}/checkpoint-tokens-{FORMAL_START_TOKENS}.pt"


def gate_checkpoint(gate: int, seed: int) -> Path:
    tokens = FORMAL_START_TOKENS + gate * GATE_TOKENS
    return OUTPUT_ROOT / f"gate-{gate}/seed-{seed}/checkpoint-tokens-{tokens}.pt"


def source_checkpoint(gate: int, seed: int) -> Path:
    return formal_checkpoint(seed) if gate == 1 else gate_checkpoint(gate - 1, seed)


def _scheduler_state(source_state: dict, update: int) -> dict:
    state = dict(source_state)
    state["global_step"] = int(update)
    return state


def verify_payload(payload: dict, *, seed: int, update: int, tokens: int) -> dict:
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
    checks = {
        "strict_identity": int(payload.get("seed", -1)) == seed,
        "optimizer": bool(payload.get("optimizer_state")),
        "scheduler": payload.get("scheduler_state", {}).get("global_step") == update,
        "cpu_rng": "torch_cpu" in payload.get("random_state", {}),
        "cuda_rng": "torch_cuda" in payload.get("random_state", {}),
        "all_rng": required_rng.issubset(payload.get("random_state", {})),
        "sampler": torch.is_tensor(payload.get("permutation")),
        "processed_tokens": int(payload.get("tokens_processed", -1)) == tokens,
        "precision_fp32": payload.get("precision_mode") == "fp32",
        "phase": payload.get("phase") == 44,
        "eos_weight": payload.get("eos_loss_weight") == EOS_WEIGHT,
        "repetition_auxiliary_off": payload.get("repetition_auxiliary") is False,
        "phase43_pilot_not_promoted": payload.get("phase43_experimental_promoted") is False,
    }
    return {"checks": checks, "pass": all(checks.values())}


def train_gate(gate: int, seed: int) -> dict:
    if gate not in (1, 2) or seed not in SEEDS:
        raise ValueError("unsupported gate or seed")
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE 44 requires CUDA")
    source = source_checkpoint(gate, seed)
    source_hash = file_sha256(source)
    if gate == 1 and source_hash != EXPECTED_FORMAL_SHA256[seed]:
        raise RuntimeError(f"formal checkpoint SHA256 mismatch for seed {seed}")

    device = torch.device("cuda")
    payload, model, optimizer = load(source, device)
    if model.parameter_count() != EXPECTED_PARAMETERS:
        raise RuntimeError("architecture parameter count changed")
    tokenizer = FoundationTokenizer.load(ROOT / "tokenizer/foundation-v11-base-4096.json")
    train = np.memmap(
        ROOT / "data/foundation_v11/packed/vocab-4096/train.bin",
        dtype=np.uint16,
        mode="r",
    )
    start_update = int(payload["update"])
    end_update = start_update + GATE_TOKENS // TOKENS_PER_UPDATE
    end_tokens = int(payload["tokens_processed"]) + GATE_TOKENS
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    monitor = DurationAwareGpuMonitor()
    monitor.start()
    losses: list[float] = []
    eos_losses: list[float] = []
    non_eos_losses: list[float] = []
    gradient_norms: list[float] = []
    started = time.perf_counter()
    finite = True
    try:
        for update in range(start_update + 1, end_update + 1):
            inputs, targets = macro_batch(
                train, int(payload["permutation"][update - 1]), TOKENS_PER_UPDATE
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss, eos_loss, non_eos_loss = weighted_lm_loss(
                logits, targets, tokenizer.eos_id, EOS_WEIGHT
            )
            if not torch.isfinite(loss):
                finite = False
                raise RuntimeError("non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                finite = False
                raise RuntimeError("non-finite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            eos_losses.append(float(eos_loss.detach()))
            non_eos_losses.append(float(non_eos_loss.detach()))
            gradient_norms.append(float(norm))
        torch.cuda.synchronize(device)
    finally:
        elapsed = time.perf_counter() - started
        telemetry = monitor.stop()

    target = gate_checkpoint(gate, seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.tmp")
    saved = {
        **payload,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": _scheduler_state(payload["scheduler_state"], end_update),
        "random_state": random_state(device),
        "update": end_update,
        "tokens_processed": end_tokens,
        "phase": 44,
        "formal_research": True,
        "context_gate": gate,
        "eos_loss_weight": EOS_WEIGHT,
        "repetition_auxiliary": False,
        "phase43_experimental_promoted": False,
        "source_checkpoint": str(source.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": source_hash,
        "precision_mode": "fp32",
    }
    torch.save(saved, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    strict_model = DiagnosticTransformerV17(DiagnosticConfigV17(**loaded["config"]))
    strict_model.load_state_dict(loaded["model_state"], strict=True)
    strict_optimizer = create_optimizer(strict_model, 1e-4, 0.1)
    strict_optimizer.load_state_dict(loaded["optimizer_state"])
    integrity = verify_payload(
        loaded, seed=seed, update=end_update, tokens=end_tokens
    )
    integrity.update({"strict_model_reload": True, "strict_optimizer_reload": True})
    integrity["source_unchanged"] = file_sha256(source) == source_hash
    integrity["pass"] = integrity["pass"] and integrity["source_unchanged"]
    if not integrity["pass"]:
        raise RuntimeError("gate checkpoint integrity failed")
    temporary.replace(target)
    result = {
        "gate": gate,
        "seed": seed,
        "start_tokens": int(payload["tokens_processed"]),
        "end_tokens": end_tokens,
        "budget_tokens": GATE_TOKENS,
        "source_sha256": source_hash,
        "checkpoint": {
            "path": str(target.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_sha256(target),
            "integrity": integrity,
        },
        "training": {
            "device": torch.cuda.get_device_name(device),
            "precision": "fp32",
            "amp": False,
            "seconds": elapsed,
            "tokens_per_second": GATE_TOKENS / elapsed,
            "mean_weighted_loss": float(np.mean(losses)),
            "mean_eos_loss": float(np.mean(eos_losses)),
            "mean_non_eos_loss": float(np.mean(non_eos_losses)),
            "mean_gradient_norm": float(np.mean(gradient_norms)),
            "max_gradient_norm": float(np.max(gradient_norms)),
            "all_finite": finite,
            "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1_048_576,
            "telemetry": telemetry,
        },
        "parallel_cpu_evaluation": "DISABLED",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=int, choices=(1, 2), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    if tuple(args.seeds) != SEEDS:
        raise RuntimeError("formal PHASE 44 gates require seeds 42, 123, and 2026")
    results = []
    destination = ROOT / f"evaluation/foundation-v33-gate{args.gate}-training.json"
    for seed in args.seeds:
        result = train_gate(args.gate, seed)
        results.append(result)
        destination.write_text(
            json.dumps(
                {
                    "schema": "foundation-v33-gate-training-v1",
                    "phase": 44,
                    "gate": args.gate,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
