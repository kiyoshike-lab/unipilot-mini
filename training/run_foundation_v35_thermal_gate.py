"""PHASE 46 formal CUDA FP32 continuation with cooldown-only execution control."""
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
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.foundation_v31_objective import weighted_lm_loss
from training.optimizer import create_optimizer
from training.checkpoint_paths import checkpoint_path as routed_checkpoint_path, display_path, ensure_checkpoint_storage, existing_checkpoint_path
from training.run_foundation_v30_eos_experiment import load
from training.train_foundation_v15_controlled import macro_batch
from training.train_foundation_v21_ab import file_sha256, random_state


SEEDS = (42, 123, 2026)
TOKENS_PER_UPDATE = 512
GATE_TOKENS = 256_000
START_TOKENS = 15_872_000
EOS_WEIGHT = 1.5
EXPECTED_PARAMETERS = 19_514_880
SOURCE_PARTS = ("foundation-v33-context-gate", "gate-2")
OUTPUT_PARTS = ("foundation-v35-thermal-short-gate",)
COOLDOWN_TARGET_C = 60.0
COOLDOWN_TIMEOUT_SECONDS = 300.0


def query_gpu() -> dict:
    command = (
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw,"
        "clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown "
        "--format=csv,noheader,nounits"
    )
    completed = subprocess.run(
        command, shell=True, check=True, capture_output=True, text=True, timeout=5
    )
    values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
    return {
        "gpu_utilization_percent": float(values[0]),
        "gpu_memory_used_mib": float(values[1]),
        "gpu_temperature_c": float(values[2]),
        "sm_clock_mhz": float(values[3]),
        "power_w": float(values[4]),
        "software_thermal_slowdown": values[5] == "Active",
        "hardware_thermal_slowdown": values[6] == "Active",
    }


def cooldown() -> dict:
    started = time.monotonic()
    records = []
    while True:
        sample = query_gpu()
        sample["elapsed_seconds"] = time.monotonic() - started
        records.append(sample)
        if sample["gpu_temperature_c"] <= COOLDOWN_TARGET_C:
            return {
                "target_c": COOLDOWN_TARGET_C,
                "start": records[0],
                "end": sample,
                "waited_seconds": sample["elapsed_seconds"],
                "target_reached": True,
                "samples": records,
            }
        if sample["elapsed_seconds"] >= COOLDOWN_TIMEOUT_SECONDS:
            return {
                "target_c": COOLDOWN_TARGET_C,
                "start": records[0],
                "end": sample,
                "waited_seconds": sample["elapsed_seconds"],
                "target_reached": False,
                "samples": records,
            }
        time.sleep(5)


class Monitor:
    def __init__(self, interval_seconds: float = 0.25):
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.samples: list[dict] = []
        self.started = 0.0
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                sample = query_gpu()
                sample["elapsed_seconds"] = time.monotonic() - self.started
                self.samples.append(sample)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.stop_event.wait(self.interval_seconds)

    def finish(self) -> dict:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        if not self.samples:
            return {"samples": 0, "thermal_classification": "NO_TELEMETRY"}
        temperature = [row["gpu_temperature_c"] for row in self.samples]
        clock = [row["sm_clock_mhz"] for row in self.samples]
        power = [row["power_w"] for row in self.samples]
        utilization = [row["gpu_utilization_percent"] for row in self.samples]
        vram = [row["gpu_memory_used_mib"] for row in self.samples]
        software = any(row["software_thermal_slowdown"] for row in self.samples)
        hardware = any(row["hardware_thermal_slowdown"] for row in self.samples)
        if hardware:
            classification = "HARDWARE_THERMAL_THROTTLING"
        elif software:
            classification = "SOFTWARE_THERMAL_SLOWDOWN"
        elif max(temperature) >= 80:
            classification = "HOT_BUT_STABLE"
        else:
            classification = "NO_THERMAL_CONCERN"
        midpoint = self.samples[len(self.samples) // 2]
        maximum = max(self.samples, key=lambda row: row["gpu_temperature_c"])
        return {
            "samples": len(self.samples),
            "start": self.samples[0],
            "midpoint": midpoint,
            "end": self.samples[-1],
            "maximum_temperature_sample": maximum,
            "gpu_utilization_percent_mean": float(np.mean(utilization)),
            "gpu_utilization_percent_max": max(utilization),
            "gpu_memory_used_mib_max": max(vram),
            "gpu_temperature_c_max": max(temperature),
            "sm_clock_mhz_min": min(clock),
            "sm_clock_mhz_max": max(clock),
            "gpu_power_w_max": max(power),
            "software_thermal_slowdown": software,
            "hardware_thermal_slowdown": hardware,
            "thermal_classification": classification,
            "thermal_attention": max(temperature) >= 83 and (software or min(clock) < max(clock) * 0.95),
        }


def checkpoint(gate: int, seed: int) -> Path:
    tokens = START_TOKENS + gate * GATE_TOKENS
    return routed_checkpoint_path(ROOT, *OUTPUT_PARTS, f"gate-{gate}", f"seed-{seed}", f"checkpoint-tokens-{tokens}.pt")


def source_checkpoint(gate: int, seed: int) -> Path:
    if gate == 1:
        return existing_checkpoint_path(ROOT, *SOURCE_PARTS, f"seed-{seed}", "checkpoint-tokens-15872000.pt")
    return checkpoint(1, seed)


def resume_checks(payload: dict, seed: int, expected_tokens: int, gate: int) -> dict:
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
    expected_phase = 44 if gate == 1 else 46
    expected_gate = 2 if gate == 1 else 1
    checks = {
        "seed": int(payload.get("seed", -1)) == seed,
        "model_state": bool(payload.get("model_state")),
        "optimizer_state": bool(payload.get("optimizer_state")),
        "scheduler_state": bool(payload.get("scheduler_state")),
        "learning_rate": all(
            group.get("lr") == 1e-4 for group in payload.get("optimizer_state", {}).get("param_groups", [])
        ),
        "processed_tokens": int(payload.get("tokens_processed", -1)) == expected_tokens,
        "sampler": torch.is_tensor(payload.get("permutation")),
        "rng": required_rng.issubset(payload.get("random_state", {})),
        "precision_fp32": payload.get("precision_mode") == "fp32",
        "eos_weight": payload.get("eos_loss_weight") == EOS_WEIGHT,
        "repetition_auxiliary_off": payload.get("repetition_auxiliary") is False,
        "source_phase": payload.get("phase") == expected_phase,
        "source_gate": (
            payload.get("context_gate") if gate == 1 else payload.get("v35_gate")
        ) == expected_gate,
    }
    return {"checks": checks, "pass": all(checks.values())}


def saved_checks(payload: dict, seed: int, update: int, tokens: int, gate: int) -> dict:
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
    checks = {
        "seed": int(payload.get("seed", -1)) == seed,
        "optimizer_state": bool(payload.get("optimizer_state")),
        "scheduler_state": payload.get("scheduler_state", {}).get("global_step") == update,
        "learning_rate": all(
            group.get("lr") == 1e-4 for group in payload.get("optimizer_state", {}).get("param_groups", [])
        ),
        "processed_tokens": int(payload.get("tokens_processed", -1)) == tokens,
        "sampler": torch.is_tensor(payload.get("permutation")),
        "rng": required_rng.issubset(payload.get("random_state", {})),
        "precision_fp32": payload.get("precision_mode") == "fp32",
        "phase": payload.get("phase") == 46,
        "gate": payload.get("v35_gate") == gate,
        "eos_weight": payload.get("eos_loss_weight") == EOS_WEIGHT,
        "repetition_auxiliary_off": payload.get("repetition_auxiliary") is False,
    }
    return {"checks": checks, "pass": all(checks.values())}


def train_gate(gate: int, seed: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE 46 requires CUDA")
    source = source_checkpoint(gate, seed)
    source_hash = file_sha256(source)
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    resume = resume_checks(source_payload, seed, START_TOKENS + (gate - 1) * GATE_TOKENS, gate)
    if not resume["pass"]:
        raise RuntimeError(f"resume preflight failed for seed {seed}: {resume}")
    thermal_cooldown = cooldown()
    device = torch.device("cuda")
    payload, model, optimizer = load(source, device)
    if model.parameter_count() != EXPECTED_PARAMETERS:
        raise RuntimeError("architecture parameter count changed")
    tokenizer = FoundationTokenizer.load(ROOT / "tokenizer/foundation-v11-base-4096.json")
    train = np.memmap(ROOT / "data/foundation_v11/packed/vocab-4096/train.bin", dtype=np.uint16, mode="r")
    start_update = int(payload["update"])
    end_update = start_update + GATE_TOKENS // TOKENS_PER_UPDATE
    end_tokens = int(payload["tokens_processed"]) + GATE_TOKENS
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    monitor = Monitor()
    monitor.start()
    losses, eos_losses, non_eos_losses, norms = [], [], [], []
    finite = True
    started = time.perf_counter()
    try:
        for update in range(start_update + 1, end_update + 1):
            inputs, targets = macro_batch(train, int(payload["permutation"][update - 1]), TOKENS_PER_UPDATE)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss, eos_loss, non_eos_loss = weighted_lm_loss(logits, targets, tokenizer.eos_id, EOS_WEIGHT)
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
            norms.append(float(norm))
        torch.cuda.synchronize(device)
    finally:
        elapsed = time.perf_counter() - started
        telemetry = monitor.finish()
    target = checkpoint(gate, seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    ensure_checkpoint_storage(target)
    temporary = target.with_suffix(".pt.tmp")
    saved = {
        **payload,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": {**payload["scheduler_state"], "global_step": end_update},
        "random_state": random_state(device),
        "update": end_update,
        "tokens_processed": end_tokens,
        "phase": 46,
        "formal_research": True,
        "v35_gate": gate,
        "eos_loss_weight": EOS_WEIGHT,
        "repetition_auxiliary": False,
        "source_checkpoint": display_path(ROOT, source),
        "source_sha256": source_hash,
        "precision_mode": "fp32",
    }
    torch.save(saved, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    strict_model = DiagnosticTransformerV17(DiagnosticConfigV17(**loaded["config"]))
    strict_model.load_state_dict(loaded["model_state"], strict=True)
    strict_optimizer = create_optimizer(strict_model, 1e-4, 0.1)
    strict_optimizer.load_state_dict(loaded["optimizer_state"])
    integrity = saved_checks(loaded, seed, end_update, end_tokens, gate)
    integrity.update({"strict_model_reload": True, "strict_optimizer_reload": True})
    integrity["source_unchanged"] = file_sha256(source) == source_hash
    integrity["pass"] = integrity["pass"] and integrity["source_unchanged"]
    if not integrity["pass"]:
        raise RuntimeError("checkpoint integrity failed")
    temporary.replace(target)
    return {
        "gate": gate, "seed": seed, "start_tokens": int(payload["tokens_processed"]),
        "end_tokens": end_tokens, "budget_tokens": GATE_TOKENS, "source_sha256": source_hash,
        "resume_preflight": resume, "cooldown": thermal_cooldown,
        "checkpoint": {"path": display_path(ROOT, target), "sha256": file_sha256(target), "integrity": integrity},
        "training": {"device": torch.cuda.get_device_name(device), "precision": "fp32", "amp": False,
            "seconds": elapsed, "tokens_per_second": GATE_TOKENS / elapsed,
            "mean_weighted_loss": float(np.mean(losses)), "mean_eos_loss": float(np.mean(eos_losses)),
            "mean_non_eos_loss": float(np.mean(non_eos_losses)), "mean_gradient_norm": float(np.mean(norms)),
            "max_gradient_norm": float(np.max(norms)), "all_finite": finite,
            "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1_048_576, "telemetry": telemetry},
        "parallel_cpu_evaluation": "DISABLED", "settings_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    results = []
    destination = ROOT / f"evaluation/foundation-v35-gate{args.gate}-training.json"
    for seed in SEEDS:
        result = train_gate(args.gate, seed)
        results.append(result)
        destination.write_text(json.dumps({"schema": "foundation-v35-gate-training-v1", "phase": 46, "gate": args.gate, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
