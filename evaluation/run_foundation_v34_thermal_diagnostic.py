"""PHASE 45 read-only representative GPU load with one-second telemetry."""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v15_controlled import macro_batch
from training.train_foundation_v21_ab import file_sha256


QUERY_FIELDS = (
    "name",
    "temperature.gpu",
    "utilization.gpu",
    "clocks.sm",
    "clocks.mem",
    "power.draw",
    "clocks_throttle_reasons.active",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.sw_power_cap",
)


def query_gpu(elapsed: float) -> dict:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    values = next(csv.reader(io.StringIO(completed.stdout.strip())))
    row = {key: value.strip() for key, value in zip(QUERY_FIELDS, values)}
    for key in ("temperature.gpu", "utilization.gpu", "clocks.sm", "clocks.mem", "power.draw"):
        row[key] = float(row[key])
    row["elapsed_seconds"] = elapsed
    return row


def active(value: str) -> bool:
    return value.lower() not in {"not active", "0x0000000000000000", "n/a", "[n/a]"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=45.0)
    args = parser.parse_args()
    if args.seconds < 10 or args.seconds > 60:
        raise ValueError("diagnostic duration must be between 10 and 60 seconds")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    path = ROOT / "checkpoints/foundation-v33-context-gate/gate-2/seed-42/checkpoint-tokens-15872000.pt"
    before = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.train()
    train = np.memmap(
        ROOT / "data/foundation_v11/packed/vocab-4096/train.bin", dtype=np.uint16, mode="r"
    )
    x, y = macro_batch(train, int(payload["permutation"][0]), 512)
    x, y = x.to(device), y.to(device)
    samples = [query_gpu(0.0)]
    started = time.perf_counter()
    next_sample = 1.0
    updates = 0
    while time.perf_counter() - started < args.seconds:
        model.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.cuda.synchronize()
        updates += 1
        elapsed = time.perf_counter() - started
        if elapsed >= next_sample:
            samples.append(query_gpu(elapsed))
            next_sample += 1.0
    elapsed = time.perf_counter() - started
    samples.append(query_gpu(elapsed))
    thermal_keys = (
        "clocks_throttle_reasons.sw_thermal_slowdown",
        "clocks_throttle_reasons.hw_thermal_slowdown",
        "clocks_throttle_reasons.hw_slowdown",
    )
    thermal_throttling = any(active(row[key]) for row in samples for key in thermal_keys)
    max_temperature = max(row["temperature.gpu"] for row in samples)
    classification = (
        "THERMAL_THROTTLING_OBSERVED"
        if thermal_throttling
        else "HOT_BUT_STABLE"
        if max_temperature >= 80
        else "NO_THERMAL_CONCERN"
    )
    result = {
        "schema": "foundation-v34-thermal-diagnostic-v1",
        "phase": 45,
        "diagnostic_only": True,
        "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": file_sha256(path),
        "checkpoint_unchanged": file_sha256(path) == before,
        "device": torch.cuda.get_device_name(0),
        "precision": "fp32",
        "optimizer_steps": 0,
        "settings_changed": False,
        "duration_seconds": elapsed,
        "forward_backward_iterations": updates,
        "tokens_per_second_equivalent": updates * 512 / elapsed,
        "query_interval_seconds": 1.0,
        "samples": samples,
        "summary": {
            "classification": classification,
            "max_temperature_c": max_temperature,
            "mean_gpu_utilization_percent": float(
                np.mean([row["utilization.gpu"] for row in samples])
            ),
            "min_sm_clock_mhz": min(row["clocks.sm"] for row in samples),
            "max_sm_clock_mhz": max(row["clocks.sm"] for row in samples),
            "memory_clock_mhz": max(row["clocks.mem"] for row in samples),
            "max_power_w": max(row["power.draw"] for row in samples),
            "thermal_throttling_observed": thermal_throttling,
            "power_cap_observed": any(
                active(row["clocks_throttle_reasons.sw_power_cap"]) for row in samples
            ),
        },
    }
    target = ROOT / "evaluation/foundation-v34-thermal.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
