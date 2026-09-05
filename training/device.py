"""Device, optimizer-state, and lightweight NVIDIA telemetry helpers."""
from __future__ import annotations

import subprocess
import threading
from typing import Any

import torch


def resolve_device(requested: str | torch.device | None = "auto") -> torch.device:
    value = "auto" if requested is None else str(requested)
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {value}")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(value)


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _move_value(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_value(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, device) for item in value)
    return value


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: str | torch.device
) -> None:
    target = torch.device(device)
    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = _move_value(state, target)


def optimizer_state_devices(optimizer: torch.optim.Optimizer) -> list[str]:
    devices: set[str] = set()

    def visit(value: Any) -> None:
        if torch.is_tensor(value):
            devices.add(str(value.device))
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(optimizer.state)
    return sorted(devices)


def describe_device(device: str | torch.device) -> dict:
    target = torch.device(device)
    result = {
        "device": str(target),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "precision_mode": "fp32",
    }
    if target.type == "cuda":
        properties = torch.cuda.get_device_properties(target)
        result.update({
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(target)),
            "total_memory_bytes": int(properties.total_memory),
        })
    return result


class NvidiaSmiMonitor:
    """Sample GPU load without retaining unbounded per-step telemetry."""

    def __init__(self, enabled: bool, interval_seconds: float = 0.25):
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, float, float, float]] = []

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        query = (
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw "
            "--format=csv,noheader,nounits"
        )
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    query,
                    capture_output=True,
                    check=True,
                    shell=True,
                    text=True,
                    timeout=5,
                )
                values = tuple(float(value.strip()) for value in completed.stdout.splitlines()[0].split(","))
                if len(values) == 4:
                    with self._lock:
                        self._samples.append(values)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval_seconds)

    def snapshot(self, *, reset: bool = False) -> dict | None:
        if not self.enabled:
            return None
        with self._lock:
            samples = list(self._samples)
            if reset:
                self._samples.clear()
        if not samples:
            return {"samples": 0}
        columns = list(zip(*samples))
        return {
            "samples": len(samples),
            "gpu_utilization_percent_mean": sum(columns[0]) / len(samples),
            "gpu_utilization_percent_max": max(columns[0]),
            "gpu_memory_used_mib_max": max(columns[1]),
            "gpu_temperature_c_max": max(columns[2]),
            "gpu_power_w_max": max(columns[3]),
        }

    def stop(self) -> dict | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
        return self.snapshot()

    def __enter__(self) -> "NvidiaSmiMonitor":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()
