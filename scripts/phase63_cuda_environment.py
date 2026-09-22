"""PHASE63A non-training CUDA recovery receipt.

This intentionally has no optimizer, no generation, no checkpoint save, and
no worker launch.  A compute-capable C+G row is a manual-review gate, not an
implicit display allowlist.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.gpu_execution_lock import atomic_json, identity, inventory
from training.phase63_worker import PARENT_SHA, parent, sha

OUT = ROOT / "evaluation" / "phase63" / "cuda-environment-recovery.json"


def new(path: Path) -> None:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(f"environment receipt collision: {path}")


def nvidia() -> dict:
    command = ["nvidia-smi", "--query-gpu=name,driver_version,cuda_version,memory.total", "--format=csv,noheader"]
    # Newer nvidia-smi may not expose cuda_version in query mode; XML remains
    # the authoritative fallback and avoids inferring toolkit installation.
    result = subprocess.run(command, capture_output=True, text=True)
    xml = subprocess.run(["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True).stdout
    rows = []
    try:
        rows = [{**row, "identity": identity(row["pid"])} for row in inventory()]
    except Exception as exc:
        rows = [{"inventory_error": repr(exc)}]
    return {"query_returncode": result.returncode, "query_stdout": result.stdout.strip(),
            "xml_driver": xml.split("<driver_version>", 1)[1].split("</driver_version>", 1)[0],
            "xml_cuda_compatibility": xml.split("<cuda_version>", 1)[1].split("</cuda_version>", 1)[0],
            "processes": rows}


def package(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    new(OUT)
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA_ENVIRONMENT_NOT_READY")
    if torch.cuda.device_count() < 1 or "RTX 2070 SUPER" not in torch.cuda.get_device_name(0):
        raise RuntimeError("CUDA_DEVICE_CONTRACT_FAIL")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    try:
        left = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        product = left @ left.t(); torch.cuda.synchronize(device)
        tensor_test = {"pass": bool(torch.isfinite(product).all()), "dtype": str(product.dtype),
                       "sum": float(product.cpu().sum())}
        before = torch.cuda.get_rng_state(device)
        torch.cuda.set_rng_state(before, device)
        after = torch.cuda.get_rng_state(device)
        rng = {"pass": bool(torch.equal(before, after)), "bytes": int(before.numel())}
        properties = torch.cuda.get_device_properties(device)
        memory = {"total_bytes": int(properties.total_memory), "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                  "reserved_bytes": int(torch.cuda.memory_reserved(device))}
        source = parent(42)
        if sha(source) != PARENT_SHA[42]:
            raise RuntimeError("PHASE48_PARENT_SHA_MISMATCH")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
        model.load_state_dict(payload["model_state"], strict=True)
        model.to(device)
        parent_load = {"pass": model.parameter_count() == 19_514_880 and all(p.dtype == torch.float32 and bool(torch.isfinite(p).all()) for p in model.parameters()),
                       "sha256": sha(source), "parameter_count": model.parameter_count(), "fp32": True,
                       "checkpoint": str(source)}
    finally:
        locals().pop("model", None); locals().pop("product", None); locals().pop("left", None)
        torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    gpu = nvidia()
    # This receipt process owns the deliberate tiny validation context.  It is
    # recorded but not considered a foreign process; all other C+G rows block.
    c_plus_g = [row for row in gpu["processes"] if row.get("type") == "C+G" and row.get("pid") != os.getpid()]
    result = {"phase": "63A", "kind": "CUDA_ENVIRONMENT_RECOVERY", "optimizer_steps": 0, "training": False,
              "generation_evaluation": False, "checkpoints_created": 0, "python_executable": str(Path(sys.executable).resolve()),
              "python_version": sys.version, "pip_version": package("pip"), "packages": {"torch": torch.__version__, "torchvision": package("torchvision"), "torchaudio": package("torchaudio")},
              "torch_file": torch.__file__, "torch_cuda": torch.version.cuda, "cuda_available": True,
              "device_count": torch.cuda.device_count(), "gpu_name": torch.cuda.get_device_name(device),
              "compute_capability": list(torch.cuda.get_device_capability(device)), "tensor_test": tensor_test,
              "rng_roundtrip": rng, "fp32_contract": {"pass": tensor_test["dtype"] == "torch.float32" and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32, "amp": False, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32},
              "memory": memory, "project_imports": {"model": "PASS", "checkpoint_loader": "PASS", "cuda_device": "PASS"},
              "parent_checkpoint_cuda_strict_load": parent_load, "nvidia": gpu,
              "c_plus_g_inventory_status": "CUDA_PROCESS_PREFLIGHT_REQUIRES_MANUAL_REVIEW" if c_plus_g else "NO_COMPUTE_CAPABLE_ROWS",
              "worker_no_train_spawn": "NOT_RUN_C+G_MANUAL_REVIEW" if c_plus_g else "PENDING_EXCLUSIVE_SUPERVISOR_TEST",
              "gpu_lock_lifecycle": "NOT_RUN_C+G_MANUAL_REVIEW" if c_plus_g else "PENDING_EXCLUSIVE_SUPERVISOR_TEST",
              "supervisor_interpreter_binding": {"supervisor": str(Path(sys.executable).resolve()), "worker_contract": "sys.executable; execution binding enforced"},
              "final_gate": "PHASE63_CUDA_ENV_REQUIRES_MANUAL_REVIEW" if c_plus_g else "PHASE63_CUDA_ENV_READY"}
    atomic_json(OUT, result)
    print(result["final_gate"], flush=True)


if __name__ == "__main__":
    main()
