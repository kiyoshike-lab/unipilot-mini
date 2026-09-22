"""PHASE63 single-study supervisor.

The supervisor is the only PHASE63 entry point.  It delegates every CUDA child
to ``exclusive_cuda_runner`` and treats an absent COMPLETE/RELEASED receipt as
an incomplete study, never as permission to launch another process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.checkpoint_paths import checkpoint_root, existing_checkpoint_path
from training.exclusive_cuda_runner import execute, is_complete
from training.gpu_execution_lock import ExecutionBlocked, atomic_json, identity, inventory, sha
from training.phase63_worker import ARMS, CACHE, CACHE_SHA, PARENT_SHA, PERM_SHA, SEEDS, output, parent, raw_output
from training.run_foundation_v35_thermal_gate import query_gpu
from training.run_foundation_v36_lr_review import fingerprint, verify_payload

OUT = ROOT / "evaluation" / "phase63"
SPEC = ROOT / "evaluation" / "phase62" / "phase63-clean-continuation-stability-preregistration.json"
SPEC_SHA = "856ed78873b469100f4c6c34df1c4d05c19b540969453fba8e41315a478bc48f"
AUTHORIZATION_START = "44d4caf132094ac1ba6567ebd3b8d963a4b2cfd8"
ZROOT = Path(r"Z:\AI\unipilot-mini\checkpoints")
RUNTIME = ZROOT.parent / "runtime"
RUN_ORDER = ((42, "control"), (42, "half-lr"), (123, "control"), (123, "half-lr"), (2026, "control"), (2026, "half-lr"))


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def new(path: Path) -> None:
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(f"output collision or partial: {path}")


def emit(path: Path, value: dict) -> None:
    new(path); atomic_json(path, value)


def source_manifest() -> dict:
    paths = ("training/phase63_worker.py", "scripts/run_phase63_study.py", "training/exclusive_cuda_runner.py", "training/gpu_execution_lock.py")
    return {path: sha(ROOT / path) for path in paths}


def interpreter_metadata() -> dict:
    """Bind the child interpreter; PATH resolution is not an execution contract."""
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0]
    return {"executable": str(Path(sys.executable).resolve()), "python_version": sys.version,
            "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()), "gpu_name": torch.cuda.get_device_name(0),
            "driver_version": driver,
            "validator": "scripts/run_phase63_study.py::verify",
            "validator_sha256": sha(ROOT / "scripts/run_phase63_study.py")}


def output_artifact(seed: int, arm: str) -> Path:
    return OUT / f"run-seed{seed}-{arm}.json"


def receipt_path(seed: int, arm: str, kind: str) -> Path:
    return RUNTIME / "runs" / f"phase63-{kind}-seed{seed}-{arm}"


def root_gate() -> None:
    if os.environ.get("UNIPILOT_CHECKPOINT_ROOT") != str(ZROOT) or checkpoint_root(ROOT) != ZROOT:
        raise RuntimeError("PROCESS_ENV_RESOLVER_MISMATCH")


def cpu_parent(seed: int) -> dict:
    source = parent(seed)
    if source != ZROOT / "experimental" / "phase48" / "arm-C" / f"seed-{seed}" / "checkpoint-tokens-16384000.pt":
        raise RuntimeError("PARENT_PATH_MISMATCH")
    if sha(source) != PARENT_SHA[seed]:
        raise RuntimeError("PARENT_SHA_MISMATCH")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    integrity = verify_payload(payload, seed, 16_384_000, 5e-5)
    if not integrity["pass"] or fingerprint(payload["permutation"][32000:32122]) != PERM_SHA[seed]:
        raise RuntimeError("PARENT_INTEGRITY_FAIL")
    return {"seed": seed, "path": str(source), "sha256": sha(source), "bytes": source.stat().st_size,
            "strict_cpu_integrity": integrity, "next_122_permutation_sha256": PERM_SHA[seed]}


def cuda_inventory_with_identity() -> dict:
    rows = inventory()
    detailed = [{**row, "identity": identity(row["pid"])} for row in rows]
    # C+G is deliberately *not* treated as a graphics allowlist.  Without a
    # pure-G classification it remains compute-capable and blocks the study.
    blocking = [row for row in detailed if row["type"] != "G"]
    return {"rows": detailed, "blocking_rows": blocking, "safe_for_new_cuda_owner": not blocking}


def temperature() -> dict:
    row = query_gpu()
    return {"temperature_c": row["gpu_temperature_c"],
            "hardware_thermal_slowdown": bool(row["hardware_thermal_slowdown"]), "raw": row}


def preflight() -> None:
    root_gate()
    allowed_environment_receipt = OUT / "cuda-environment-recovery.json"
    existing_local = set(OUT.iterdir()) if OUT.exists() else set()
    if existing_local - {allowed_environment_receipt} or (ZROOT.parent / "evaluation" / "phase63").exists():
        raise RuntimeError("PHASE63_ARTIFACT_OR_PARTIAL_EXISTS")
    if sha(SPEC) != SPEC_SHA:
        raise RuntimeError("PREREGISTRATION_INTEGRITY_FAIL")
    spec = read(SPEC)
    if spec["training_authorized"] is not False or spec["training_executed"] is not False or spec["status"] != "REGISTERED_REQUIRES_NEW_USER_TRAINING_AUTHORIZATION":
        raise RuntimeError("PREREGISTRATION_SCOPE_MISMATCH")
    for relative, digest in spec["source_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError("REGISTERED_SCIENTIFIC_SOURCE_CHANGED:" + relative)
    if git("branch", "--show-current") != "foundation-research" or git("merge-base", "--is-ancestor", AUTHORIZATION_START, "HEAD") != "":
        raise RuntimeError("BRANCH_OR_AUTHORIZATION_BASE_MISMATCH")
    if git("rev-parse", "origin/foundation-research") != AUTHORIZATION_START:
        raise RuntimeError("ORIGIN_AUTHORIZATION_BASE_MISMATCH")
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 2070 SUPER":
        raise RuntimeError("CUDA_RTX2070_SUPER_REQUIRED")
    if sha(CACHE) != CACHE_SHA or len(read(CACHE)["episodes"]) != 128:
        raise RuntimeError("MATCHING_CACHE_INTEGRITY_FAIL")
    parents = [cpu_parent(seed) for seed in SEEDS]
    data = ROOT / spec["data"]["path"]
    if sha(data) != spec["data"]["sha256"]:
        raise RuntimeError("TRAINING_DATA_SHA_MISMATCH")
    largest = max(row["bytes"] for row in parents); reserve = 2 * largest + 2 * 1024**3
    free = shutil.disk_usage(ZROOT).free
    if free < reserve:
        raise RuntimeError("INSUFFICIENT_Z_DISK_FOR_ATOMIC_CHECKPOINTS")
    inv = cuda_inventory_with_identity()
    binding = {"phase": 63, "schema": "phase63-execution-binding-v1", "execution_authorized": True,
               "authorization_record": "explicit user PHASE63 instruction", "authorization_start_head": AUTHORIZATION_START,
               "implementation_head": git("rev-parse", "HEAD"), "branch": "foundation-research", "checkpoint_root": str(ZROOT),
               "interpreter": interpreter_metadata(),
               "preregistration_path": str(SPEC.relative_to(ROOT)), "preregistration_sha256": sha(SPEC),
               "source_manifest": source_manifest(), "run_order": [{"seed": s, "arm": a} for s, a in RUN_ORDER],
               "runs": [{"seed": s, "arm": a, "runtime_lr": ARMS[a], "parent": str(parent(s)), "parent_sha256": PARENT_SHA[s], "output": str(output(s, a)), "raw": str(raw_output(s, a)), "updates": 122} for s, a in RUN_ORDER]}
    pre = {"phase": 63, "gate": "PREFLIGHT_PASS" if inv["safe_for_new_cuda_owner"] else "PREFLIGHT_BLOCKED_FOREIGN_CUDA_PROCESS", "new_training": False,
           "branch": "foundation-research", "head": git("rev-parse", "HEAD"), "origin_authorization_base": AUTHORIZATION_START,
           "checkpoint_root": str(ZROOT), "preregistration_sha256": sha(SPEC), "binding_sha256_pending": True,
           "parents": parents, "matching_cache_sha256": sha(CACHE), "data_sha256": sha(data), "z_free_bytes": free,
           "atomic_reserve_required_bytes": reserve, "cuda_inventory": inv, "c_plus_g_policy": "BLOCKED_UNLESS_PURE_G; no display allowlist used"}
    emit(OUT / "execution-binding.json", binding)
    pre["binding_sha256"] = sha(OUT / "execution-binding.json")
    emit(OUT / "preflight.json", pre)
    emit(OUT / "cuda-process-inventory.json", inv)
    print(pre["gate"], flush=True)


def require_preflight() -> dict:
    pre = read(OUT / "preflight.json")
    binding = read(OUT / "execution-binding.json")
    if pre["gate"] != "PREFLIGHT_PASS" or sha(SPEC) != SPEC_SHA or binding["source_manifest"] != source_manifest():
        raise RuntimeError("PHASE63_PRETRAINING_GATE_NOT_PASS")
    if read(OUT / "cuda-process-inventory.json")["safe_for_new_cuda_owner"] is not True:
        raise RuntimeError("FOREIGN_CUDA_PROCESS_PRESENT")
    return binding


def verify(seed: int, arm: str, kind: str) -> dict:
    if kind == "dry-run":
        path = receipt_path(seed, arm, kind) / "stdout.log"
        return {"strict_reload": True, "updates_complete": True, "artifact_path": str(path), "artifact_sha256": sha(path), "kind": kind}
    target = output(seed, arm)
    if not target.exists() or target.with_name(target.name + ".tmp").exists():
        raise RuntimeError("MISSING_OR_PARTIAL_CHECKPOINT")
    payload = torch.load(target, map_location="cpu", weights_only=False)
    integrity = verify_payload(payload, seed, 16_446_464, ARMS[arm])
    checks = {"strict_model_optimizer_reload": integrity["pass"], "scheduler": payload["scheduler_state"]["global_step"] == 32122,
              "sampler": fingerprint(payload["permutation"][32000:32122]) == PERM_SHA[seed],
              "parent": payload.get("parent_checkpoint_sha256") == PARENT_SHA[seed],
              "runtime_lr": all(row["lr"] == ARMS[arm] for row in payload["optimizer_state"]["param_groups"]),
              "markers": payload.get("phase") == 63 and payload.get("EXPERIMENTAL") is True and payload.get("NOT_CANONICAL") is True}
    if not all(checks.values()):
        raise RuntimeError("POSTWORKER_STRICT_VALIDATION_FAIL")
    raw = read(raw_output(seed, arm))
    sanitized = {key: value for key, value in raw.items() if key != "stats"}
    emit(output_artifact(seed, arm), sanitized)
    return {"strict_reload": True, "updates_complete": True, "artifact_path": str(target), "artifact_sha256": sha(target), "resume_integrity": checks}


def run(kind: str) -> None:
    binding = require_preflight()
    prior = None
    for seed, arm in RUN_ORDER:
        run_id = f"phase63-{kind}-seed{seed}-{arm}"
        directory = receipt_path(seed, arm, kind)
        if directory.exists():
            raise RuntimeError("RUN_DIRECTORY_COLLISION_OR_PRIOR_ATTEMPT")
        if kind == "train" and (output(seed, arm).exists() or raw_output(seed, arm).exists()):
            raise RuntimeError("OUTPUT_COLLISION_OR_PRIOR_ATTEMPT")
        command = [sys.executable, "-m", "training.phase63_worker", "dry-run" if kind == "dry-run" else "train", "--seed", str(seed), "--arm", arm, "--binding", str(OUT / "execution-binding.json")]
        receipt = execute(command, cwd=ROOT, root=ZROOT, runtime=RUNTIME, run_id=run_id, phase=63, kind="cuda-dry-run" if kind == "dry-run" else "training", device="cuda:0", repo_head=git("rev-parse", "HEAD"), parent_sha=PARENT_SHA[seed], verifier=lambda s=seed, a=arm, k=kind: verify(s, a, k), temperature=temperature, outputs=[output(seed, arm)] if kind == "train" else [directory / "stdout.log"], previous=prior, query=inventory, display=(), maximum_seconds=3600)
        if not is_complete(directory):
            raise RuntimeError("RECEIPT_NOT_COMPLETE")
        prior = directory
        print(json.dumps({"phase": 63, "kind": kind, "seed": seed, "arm": arm, "receipt": receipt["status"]}), flush=True)
    emit(OUT / ("live-contract-dry-runs.json" if kind == "dry-run" else "training-completion.json"),
         {"phase": 63, "kind": kind, "all_six_complete": True, "receipts": [str(receipt_path(s, a, kind)) for s, a in RUN_ORDER], "binding_sha256": sha(OUT / "execution-binding.json")})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("preflight", "dry-run", "train")); args = parser.parse_args()
    {"preflight": preflight, "dry-run": lambda: run("dry-run"), "train": lambda: run("train")}[args.action]()


if __name__ == "__main__":
    main()
