"""One bounded PHASE63 CUDA worker; it never owns a GPU lock or spawns work.

This module is deliberately separate from the invalidated PHASE61 runner.  It
is invoked only as a suspended child of ``exclusive_cuda_runner.execute``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.checkpoint_paths import checkpoint_root, existing_checkpoint_path
from training.foundation_v31_objective import weighted_lm_loss
from training.optimizer import create_optimizer
from training.run_foundation_v30_eos_experiment import load
from training.run_foundation_v35_thermal_gate import Monitor, cooldown, query_gpu
from training.run_foundation_v36_lr_review import fingerprint, verify_payload
from training.train_foundation_v15_controlled import macro_batch
from training.train_foundation_v21_ab import random_state, restore_random_state

SEEDS = (42, 123, 2026)
ARMS = {"control": 5e-5, "half-lr": 2.5e-5}
PARENT_SHA = {
    42: "a55369c0e98779839750d727517ce6621bc40b5746f975a96bd9cbc0574db2e8",
    123: "78df96b70016dda180f4529833d904e2d448e11469a87278d4881ed09c93dc20",
    2026: "7dcf5fa2da58777040cf9c03f1f513d707e6866f2eaf237bf203bd1b0135f069",
}
PERM_SHA = {
    42: "58551fa0110f3221caa72cd1382ce6e5ca05a0757bd88776071b5efeddb426f8",
    123: "685d76f443d25bd4d271e1550a174e0b6380ad5b8d47020607e12947eb3440c9",
    2026: "8e6cd015ea3e49605306d2be18a67e6c95b1126bcfdf82eb4e25bdda4efc136c",
}
ZROOT = Path(r"Z:\AI\unipilot-mini\checkpoints")
CACHE = Path(r"Z:\AI\unipilot-mini\evaluation\phase57\cache-raw.json")
CACHE_SHA = "4978d51cb66326bccee71a966d6337555387ce74f5ac2d4415506076a69416dc"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_new_json(path: Path, value: dict) -> None:
    path = Path(path)
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError(f"output collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.link(tmp, path)
    tmp.unlink()


def parent(seed: int) -> Path:
    return existing_checkpoint_path(ROOT, "experimental", "phase48", "arm-C", f"seed-{seed}",
                                    "checkpoint-tokens-16384000.pt")


def output(seed: int, arm: str) -> Path:
    return checkpoint_root(ROOT) / "experimental" / "phase63" / "continuation-stability" / arm / f"seed-{seed}" / "checkpoint-tokens-16446464.pt"


def raw_output(seed: int, arm: str) -> Path:
    return ZROOT.parent / "evaluation" / "phase63" / f"run-seed{seed}-{arm}-training-raw.json"


def finite(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def read_binding(path: Path, seed: int, arm: str) -> dict:
    binding = json.loads(Path(path).read_text(encoding="utf-8"))
    if binding.get("phase") != 63 or binding.get("execution_authorized") is not True:
        raise RuntimeError("PHASE63_EXECUTION_BINDING_REQUIRED")
    if binding.get("checkpoint_root") != str(ZROOT) or os.environ.get("UNIPILOT_CHECKPOINT_ROOT") != str(ZROOT):
        raise RuntimeError("PROCESS_ENV_RESOLVER_MISMATCH")
    if checkpoint_root(ROOT) != ZROOT:
        raise RuntimeError("PROCESS_ENV_RESOLVER_MISMATCH")
    rows = {(row["seed"], row["arm"]): row for row in binding["runs"]}
    row = rows.get((seed, arm))
    if row is None or row["parent_sha256"] != PARENT_SHA[seed] or row["runtime_lr"] != ARMS[arm]:
        raise RuntimeError("BINDING_RUN_SCOPE_MISMATCH")
    if Path(row["output"]) != output(seed, arm):
        raise RuntimeError("BINDING_OUTPUT_MISMATCH")
    return binding


def strict_parent(seed: int) -> tuple[dict, Path]:
    source = parent(seed)
    if source != ZROOT / "experimental" / "phase48" / "arm-C" / f"seed-{seed}" / "checkpoint-tokens-16384000.pt" or sha(source) != PARENT_SHA[seed]:
        raise RuntimeError("PARENT_SHA_OR_PATH_MISMATCH")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    integrity = verify_payload(payload, seed, 16_384_000, 5e-5)
    if not integrity["pass"] or fingerprint(payload["permutation"][32000:32122]) != PERM_SHA[seed]:
        raise RuntimeError("PARENT_INTEGRITY_OR_PERMUTATION_FAIL")
    restore_random_state(payload["random_state"], "cuda", cuda_seed=seed)
    if fingerprint(random_state("cuda")) != fingerprint(payload["random_state"]):
        raise RuntimeError("PARENT_RNG_ROUNDTRIP_MISMATCH")
    return payload, source


def state_fingerprint(model, optimizer, payload) -> dict:
    return {"rng": fingerprint(random_state("cuda")), "model": fingerprint(model.state_dict()),
            "optimizer": fingerprint(optimizer.state_dict()), "scheduler": fingerprint(payload["scheduler_state"])}


def matching_forward(model, optimizer, payload, episode: dict) -> dict:
    before = state_fingerprint(model, optimizer, payload)
    mode = model.training
    model.eval()
    with torch.no_grad():
        model(torch.tensor([episode["prefix"] + episode["generated"][:31]], device="cuda"))
    model.train(mode)
    if before != state_fingerprint(model, optimizer, payload):
        raise RuntimeError("MATCHING_FORWARD_STATE_MUTATION")
    return {"state_unchanged": True, "episode_document_index": episode["document_index"]}


def disk_guard(estimated: int) -> None:
    free = shutil.disk_usage(ZROOT).free
    if free < 2 * estimated + 2 * 1024**3:
        raise RuntimeError("DISK_RESERVE_STOP")
    sample = query_gpu()
    if sample["gpu_temperature_c"] >= 85 or sample["hardware_thermal_slowdown"]:
        raise RuntimeError("THERMAL_STOP")


def save(seed: int, arm: str, payload: dict, model, optimizer, source_sha: str, stats: list, telemetry: dict) -> dict:
    destination = output(seed, arm)
    temporary = destination.with_name(destination.name + ".tmp")
    if destination.exists() or temporary.exists():
        raise FileExistsError("OUTPUT_COLLISION_OR_PARTIAL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    runtime = {"arm": arm, "optimizer_group_lr": ARMS[arm], "constant_schedule": True,
               "scheduler_step_called": False,
               "historical_scheduler_metadata_warning": "learning_rate and peak_learning_rate are historical metadata, not runtime authority"}
    saved = {**payload, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
             "scheduler_state": {**payload["scheduler_state"], "global_step": 32122},
             "random_state": random_state("cuda"), "update": 32122, "tokens_processed": 16_446_464,
             "phase": 63, "arm": arm, "experimental": True, "EXPERIMENTAL": True,
             "canonical": False, "NOT_CANONICAL": True, "not_canonical": True,
             "NOT_PRODUCTION": True, "not_production": True, "precision_mode": "fp32",
             "eos_loss_weight": 1.5, "repetition_auxiliary": False,
             "parent_checkpoint_sha256": source_sha, "phase63_runtime_lr_contract": runtime,
             "phase63_training": {"updates": 122, "lm_gradient_tokens": 62464, "matching_slots": 15,
                                  "conservative_positions": 63904, "stats_count": len(stats)}}
    torch.save(saved, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    strict_model = DiagnosticTransformerV17(DiagnosticConfigV17(**loaded["config"]))
    strict_model.load_state_dict(loaded["model_state"], strict=True)
    strict_optimizer = create_optimizer(strict_model, ARMS[arm], .1)
    strict_optimizer.load_state_dict(loaded["optimizer_state"])
    checks = {"strict_model_reload": True, "strict_optimizer_reload": True,
              "payload": verify_payload(loaded, seed, 16_446_464, ARMS[arm])["pass"],
              "scheduler": loaded["scheduler_state"]["global_step"] == 32122,
              "sampler": fingerprint(loaded["permutation"]) == fingerprint(payload["permutation"]),
              "rng": fingerprint(loaded["random_state"]) == fingerprint(saved["random_state"]),
              "runtime_lr": all(g["lr"] == ARMS[arm] for g in strict_optimizer.param_groups),
              "finite_optimizer": finite(strict_optimizer.state_dict()),
              "markers": all(loaded[k] is True for k in ("EXPERIMENTAL", "NOT_CANONICAL", "NOT_PRODUCTION"))}
    if not all(checks.values()):
        raise RuntimeError("OUTPUT_STRICT_RELOAD_FAIL")
    os.link(temporary, destination)
    temporary.unlink()
    return {"path": str(destination), "sha256": sha(destination), "bytes": destination.stat().st_size,
            "strict_reload": True, "updates_complete": True, "resume_integrity": checks}


def dry_run(seed: int, arm: str, binding: Path) -> None:
    read_binding(binding, seed, arm)
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA_REQUIRED")
    payload, source = strict_parent(seed)
    source_sha = sha(source)
    cool = cooldown()
    if not cool["target_reached"]:
        raise RuntimeError("THERMAL_COOLDOWN_FAILED")
    _loaded, model, _optimizer = load(source, torch.device("cuda"))
    try:
        # Contract dry-run intentionally performs one FP32 no-grad CUDA forward only.
        model.eval()
        with torch.no_grad():
            model(torch.tensor([[1, 2, 3, 4]], device="cuda"))
        torch.cuda.synchronize()
    finally:
        del model; gc.collect(); torch.cuda.empty_cache()
    if sha(source) != source_sha or payload["update"] != 32000:
        raise RuntimeError("DRY_RUN_PARENT_MUTATED")
    print(json.dumps({"phase": 63, "kind": "cuda-dry-run", "seed": seed, "arm": arm, "pass": True}), flush=True)


def train(seed: int, arm: str, binding: Path) -> None:
    read_binding(binding, seed, arm)
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 2070 SUPER":
        raise RuntimeError("CUDA_RTX2070_SUPER_REQUIRED")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    payload, source = strict_parent(seed); source_sha = sha(source)
    cool = cooldown()
    if not cool["target_reached"]:
        raise RuntimeError("THERMAL_COOLDOWN_FAILED")
    if sha(CACHE) != CACHE_SHA:
        raise RuntimeError("MATCHING_CACHE_INTEGRITY_FAIL")
    episodes = json.loads(CACHE.read_text(encoding="utf-8"))["episodes"]
    if len(episodes) != 128:
        raise RuntimeError("MATCHING_CACHE_GEOMETRY_FAIL")
    loaded, model, optimizer = load(source, torch.device("cuda"))
    if loaded["update"] != 32000:
        raise RuntimeError("PARENT_STEP_MISMATCH")
    before_moments = fingerprint(optimizer.state_dict()["state"])
    before_groups = [{k: v for k, v in group.items() if k != "lr"} for group in optimizer.param_groups]
    for group in optimizer.param_groups: group["lr"] = ARMS[arm]
    if fingerprint(optimizer.state_dict()["state"]) != before_moments or before_groups != [{k: v for k, v in group.items() if k != "lr"} for group in optimizer.param_groups]:
        raise RuntimeError("LR_ONLY_ASSIGNMENT_FAIL")
    if not all(group["lr"] == ARMS[arm] for group in optimizer.param_groups):
        raise RuntimeError("RUNTIME_LR_CONTRACT_FAIL")
    tokenizer = FoundationTokenizer.load(ROOT / "tokenizer/foundation-v11-base-4096.json")
    train_data = np.memmap(ROOT / "data/foundation_v11/packed/vocab-4096/train.bin", dtype=np.uint16, mode="r")
    estimated = source.stat().st_size; model.train(); torch.cuda.reset_peak_memory_stats(); monitor = Monitor(); monitor.start()
    stats = []; over10 = 0; started = time.perf_counter()
    try:
        for update in range(32001, 32123):
            disk_guard(estimated)
            if not all(group["lr"] == ARMS[arm] for group in optimizer.param_groups):
                raise RuntimeError("RUNTIME_LR_CONTRACT_FAIL")
            x, y = macro_batch(train_data, int(payload["permutation"][update - 1]), 512); x, y = x.cuda(), y.cuda()
            optimizer.zero_grad(set_to_none=True); logits, _ = model(x); lm, eos, non = weighted_lm_loss(logits, y, tokenizer.eos_id, 1.5)
            matching = matching_forward(model, optimizer, payload, episodes[(update // 8 - 1) % 128]) if update % 8 == 0 else None
            if not torch.isfinite(lm): raise RuntimeError("NONFINITE_LOSS")
            lm.backward(); norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); over10 = over10 + 1 if norm > 10 else 0
            if not math.isfinite(norm) or norm > 100 or over10 >= 3: raise RuntimeError("GRADIENT_STOP")
            optimizer.step()
            if not finite(model.state_dict()) or not finite(optimizer.state_dict()): raise RuntimeError("NONFINITE_STATE")
            stats.append({"update": update, "lm_loss": float(lm.detach()), "eos_loss": float(eos.detach()), "non_eos_loss": float(non.detach()), "gradient_norm_raw": norm, "matching_forward": matching, "runtime_lr": ARMS[arm]})
        torch.cuda.synchronize()
    finally:
        elapsed = time.perf_counter() - started; telemetry = monitor.finish()
    if len(stats) != 122 or sum(row["matching_forward"] is not None for row in stats) != 15:
        raise RuntimeError("TRAINING_BUDGET_OR_MATCHING_GEOMETRY_FAIL")
    if not telemetry.get("samples") or telemetry["gpu_temperature_c_max"] >= 85 or telemetry["hardware_thermal_slowdown"]:
        raise RuntimeError("THERMAL_STOP")
    if sha(source) != source_sha:
        raise RuntimeError("PARENT_MUTATED")
    checkpoint = save(seed, arm, payload, model, optimizer, source_sha, stats, telemetry)
    if sha(source) != source_sha:
        raise RuntimeError("PARENT_MUTATED")
    norms = np.asarray([row["gradient_norm_raw"] for row in stats])
    raw = {"phase": 63, "seed": seed, "arm": arm, "lr": ARMS[arm], "parent_sha256": source_sha,
           "checkpoint": checkpoint, "start_update": 32000, "end_update": 32122, "new_training": True,
           "training": {"seconds": elapsed, "tokens_per_second": 62464 / elapsed, "mean_lm_loss": float(np.mean([row["lm_loss"] for row in stats])), "gradient_norm_max": float(norms.max()), "clip_rate": float(np.mean(norms > 1)), "telemetry": telemetry}, "stats": stats}
    atomic_new_json(raw_output(seed, arm), raw)
    print(json.dumps({"phase": 63, "kind": "training", "seed": seed, "arm": arm, "checkpoint": checkpoint, "pass": True}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("kind", choices=("dry-run", "train")); parser.add_argument("--seed", type=int, choices=SEEDS, required=True); parser.add_argument("--arm", choices=tuple(ARMS), required=True); parser.add_argument("--binding", type=Path, required=True); args = parser.parse_args()
    torch.set_num_threads(2)
    {"dry-run": dry_run, "train": train}[args.kind](args.seed, args.arm, args.binding)


if __name__ == "__main__":
    main()
