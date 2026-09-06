"""PHASE 47 isolated fixed-LR experiments; never promote or overwrite checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.run_foundation_v35_thermal_gate import Monitor, cooldown, resume_checks
from training.run_foundation_v30_eos_experiment import load
from training.train_foundation_v21_ab import file_sha256, random_state
from training.train_foundation_v15_controlled import macro_batch
from training.foundation_v31_objective import weighted_lm_loss
from training.optimizer import create_optimizer
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17

ARMS = {"A": 1e-4, "B": 7.5e-5, "C": 5e-5}
SEEDS = (42, 123, 2026)
START = 15_872_000
BUDGET = 256_000
OUT = ROOT / "checkpoints/experimental/phase47"
EVAL = ROOT / "evaluation/phase47"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def official(seed, final=False):
    if final:
        return ROOT / f"checkpoints/foundation-v35-thermal-short-gate/gate-1/seed-{seed}/checkpoint-tokens-16128000.pt"
    return ROOT / f"checkpoints/foundation-v33-context-gate/gate-2/seed-{seed}/checkpoint-tokens-15872000.pt"


def checkpoint(arm, seed):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("unregistered arm or seed")
    return OUT / f"arm-{arm}/seed-{seed}/checkpoint-tokens-16128000.pt"


def fingerprint(value):
    digest = hashlib.sha256()
    def visit(x):
        digest.update(type(x).__name__.encode())
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous().numpy()
            digest.update(str((a.shape, a.dtype)).encode()); digest.update(a.tobytes())
        elif isinstance(x, np.ndarray):
            digest.update(str((x.shape, x.dtype)).encode()); digest.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                visit(key); visit(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x:
                visit(item)
        else:
            digest.update(repr(x).encode())
    visit(value)
    return digest.hexdigest()


def set_lr_only(optimizer, lr):
    before = fingerprint(optimizer.state_dict()["state"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    if fingerprint(optimizer.state_dict()["state"]) != before:
        raise RuntimeError("LR assignment changed optimizer moments")
    return before


def verify_payload(payload, seed, tokens, lr):
    model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer = create_optimizer(model, lr, 0.1)
    optimizer.load_state_dict(payload["optimizer_state"])
    checks = {
        "strict_model_optimizer_reload": True,
        "parameter_count": model.parameter_count() == 19_514_880,
        "fp32": payload["precision_mode"] == "fp32" and all(p.dtype == torch.float32 for p in model.parameters()),
        "finite_weights": all(torch.isfinite(p).all().item() for p in model.parameters()),
        "finite_optimizer": all(not torch.is_tensor(v) or torch.isfinite(v).all().item() for state in optimizer.state.values() for v in state.values()),
        "seed_tokens": payload["seed"] == seed and payload["tokens_processed"] == tokens and payload["update"] * 512 == tokens,
        "scheduler": payload["scheduler_state"]["global_step"] == payload["update"],
        "lr": all(g["lr"] == lr for g in optimizer.param_groups),
        "optimizer_steps": all(int(s["step"]) == payload["update"] for s in optimizer.state.values()),
        "sampler": torch.is_tensor(payload["permutation"]) and len(payload["permutation"]) > payload["update"],
        "rng": {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(payload["random_state"]),
        "eos_rep": payload["eos_loss_weight"] == 1.5 and payload["repetition_auxiliary"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"integrity failure: {checks}")
    return {"checks": checks, "pass": True}


def preflight():
    torch.set_num_threads(4)
    free = shutil.disk_usage(ROOT).free
    if free < 20 * 1024**3:
        raise RuntimeError("C drive needs >=20 GiB free")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA build/device required")
    gpu = torch.cuda.get_device_name(0)
    if gpu != "NVIDIA GeForce RTX 2070 SUPER":
        raise RuntimeError(gpu)
    rows = []
    for final in (False, True):
        for seed in SEEDS:
            path = official(seed, final)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            expected = read_json(ROOT / f"evaluation/phase46/{'gate1' if final else 'baseline'}/seed-{seed}.json")["checkpoint_sha256"]
            digest = file_sha256(path)
            assert digest == expected, f"historical SHA mismatch: {path}"
            rows.append({"path": str(path.relative_to(ROOT)), "sha256": digest,
                         "integrity": verify_payload(payload, seed, START + (BUDGET if final else 0), 1e-4)})
    result = {"free_bytes": free, "gpu": gpu, "torch": torch.__version__, "cuda_build": torch.version.cuda,
              "checkpoints": rows, "pass": True,
              "previous_failures": [
                  {"test": "test_worker_output_directories_do_not_collide", "expected": "three isolated directories", "actual": "AttributeError: OUT missing"},
                  {"test": "test_ready_protocol_rejects_missing_marker", "expected": "RuntimeError for absent READY marker", "actual": "AttributeError: OUT missing"}],
              "failure_root_cause": "Existing storage routing replaced OUT with OUT_PARTS/checkpoint(); tests referenced removed internal constant. Tests now use checkpoint() and preserve behavioral assertions."}
    write_json(EVAL / "preflight.json", result)
    print(json.dumps({"preflight": True, "free_gib": free / 1024**3, "checkpoints": len(rows), "gpu": gpu}), flush=True)


def train(arm, seed):
    if read_json(EVAL / "tests-preflight.json")["failed"] != 0:
        raise RuntimeError("full pytest pass required before training")
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError("C drive space below 20 GiB")
    target = checkpoint(arm, seed)
    if target.exists() or target.with_suffix(".pt.tmp").exists():
        raise FileExistsError(target)
    if seed != 42:
        selection = read_json(EVAL / "selection.json")
        if selection.get("best_arm") != arm or not selection.get("clear_best"):
            raise RuntimeError("3-seed confirmation requires a clear selected candidate")
    source = official(seed)
    expected = next(r["sha256"] for r in read_json(EVAL / "preflight.json")["checkpoints"] if Path(r["path"]) == source.relative_to(ROOT))
    assert file_sha256(source) == expected
    copy_path = OUT / f"sources/seed-{seed}/checkpoint-tokens-15872000.pt"
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    if not copy_path.exists():
        shutil.copy2(source, copy_path)
    assert file_sha256(copy_path) == expected
    thermal = cooldown()
    device = torch.device("cuda")
    payload, model, optimizer = load(copy_path, device)
    assert resume_checks(payload, seed, START, 1)["pass"]
    continuity = {key: fingerprint(payload[key]) for key in ("model_state", "optimizer_state", "scheduler_state", "permutation", "random_state")}
    assert fingerprint(optimizer.state_dict()) == continuity["optimizer_state"]
    assert fingerprint(random_state(device)) == continuity["random_state"]
    moments = set_lr_only(optimizer, ARMS[arm])
    tok = FoundationTokenizer.load(ROOT / "tokenizer/foundation-v11-base-4096.json")
    data = np.memmap(ROOT / "data/foundation_v11/packed/vocab-4096/train.bin", dtype=np.uint16, mode="r")
    start_update = payload["update"]
    end_update = start_update + BUDGET // 512
    model.train(); torch.cuda.reset_peak_memory_stats()
    monitor = Monitor(); monitor.start()
    norms, losses = [], []
    started = time.perf_counter()
    try:
        for step in range(start_update + 1, end_update + 1):
            x, y = macro_batch(data, int(payload["permutation"][step - 1]), 512)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss, _, _ = weighted_lm_loss(logits, y, tok.eos_id, 1.5)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(norm)
            optimizer.step()
            norms.append(float(norm)); losses.append(float(loss.detach()))
        torch.cuda.synchronize()
    finally:
        seconds = time.perf_counter() - started
        telemetry = monitor.finish()
    saved = {**payload, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
             "scheduler_state": {**payload["scheduler_state"], "global_step": end_update},
             "random_state": random_state(device), "update": end_update, "tokens_processed": START+BUDGET,
             "phase": 47, "experimental": True, "formal_research": False, "promoted": False,
             "arm": arm, "experimental_lr": ARMS[arm], "source_sha256": expected, "source_checkpoint": str(copy_path.relative_to(ROOT))}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".pt.tmp")
    torch.save(saved, temp)
    loaded = torch.load(temp, map_location="cpu", weights_only=False)
    integrity = verify_payload(loaded, seed, START+BUDGET, ARMS[arm])
    assert fingerprint(loaded["permutation"]) == continuity["permutation"]
    assert file_sha256(source) == file_sha256(copy_path) == expected
    temp.replace(target)
    result = {"arm": arm, "seed": seed, "lr": ARMS[arm], "budget": BUDGET,
              "checkpoint": str(target.relative_to(ROOT)), "sha256": file_sha256(target), "source_sha256": expected,
              "continuity_fingerprints": continuity, "optimizer_moments_before_lr_assignment": moments,
              "integrity": integrity, "source_unchanged": True, "cooldown": thermal,
              "seconds": seconds, "tokens_per_second": BUDGET/seconds,
              "train_loss": float(np.mean(losses)), "gradient_mean": float(np.mean(norms)),
              "gradient_std": float(np.std(norms, ddof=1)), "gradient_max": max(norms),
              "gradient_norms": norms, "weighted_losses": losses,
              "peak_vram_mib": torch.cuda.max_memory_allocated()/1048576, "telemetry": telemetry,
              "parallel_cpu_evaluation": "DISABLED", "settings_changed": False,
              "precision": "FP32", "amp": False, "eos_weight": 1.5, "repetition_auxiliary": False}
    write_json(EVAL/f"arm-{arm}/seed-{seed}-training.json", result)
    print(json.dumps({"arm": arm, "seed": seed, "tps": result["tokens_per_second"], "integrity": True, "max_temp": telemetry.get("gpu_temperature_c_max")}), flush=True)


def verify_final():
    results=[]
    for row in read_json(EVAL/'preflight.json')['checkpoints']:
        path=ROOT/row['path']; assert file_sha256(path)==row['sha256']
        p=torch.load(path,map_location='cpu',weights_only=False)
        results.append({'path':row['path'],'sha256_unchanged':True,'integrity':verify_payload(p,p['seed'],p['tokens_processed'],1e-4)})
    for path in sorted(EVAL.glob('arm-*/seed-*-training.json')):
        row=read_json(path); target=ROOT/row['checkpoint']
        assert file_sha256(target)==row['sha256']
        p=torch.load(target,map_location='cpu',weights_only=False)
        assert p['phase']==47 and p['experimental'] and not p['promoted']
        results.append({'path':row['checkpoint'],'sha256_unchanged':True,'integrity':verify_payload(p,row['seed'],START+BUDGET,row['lr'])})
    write_json(EVAL/'integrity-final.json',{'count':len(results),'pass':True,'checkpoints':results})
    print(json.dumps({'checkpoint_readback_pass':len(results)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify-final", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.preflight:
        preflight()
    elif args.verify_final:
        verify_final()
    else:
        train(args.arm, args.seed)
