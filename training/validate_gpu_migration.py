"""PHASE 39 CPU/CUDA checkpoint, parity, resume, and speed gates."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.device import (
    NvidiaSmiMonitor,
    describe_device,
    move_optimizer_state_to_device,
    optimizer_state_devices,
    resolve_device,
)
from training.optimizer import create_optimizer
from training.train_foundation_v15_controlled import macro_batch
from training.train_foundation_v21_ab import (
    TOKENS_PER_UPDATE,
    build_paired_model,
    file_sha256,
    load_json,
    restore_random_state,
    save_checkpoint,
    stateless_scheduler_state,
)
from training.train_foundation_v22_current import preflight_resume, write_json


SEED = 42
START_TOKENS = 10_240_000
CHECKPOINT = ROOT / (
    "checkpoints/foundation-v26-current/current/seed-42/"
    "checkpoint-tokens-10240000.pt"
)
SETTINGS_PATH = "configs/unipilot-foundation-v28.json"
PARITY_TOLERANCES = {
    "loss_absolute": 5e-4,
    "logit_mean_absolute": 2e-4,
    "logit_std_absolute": 2e-4,
    "topk_absolute": 2e-3,
    "one_step_loss_absolute": 1e-3,
    "gradient_norm_relative": 1e-2,
    "parameter_delta_relative": 2e-2,
    "learning_rate_absolute": 1e-12,
}


def _load_model_optimizer(
    settings: dict, checkpoint: Path, device: str | torch.device
) -> tuple[dict, DiagnosticTransformerV17, torch.optim.Optimizer]:
    target = resolve_device(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    model = build_paired_model(settings, tokenizer, "current", int(payload["seed"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(target)
    optimizer = create_optimizer(
        model,
        float(settings["training"]["peak_learning_rate"]),
        float(settings["training"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer_state"])
    move_optimizer_state_to_device(optimizer, target)
    return payload, model, optimizer


def _train_memmap(settings: dict) -> np.memmap:
    corpus = load_json(settings["corpus_manifest"])
    return np.memmap(
        ROOT / corpus["splits"]["train"]["path"], dtype=np.uint16, mode="r"
    )


def _batch_at(
    train: np.memmap, permutation: torch.Tensor, update_index: int, context: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return macro_batch(train, int(permutation[update_index]), context)


@torch.inference_mode()
def _forward_metrics(
    model: DiagnosticTransformerV17, batch: tuple[torch.Tensor, torch.Tensor]
) -> tuple[dict, torch.Tensor]:
    device = next(model.parameters()).device
    inputs, targets = (value.to(device) for value in batch)
    model.eval()
    logits, loss = model(inputs, targets)
    if loss is None:
        raise RuntimeError("forward loss was not produced")
    top = logits.topk(10, dim=-1).indices
    result = {
        "loss": float(loss),
        "logit_mean": float(logits.float().mean()),
        "logit_std": float(logits.float().std()),
        "logit_min": float(logits.float().min()),
        "logit_max": float(logits.float().max()),
        "top_1": float((top[..., 0] == targets).float().mean()),
        "top_5": float((top[..., :5] == targets[..., None]).any(-1).float().mean()),
        "top_10": float((top == targets[..., None]).any(-1).float().mean()),
        "all_finite": bool(torch.isfinite(logits).all() and torch.isfinite(loss)),
    }
    return result, logits.detach().float().cpu()


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def _compare_forward(cpu: dict, gpu: dict, cpu_logits: torch.Tensor, gpu_logits: torch.Tensor) -> dict:
    differences = {
        "loss_absolute": abs(cpu["loss"] - gpu["loss"]),
        "logit_mean_absolute": abs(cpu["logit_mean"] - gpu["logit_mean"]),
        "logit_std_absolute": abs(cpu["logit_std"] - gpu["logit_std"]),
        "logit_max_absolute": float((cpu_logits - gpu_logits).abs().max()),
        "top_1_absolute": abs(cpu["top_1"] - gpu["top_1"]),
        "top_5_absolute": abs(cpu["top_5"] - gpu["top_5"]),
        "top_10_absolute": abs(cpu["top_10"] - gpu["top_10"]),
    }
    passed = (
        differences["loss_absolute"] <= PARITY_TOLERANCES["loss_absolute"]
        and differences["logit_mean_absolute"] <= PARITY_TOLERANCES["logit_mean_absolute"]
        and differences["logit_std_absolute"] <= PARITY_TOLERANCES["logit_std_absolute"]
        and all(
            differences[key] <= PARITY_TOLERANCES["topk_absolute"]
            for key in ("top_1_absolute", "top_5_absolute", "top_10_absolute")
        )
        and cpu["all_finite"] and gpu["all_finite"]
    )
    return {
        "cpu": cpu,
        "gpu": gpu,
        "differences": differences,
        "tolerances": PARITY_TOLERANCES,
        "status": "PASS" if passed else "FAIL",
    }


def _optimizer_step(
    model: DiagnosticTransformerV17,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor],
    *,
    gradient_clip: float,
    training_mode: bool,
) -> dict:
    device = next(model.parameters()).device
    inputs, targets = (value.to(device) for value in batch)
    model.train(training_mode)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(inputs, targets)
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("non-finite parity loss")
    pre_loss = float(loss.detach())
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip))
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    squared_delta = sum(
        float((parameter.detach() - initial).float().square().sum())
        for parameter, initial in zip(model.parameters(), before)
    )
    model.eval()
    with torch.inference_mode():
        _, post_loss = model(inputs, targets)
    result = {
        "pre_step_loss": pre_loss,
        "post_step_loss": float(post_loss),
        "gradient_norm": gradient_norm,
        "parameter_delta_norm": math.sqrt(squared_delta),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "all_finite": all(
            math.isfinite(value)
            for value in (pre_loss, float(post_loss), gradient_norm, math.sqrt(squared_delta))
        ),
    }
    del before
    return result


def _one_step_parity(settings: dict, batch: tuple[torch.Tensor, torch.Tensor]) -> dict:
    payload_cpu, cpu_model, cpu_optimizer = _load_model_optimizer(settings, CHECKPOINT, "cpu")
    restore_random_state(payload_cpu["random_state"], "cpu", cuda_seed=SEED)
    cpu = _optimizer_step(
        cpu_model, cpu_optimizer, batch,
        gradient_clip=float(settings["training"]["gradient_clip"]),
        training_mode=False,
    )
    del payload_cpu, cpu_model, cpu_optimizer

    payload_gpu, gpu_model, gpu_optimizer = _load_model_optimizer(settings, CHECKPOINT, "cuda")
    rng_action = restore_random_state(payload_gpu["random_state"], "cuda", cuda_seed=SEED)
    gpu = _optimizer_step(
        gpu_model, gpu_optimizer, batch,
        gradient_clip=float(settings["training"]["gradient_clip"]),
        training_mode=False,
    )
    gpu_devices = optimizer_state_devices(gpu_optimizer)
    del payload_gpu, gpu_model, gpu_optimizer
    torch.cuda.empty_cache()

    differences = {
        "pre_step_loss_absolute": abs(cpu["pre_step_loss"] - gpu["pre_step_loss"]),
        "post_step_loss_absolute": abs(cpu["post_step_loss"] - gpu["post_step_loss"]),
        "gradient_norm_relative": _relative_difference(cpu["gradient_norm"], gpu["gradient_norm"]),
        "parameter_delta_relative": _relative_difference(
            cpu["parameter_delta_norm"], gpu["parameter_delta_norm"]
        ),
        "learning_rate_absolute": abs(cpu["learning_rate"] - gpu["learning_rate"]),
    }
    passed = (
        differences["pre_step_loss_absolute"] <= PARITY_TOLERANCES["one_step_loss_absolute"]
        and differences["post_step_loss_absolute"] <= PARITY_TOLERANCES["one_step_loss_absolute"]
        and differences["gradient_norm_relative"] <= PARITY_TOLERANCES["gradient_norm_relative"]
        and differences["parameter_delta_relative"] <= PARITY_TOLERANCES["parameter_delta_relative"]
        and differences["learning_rate_absolute"] <= PARITY_TOLERANCES["learning_rate_absolute"]
        and cpu["all_finite"] and gpu["all_finite"]
        and gpu_devices and all(value.startswith("cuda") for value in gpu_devices)
    )
    return {
        "method": "optimizer step with dropout disabled to isolate CPU/CUDA kernels",
        "cpu": cpu,
        "gpu": gpu,
        "gpu_optimizer_state_devices": gpu_devices,
        "cuda_rng_resume_action": rng_action,
        "differences": differences,
        "status": "PASS" if passed else "FAIL",
    }


def _resume_safety(
    settings: dict, train: np.memmap, batch: tuple[torch.Tensor, torch.Tensor]
) -> dict:
    payload, model, optimizer = _load_model_optimizer(settings, CHECKPOINT, "cuda")
    rng_action = restore_random_state(payload["random_state"], "cuda", cuda_seed=SEED)
    first_step = _optimizer_step(
        model, optimizer, batch,
        gradient_clip=float(settings["training"]["gradient_clip"]),
        training_mode=True,
    )
    update = int(payload["update"]) + 1
    test_path = ROOT / "checkpoints/foundation-v28-gpu-validation/seed-42/gpu-resume-step-1.pt"
    saved = save_checkpoint(
        test_path,
        model=model,
        optimizer=optimizer,
        variant="current",
        seed=SEED,
        update=update,
        permutation=payload["permutation"],
        history=payload["history"],
        training_seconds=float(payload["training_seconds"]),
        settings=settings,
        precision_mode="fp32",
    )
    del model, optimizer
    torch.cuda.empty_cache()

    cpu_payload = torch.load(test_path, map_location="cpu", weights_only=False)
    cpu_model = DiagnosticTransformerV17(DiagnosticConfigV17(**cpu_payload["config"]))
    cpu_model.load_state_dict(cpu_payload["model_state"], strict=True)
    cpu_optimizer = create_optimizer(
        cpu_model,
        float(settings["training"]["peak_learning_rate"]),
        float(settings["training"]["weight_decay"]),
    )
    cpu_optimizer.load_state_dict(cpu_payload["optimizer_state"])
    cpu_fallback = {
        "strict_model_reload": True,
        "optimizer_state": bool(cpu_payload["optimizer_state"]["state"]),
        "optimizer_devices": optimizer_state_devices(cpu_optimizer),
        "scheduler": cpu_payload["scheduler_state"] == stateless_scheduler_state(settings, update),
        "rng": {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(
            cpu_payload["random_state"]
        ),
        "sampler": torch.equal(cpu_payload["permutation"], payload["permutation"]),
        "processed_tokens": int(cpu_payload["tokens_processed"]) == update * TOKENS_PER_UPDATE,
    }
    del cpu_model, cpu_optimizer, cpu_payload

    reloaded_payload, reloaded_model, reloaded_optimizer = _load_model_optimizer(
        settings, test_path, "cuda"
    )
    reload_rng_action = restore_random_state(
        reloaded_payload["random_state"], "cuda", cuda_seed=SEED
    )
    next_batch = _batch_at(
        train,
        reloaded_payload["permutation"],
        int(reloaded_payload["update"]),
        int(reloaded_model.config.context_length),
    )
    continuation = _optimizer_step(
        reloaded_model, reloaded_optimizer, next_batch,
        gradient_clip=float(settings["training"]["gradient_clip"]),
        training_mode=True,
    )
    optimizer_devices = optimizer_state_devices(reloaded_optimizer)
    passed = (
        saved["integrity"] == "PASS"
        and all(cpu_fallback.values())
        and continuation["all_finite"]
        and reload_rng_action == "restored_from_checkpoint"
        and optimizer_devices and all(value.startswith("cuda") for value in optimizer_devices)
    )
    del payload, reloaded_payload, reloaded_model, reloaded_optimizer
    torch.cuda.empty_cache()
    return {
        "legacy_cpu_checkpoint_cuda_rng_action": rng_action,
        "gpu_checkpoint": saved,
        "gpu_to_cpu": cpu_fallback,
        "gpu_reload_cuda_rng_action": reload_rng_action,
        "gpu_optimizer_state_devices_after_reload": optimizer_devices,
        "first_step": first_step,
        "continuation_step": continuation,
        "status": "PASS" if passed else "FAIL",
    }


def _benchmark(
    settings: dict,
    train: np.memmap,
    device: str,
    steps: int,
    *,
    amp: bool = False,
) -> dict:
    payload, model, optimizer = _load_model_optimizer(settings, CHECKPOINT, device)
    actual = next(model.parameters()).device
    restore_random_state(payload["random_state"], actual, cuda_seed=SEED)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    warmup_steps = 5
    process = psutil.Process()
    losses: list[float] = []
    gradients: list[float] = []
    finite = True

    def run(index: int) -> None:
        nonlocal finite
        batch = _batch_at(
            train,
            payload["permutation"],
            int(payload["update"]) + index,
            int(model.config.context_length),
        )
        inputs, targets = (value.to(actual) for value in batch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp else nullcontext()
        )
        with context:
            _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            finite = False
            raise RuntimeError("non-finite benchmark loss")
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings["training"]["gradient_clip"])
        )
        if amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        losses.append(float(loss.detach()))
        gradients.append(float(gradient))

    for index in range(warmup_steps):
        run(index)
    if actual.type == "cuda":
        torch.cuda.synchronize(actual)
        torch.cuda.reset_peak_memory_stats(actual)
    losses.clear()
    gradients.clear()
    monitor = NvidiaSmiMonitor(actual.type == "cuda", interval_seconds=0.1)
    monitor.start()
    process.cpu_percent(None)
    peak_ram = process.memory_info().rss / (1024 * 1024)
    started = time.perf_counter()
    for index in range(warmup_steps, warmup_steps + steps):
        run(index)
        peak_ram = max(peak_ram, process.memory_info().rss / (1024 * 1024))
    if actual.type == "cuda":
        torch.cuda.synchronize(actual)
    elapsed = time.perf_counter() - started
    cpu_percent = process.cpu_percent(None)
    telemetry = monitor.stop()
    peak_vram = (
        torch.cuda.max_memory_allocated(actual) / (1024 * 1024)
        if actual.type == "cuda" else None
    )
    result = {
        "device": str(actual),
        "precision": "amp_fp16" if amp else "fp32",
        "warmup_steps": warmup_steps,
        "measurement_steps": steps,
        "tokens": steps * TOKENS_PER_UPDATE,
        "seconds": elapsed,
        "tokens_per_second": steps * TOKENS_PER_UPDATE / elapsed,
        "milliseconds_per_update": elapsed * 1000 / steps,
        "peak_ram_mb": peak_ram,
        "peak_vram_mb": peak_vram,
        "process_cpu_percent": cpu_percent,
        "gpu_telemetry": telemetry,
        "mean_loss": sum(losses) / len(losses),
        "max_gradient_norm": max(gradients),
        "all_finite": finite and all(math.isfinite(value) for value in losses + gradients),
    }
    del payload, model, optimizer
    if actual.type == "cuda":
        torch.cuda.empty_cache()
    return result


def preflight(settings: dict) -> dict:
    device = resolve_device("cuda")
    phase38 = load_json("evaluation/foundation-v27-summary.json")
    checkpoint_audit = preflight_resume(settings, SEED, CHECKPOINT)
    blind = settings["final_blind"]
    blind_hash = file_sha256(ROOT / blind["path"])
    result = {
        "schema": "gpu-migration-preflight-v1",
        "phase": 39,
        "phase38_final_tokens": phase38["training_curve"][-1]["tokens"],
        "phase38_gate": phase38["final_gate"],
        "device": describe_device(device),
        "checkpoint": checkpoint_audit,
        "final_blind": {
            "sha256": blind_hash,
            "expected_sha256": blind["expected_sha256"],
            "content_opened": False,
            "status": "PASS" if blind_hash == blind["expected_sha256"] else "FAIL",
        },
    }
    passed = (
        result["phase38_final_tokens"] == START_TOKENS
        and result["phase38_gate"] == "CONTINUE_15M_GENERATION_LAG"
        and result["device"]["cuda_available"]
        and result["device"].get("name") == "NVIDIA GeForce RTX 2070 SUPER"
        and checkpoint_audit["status"] == "PASS"
        and result["final_blind"]["status"] == "PASS"
    )
    result["status"] = "PASS" if passed else "FAIL"
    return result


def run_validation(benchmark_steps: int = 50, test_amp: bool = True) -> dict:
    if not 50 <= benchmark_steps <= 200:
        raise ValueError("benchmark steps must be between 50 and 200")
    settings = load_json(SETTINGS_PATH)
    preliminary = preflight(settings)
    if preliminary["status"] != "PASS":
        raise RuntimeError(f"GPU migration preflight failed: {preliminary}")
    train = _train_memmap(settings)
    base_payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    batch = _batch_at(
        train,
        base_payload["permutation"],
        int(base_payload["update"]),
        int(base_payload["config"]["context_length"]),
    )
    del base_payload

    cpu_payload, cpu_model, cpu_optimizer = _load_model_optimizer(settings, CHECKPOINT, "cpu")
    cpu_metrics, cpu_logits = _forward_metrics(cpu_model, batch)
    del cpu_payload, cpu_model, cpu_optimizer
    gpu_payload, gpu_model, gpu_optimizer = _load_model_optimizer(settings, CHECKPOINT, "cuda")
    gpu_metrics, gpu_logits = _forward_metrics(gpu_model, batch)
    optimizer_gpu_devices = optimizer_state_devices(gpu_optimizer)
    numerical = _compare_forward(cpu_metrics, gpu_metrics, cpu_logits, gpu_logits)
    del gpu_payload, gpu_model, gpu_optimizer, cpu_logits, gpu_logits
    torch.cuda.empty_cache()

    one_step = _one_step_parity(settings, batch)
    resume = _resume_safety(settings, train, batch)
    cpu_benchmark = _benchmark(settings, train, "cpu", benchmark_steps)
    gpu_benchmark = _benchmark(settings, train, "cuda", benchmark_steps)
    speedup = gpu_benchmark["tokens_per_second"] / cpu_benchmark["tokens_per_second"]
    amp_benchmark = None
    if test_amp:
        amp_benchmark = _benchmark(settings, train, "cuda", benchmark_steps, amp=True)

    device = preliminary["device"]
    total_vram_mb = device["total_memory_bytes"] / (1024 * 1024)
    vram_safe = (
        gpu_benchmark["peak_vram_mb"] is not None
        and gpu_benchmark["peak_vram_mb"] < total_vram_mb * 0.85
    )
    telemetry = gpu_benchmark["gpu_telemetry"] or {}
    gpu_utilization_confirmed = telemetry.get("gpu_utilization_percent_max", 0) > 0
    no_nan_inf = (
        numerical["cpu"]["all_finite"]
        and numerical["gpu"]["all_finite"]
        and one_step["cpu"]["all_finite"]
        and one_step["gpu"]["all_finite"]
        and resume["first_step"]["all_finite"]
        and resume["continuation_step"]["all_finite"]
        and gpu_benchmark["all_finite"]
    )
    gate_checks = {
        "cuda_available": device["cuda_available"],
        "rtx_2070_super_recognized": device.get("name") == "NVIDIA GeForce RTX 2070 SUPER",
        "checkpoint_load": preliminary["checkpoint"]["status"] == "PASS",
        "optimizer_device_migration": (
            optimizer_gpu_devices and all(value.startswith("cuda") for value in optimizer_gpu_devices)
        ),
        "scheduler_continuity": preliminary["checkpoint"]["scheduler"]["learning_rate"] == 1e-4,
        "rng_sampler_continuity": (
            preliminary["checkpoint"]["duplicate_data_prevented"]
            and resume["status"] == "PASS"
        ),
        "numerical_parity": numerical["status"] == "PASS",
        "one_step_parity": one_step["status"] == "PASS",
        "gpu_save_reload": resume["status"] == "PASS",
        "gpu_to_cpu_checkpoint": all(resume["gpu_to_cpu"].values()),
        "vram_safe": vram_safe,
        "gpu_utilization_confirmed": gpu_utilization_confirmed,
        "no_nan_inf": no_nan_inf,
    }
    migration_pass = all(gate_checks.values())
    speed_pass = speedup > 1.0
    amp_speedup = (
        amp_benchmark["tokens_per_second"] / gpu_benchmark["tokens_per_second"]
        if amp_benchmark else None
    )
    report = {
        "schema": "gpu-migration-report-v1",
        "phase": 39,
        "preflight": preliminary,
        "pytorch_selection": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "wheel_source": "https://download.pytorch.org/whl/cu130",
            "reason": "official stable patch matching the repository torch==2.12.1 pin",
        },
        "cpu_checkpoint_to_gpu": "PASS",
        "optimizer_gpu_state_devices": optimizer_gpu_devices,
        "numerical_parity": numerical,
        "one_step_parity": one_step,
        "resume_safety": resume,
        "benchmark": {
            "cpu_fp32": cpu_benchmark,
            "gpu_fp32": gpu_benchmark,
            "speedup": speedup,
            "speed_gate": "PASS" if speed_pass else "FAIL",
        },
        "amp": {
            "tested": bool(amp_benchmark),
            "benchmark": amp_benchmark,
            "speedup_vs_gpu_fp32": amp_speedup,
            "adopted": False,
            "reason": "diagnostic only; formal PHASE 39 keeps the FP32 safety baseline",
        },
        "peak_vram_mb": gpu_benchmark["peak_vram_mb"],
        "total_vram_mb": total_vram_mb,
        "gate_checks": gate_checks,
        "gpu_migration_gate": "GPU_MIGRATION_PASS" if migration_pass else "GPU_MIGRATION_FAIL",
        "speed_gate": "PASS" if speed_pass else "FAIL",
        "formal_training_allowed": migration_pass and speed_pass,
    }
    return report


def report_markdown(report: dict) -> str:
    cpu = report["benchmark"]["cpu_fp32"]
    gpu = report["benchmark"]["gpu_fp32"]
    amp = report["amp"]
    return (
        "# PHASE 39 GPU Migration\n\n"
        f"GPU Migration Gate: **{report['gpu_migration_gate']}**\n"
        f"Speed Gate: **{report['speed_gate']}**\n\n"
        f"- GPU: {report['preflight']['device']['name']}\n"
        f"- PyTorch: {report['pytorch_selection']['version']}\n"
        f"- CUDA runtime: {report['pytorch_selection']['cuda_runtime']}\n"
        f"- Numerical parity: {report['numerical_parity']['status']}\n"
        f"- One-step parity: {report['one_step_parity']['status']}\n"
        f"- Resume / GPU-to-CPU: {report['resume_safety']['status']}\n"
        f"- CPU FP32: {cpu['tokens_per_second']:.2f} tok/s\n"
        f"- GPU FP32: {gpu['tokens_per_second']:.2f} tok/s\n"
        f"- Speedup: {report['benchmark']['speedup']:.2f}x\n"
        f"- Peak VRAM: {report['peak_vram_mb']:.1f} MiB\n"
        f"- AMP tested/adopted: {'YES' if amp['tested'] else 'NO'}/"
        f"{'YES' if amp['adopted'] else 'NO'}\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=50)
    parser.add_argument("--skip-amp", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_json(SETTINGS_PATH)
    preliminary = preflight(settings)
    write_json(ROOT / "evaluation/gpu-migration-preflight.json", preliminary)
    if preliminary["status"] != "PASS":
        print(json.dumps(preliminary, indent=2))
        return 1
    if args.preflight_only:
        print(json.dumps(preliminary, indent=2))
        return 0
    report = run_validation(args.benchmark_steps, test_amp=not args.skip_amp)
    write_json(ROOT / "evaluation/gpu-migration-report.json", report)
    (ROOT / "evaluation/gpu-migration-report.md").write_text(
        report_markdown(report), encoding="utf-8"
    )
    print(json.dumps({
        "gpu_migration_gate": report["gpu_migration_gate"],
        "speed_gate": report["speed_gate"],
        "formal_training_allowed": report["formal_training_allowed"],
        "cpu_tokens_per_second": report["benchmark"]["cpu_fp32"]["tokens_per_second"],
        "gpu_tokens_per_second": report["benchmark"]["gpu_fp32"]["tokens_per_second"],
        "speedup": report["benchmark"]["speedup"],
    }, indent=2))
    return 0 if report["formal_training_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
