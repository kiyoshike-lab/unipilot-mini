from __future__ import annotations

from pathlib import Path

import pytest
import torch

from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.device import (
    move_optimizer_state_to_device,
    optimizer_state_devices,
    resolve_device,
)
from training.optimizer import create_optimizer
from training.train_foundation_continue import (
    APPROVED_TARGET_TOKENS,
    START_TOKENS,
    default_resume_path,
    validate_target_tokens,
    verify_phase38_gate,
)
from training.train_foundation_v21_ab import (
    load_json,
    restore_random_state,
    save_checkpoint,
)
from training.train_foundation_v22_current import preflight_resume


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def tiny_model(device: str = "cpu") -> DiagnosticTransformerV17:
    model = DiagnosticTransformerV17(DiagnosticConfigV17(
        model_name="PHASE 39 GPU migration unit",
        vocab_size=64,
        context_length=16,
        embedding_dim=16,
        n_layers=1,
        n_heads=2,
        ffn_dim=32,
        dropout=0,
    ))
    return model.to(device)


def step(model: DiagnosticTransformerV17, optimizer: torch.optim.Optimizer) -> float:
    device = next(model.parameters()).device
    inputs = torch.arange(16, device=device).remainder(64)[None]
    targets = torch.arange(1, 17, device=device).remainder(64)[None]
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(inputs, targets)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return float(loss.detach())


def test_device_selection_supports_cpu_auto_and_rejects_unavailable_cuda(monkeypatch) -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_phase39_target_and_resume_points_are_gate_locked() -> None:
    settings = load_json("configs/unipilot-foundation-v28.json")
    gate = verify_phase38_gate()
    assert gate["final_tokens"] == START_TOKENS
    assert gate["final_gate"] == "CONTINUE_15M_GENERATION_LAG"
    assert validate_target_tokens(APPROVED_TARGET_TOKENS) == 15_360_000
    with pytest.raises(ValueError, match="authorizes exactly"):
        validate_target_tokens(APPROVED_TARGET_TOKENS + 512)
    for seed in settings["seeds"]:
        audit = preflight_resume(settings, seed, default_resume_path(seed))
        assert audit["status"] == "PASS"
        assert audit["tokens_processed"] == START_TOKENS
        assert audit["cuda_rng_state_present"] is False


@CUDA
def test_optimizer_state_migrates_to_cuda() -> None:
    model = tiny_model("cpu")
    optimizer = create_optimizer(model, 1e-4, .1)
    step(model, optimizer)
    model.to("cuda")
    move_optimizer_state_to_device(optimizer, "cuda")
    assert optimizer_state_devices(optimizer)
    assert all(value.startswith("cuda") for value in optimizer_state_devices(optimizer))


@CUDA
def test_legacy_cpu_rng_is_backward_compatible_with_cuda() -> None:
    state = {
        "python": __import__("random").getstate(),
        "numpy": __import__("numpy").random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    action = restore_random_state(state, "cuda", cuda_seed=42)
    assert action == "initialized_from_seed_for_legacy_cpu_checkpoint"


@CUDA
def test_cpu_checkpoint_loads_on_gpu_and_gpu_checkpoint_loads_on_cpu(tmp_path: Path) -> None:
    torch.manual_seed(39)
    cpu_model = tiny_model("cpu")
    cpu_optimizer = create_optimizer(cpu_model, 1e-4, .1)
    step(cpu_model, cpu_optimizer)
    permutation = torch.randperm(32, generator=torch.Generator().manual_seed(42))
    cpu_checkpoint = tmp_path / "cpu.pt"
    save_checkpoint(
        cpu_checkpoint,
        model=cpu_model,
        optimizer=cpu_optimizer,
        variant="current",
        seed=42,
        update=1,
        permutation=permutation,
        history=[],
        training_seconds=.1,
        settings={"maximum_allowed_tokens_per_run": 15360000},
    )

    payload = torch.load(cpu_checkpoint, map_location="cpu", weights_only=False)
    gpu_model = tiny_model("cuda")
    gpu_model.load_state_dict(payload["model_state"], strict=True)
    gpu_optimizer = create_optimizer(gpu_model, 1e-4, .1)
    gpu_optimizer.load_state_dict(payload["optimizer_state"])
    move_optimizer_state_to_device(gpu_optimizer, "cuda")
    assert all(value.startswith("cuda") for value in optimizer_state_devices(gpu_optimizer))
    step(gpu_model, gpu_optimizer)

    gpu_checkpoint = tmp_path / "gpu.pt"
    metadata = save_checkpoint(
        gpu_checkpoint,
        model=gpu_model,
        optimizer=gpu_optimizer,
        variant="current",
        seed=42,
        update=2,
        permutation=permutation,
        history=[],
        training_seconds=.2,
        settings={"maximum_allowed_tokens_per_run": 15360000},
    )
    assert metadata["integrity"] == "PASS"
    restored = torch.load(gpu_checkpoint, map_location="cpu", weights_only=False)
    fallback = tiny_model("cpu")
    fallback.load_state_dict(restored["model_state"], strict=True)
    fallback_optimizer = create_optimizer(fallback, 1e-4, .1)
    fallback_optimizer.load_state_dict(restored["optimizer_state"])
    assert optimizer_state_devices(fallback_optimizer) == ["cpu"]
    assert "torch_cuda" in restored["random_state"]
    assert restored["tokens_processed"] == 2 * 512


@CUDA
def test_cpu_gpu_numerical_and_one_step_parity() -> None:
    torch.manual_seed(39)
    cpu_model = tiny_model("cpu")
    gpu_model = tiny_model("cuda")
    gpu_model.load_state_dict(cpu_model.state_dict(), strict=True)
    inputs = torch.arange(16).remainder(64)[None]
    targets = torch.arange(1, 17).remainder(64)[None]
    cpu_model.eval()
    gpu_model.eval()
    cpu_logits, cpu_loss = cpu_model(inputs, targets)
    gpu_logits, gpu_loss = gpu_model(inputs.cuda(), targets.cuda())
    assert torch.allclose(cpu_logits, gpu_logits.cpu(), atol=1e-4, rtol=1e-4)
    assert abs(float(cpu_loss.detach()) - float(gpu_loss.detach())) <= 1e-4

    cpu_optimizer = create_optimizer(cpu_model, 1e-4, .1)
    gpu_optimizer = create_optimizer(gpu_model, 1e-4, .1)
    cpu_step_loss = step(cpu_model, cpu_optimizer)
    gpu_step_loss = step(gpu_model, gpu_optimizer)
    assert abs(cpu_step_loss - gpu_step_loss) <= 1e-4
    for cpu_parameter, gpu_parameter in zip(cpu_model.parameters(), gpu_model.parameters()):
        assert torch.allclose(cpu_parameter, gpu_parameter.cpu(), atol=2e-4, rtol=2e-4)
