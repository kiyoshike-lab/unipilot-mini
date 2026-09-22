"""Offline contract checks for the PHASE63 exclusive CUDA path."""
from pathlib import Path

from training import phase63_worker as worker


def test_phase63_registry_is_exact_and_fresh():
    assert worker.SEEDS == (42, 123, 2026)
    assert worker.ARMS == {"control": 5e-5, "half-lr": 2.5e-5}
    assert worker.output(42, "control").as_posix().endswith(
        "experimental/phase63/continuation-stability/control/seed-42/checkpoint-tokens-16446464.pt"
    )


def test_phase63_worker_cannot_be_a_direct_legacy_phase61_launcher():
    source = Path(worker.__file__).read_text(encoding="utf-8")
    assert "run_foundation_v50_phase61" not in source
    assert "phase61_runtime_lr_contract" not in source
    assert "phase63_runtime_lr_contract" in source


def test_phase63_supervisor_routes_each_cuda_kind_through_exclusive_runner():
    source = (Path(__file__).parents[1] / "scripts" / "run_phase63_study.py").read_text(encoding="utf-8")
    assert "from training.exclusive_cuda_runner import execute, is_complete" in source
    assert "display=()" in source
    assert "C+G is deliberately *not* treated as a graphics allowlist" in source
    assert "phase63_worker" in source
    assert "interpreter_metadata" in source
    assert "sys.executable" in source
