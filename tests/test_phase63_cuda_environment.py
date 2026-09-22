from pathlib import Path


def test_phase63a_receipt_is_non_training_and_never_allowlists_c_plus_g():
    source = (Path(__file__).parents[1] / "scripts" / "phase63_cuda_environment.py").read_text(encoding="utf-8")
    assert "optimizer_steps\": 0" in source
    assert "training\": False" in source
    assert "CUDA_PROCESS_PREFLIGHT_REQUIRES_MANUAL_REVIEW" in source
    assert "NOT_RUN_C+G_MANUAL_REVIEW" in source


def test_phase63a_uses_exact_interpreter_contract_in_supervisor_and_worker():
    supervisor = (Path(__file__).parents[1] / "scripts" / "run_phase63_study.py").read_text(encoding="utf-8")
    worker = (Path(__file__).parents[1] / "training" / "phase63_worker.py").read_text(encoding="utf-8")
    assert '"executable": str(Path(sys.executable).resolve())' in supervisor
    assert "WORKER_INTERPRETER_BINDING_MISMATCH" in worker
