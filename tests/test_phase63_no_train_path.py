"""CPU-only contract tests for the dedicated PHASE63O NO_TRAIN path.

No test imports torch, touches CUDA, acquires a runtime lock, reads a model
checkpoint, or opens a training dataset.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from training.phase63_no_train import NoTrainContractError, SingleForward, execute_once


class FakeThermalGuard:
    def __init__(self, trace): self.trace = trace
    def wait_for_precheck(self): self.trace.append("THERMAL_PRECHECK")


class FakeWatchdog:
    def __init__(self, guard, trace): self.trace = trace
    def start(self): self.trace.append("WATCHDOG_START")
    def finish(self): self.trace.append("WATCHDOG_FINISH"); return {"final_state": "COMPLETE"}


class FakeModel:
    def __init__(self, trace): self.trace = trace; self.training = True; self.forward_calls = 0
    def eval(self): self.trace.append("MODEL_EVAL"); self.training = False
    def __call__(self, value): self.trace.append("FORWARD"); self.forward_calls += 1; return {"logits": value}


def test_no_train_happy_path_is_one_eval_inference_forward_without_training_dependencies():
    trace, model = [], FakeModel([])
    model.trace = trace

    @contextmanager
    def inference_context():
        trace.append("INFERENCE_ENTER")
        yield
        trace.append("INFERENCE_EXIT")

    result = execute_once(
        lock_attestor=lambda: trace.append("LOCK_STAGE"),
        thermal_guard=FakeThermalGuard(trace),
        watchdog_factory=lambda guard: FakeWatchdog(guard, trace),
        model_loader=lambda: model,
        synthetic_input_factory=lambda: "SYNTHETIC_NO_TRAIN_FIXTURE",
        inference_context_factory=inference_context,
        output_validator=lambda value: trace.append("OUTPUT_VALIDATED") if value == {"logits": "SYNTHETIC_NO_TRAIN_FIXTURE"} else (_ for _ in ()).throw(AssertionError()),
    )

    assert result["forward_calls"] == 1
    assert result["input_provenance"] == "SYNTHETIC_NO_TRAIN_FIXTURE"
    assert model.forward_calls == 1 and model.training is False
    assert trace == ["LOCK_STAGE", "THERMAL_PRECHECK", "WATCHDOG_START", "MODEL_EVAL",
                     "INFERENCE_ENTER", "FORWARD", "INFERENCE_EXIT", "OUTPUT_VALIDATED", "WATCHDOG_FINISH"]


def test_second_forward_is_rejected_by_the_one_forward_limiter():
    model = FakeModel([]); limiter = SingleForward()
    assert limiter.call(model, "first") == {"logits": "first"}
    with pytest.raises(NoTrainContractError, match="FORWARD_LIMIT"):
        limiter.call(model, "second")
    assert model.forward_calls == 1


def test_no_train_path_has_no_optimizer_dataloader_or_checkpoint_dependency():
    source = __import__("training.phase63_no_train", fromlist=["*"])
    text = open(source.__file__, encoding="utf-8").read()
    for forbidden in ("create_optimizer", "optimizer.step", "DataLoader", "macro_batch", "torch.save", "backward"):
        assert forbidden not in text


def test_worker_no_train_branch_isolated_from_training_functions():
    from pathlib import Path
    text = (Path(__file__).parents[1] / "training" / "phase63_worker.py").read_text(encoding="utf-8")
    branch = text[text.index("def no_train("):text.index("def train(")]
    for forbidden in ("create_optimizer", "strict_parent(", "macro_batch(", "weighted_lm_loss(", "optimizer.step", "save("):
        assert forbidden not in branch
    assert "execute_no_train_once" in branch and "load_model_weights_only" in branch


def test_no_train_supervisor_entrypoint_has_no_six_run_dispatch():
    from pathlib import Path
    text = (Path(__file__).parents[1] / "scripts" / "run_phase63_study.py").read_text(encoding="utf-8")
    section = text[text.index("def run_no_train_once("):text.index("def main()")]
    assert "RUN_ORDER" not in section
    assert '"no-train"' in section and "admit_execution" in section and "execute(" in section


def test_no_train_pre_model_gate_runs_after_lock_and_before_model_load():
    trace, model = [], FakeModel([])
    model.trace = trace
    execute_once(
        lock_attestor=lambda: trace.append("LOCK"),
        pre_model_gate=lambda: trace.append("CUDA_STAGE"),
        thermal_guard=FakeThermalGuard(trace),
        watchdog_factory=lambda guard: FakeWatchdog(guard, trace),
        model_loader=lambda: trace.append("MODEL_LOAD") or model,
        synthetic_input_factory=lambda: "fixture",
        inference_context_factory=lambda: __import__("contextlib").nullcontext(),
        output_validator=lambda value: None,
    )
    assert trace.index("LOCK") < trace.index("THERMAL_PRECHECK") < trace.index("CUDA_STAGE") < trace.index("MODEL_LOAD") < trace.index("FORWARD")
