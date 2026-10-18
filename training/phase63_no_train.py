"""One-forward, dependency-injected PHASE63 NO_TRAIN control flow.

This module deliberately imports neither torch nor training machinery.  The
future CUDA worker supplies only a weights-only loader, a synthetic input, an
inference context, and the production thermal guard/watchdog.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ContextManager


class NoTrainContractError(RuntimeError):
    pass


@dataclass
class SingleForward:
    calls: int = 0

    def call(self, model: Callable[[Any], Any], value: Any) -> Any:
        if self.calls != 0:
            raise NoTrainContractError("NO_TRAIN_FORWARD_LIMIT_EXCEEDED")
        self.calls += 1
        return model(value)


def execute_once(*, lock_attestor: Callable[[], Any], thermal_guard: Any, watchdog_factory: Callable[[Any], Any],
                 model_loader: Callable[[], Any], synthetic_input_factory: Callable[[], Any],
                 inference_context_factory: Callable[[], ContextManager[Any]],
                 output_validator: Callable[[Any], None], pre_model_gate: Callable[[], None] = lambda: None) -> dict[str, Any]:
    """Execute exactly one inference-only forward with no training dependencies."""
    lock_attestor()
    thermal_guard.wait_for_precheck()
    pre_model_gate()
    watchdog = watchdog_factory(thermal_guard)
    watchdog.start()
    limiter = SingleForward()
    model = None
    try:
        model = model_loader()
        model.eval()
        if getattr(model, "training", None) is not False:
            raise NoTrainContractError("NO_TRAIN_MODEL_NOT_EVAL")
        value = synthetic_input_factory()
        with inference_context_factory():
            output = limiter.call(model, value)
        output_validator(output)
        telemetry = watchdog.finish()
    except BaseException:
        watchdog.finish()
        raise
    if telemetry.get("final_state") == "ABORTED":
        raise NoTrainContractError("NO_TRAIN_THERMAL_ABORT")
    return {"forward_calls": limiter.calls, "model": model, "telemetry": telemetry,
            "input_provenance": "SYNTHETIC_NO_TRAIN_FIXTURE"}
