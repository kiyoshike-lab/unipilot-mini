import json
import math

from evaluation import analyze_foundation_v34_lm_gate as analysis
from evaluation.run_foundation_v34_thermal_diagnostic import active


def _load(name: str):
    return json.loads((analysis.ROOT / "evaluation" / name).read_text(encoding="utf-8"))


def test_deterministic_repeats_and_phase44_recomputation_match():
    diagnostics = _load("foundation-v34-determinism-and-rare.json")
    assert diagnostics["deterministic_evaluation"]["pass"]
    assert diagnostics["checkpoint_integrity"]["unchanged"]
    for stage in analysis.STAGES:
        for seed in analysis.SEEDS:
            assert diagnostics["trajectory"][stage][str(seed)]["matches_phase44"]["pass"]


def test_rare_ce_probability_identity_and_distribution_resolution():
    summary = _load("foundation-v34-lm-gate-review-summary.json")
    gate1 = summary["mean_std_trajectory"]["gate1"]
    gate2 = summary["mean_std_trajectory"]["gate2"]
    assert gate2["rare_cross_entropy"]["mean"] < gate1["rare_cross_entropy"]["mean"]
    assert gate2["rare_mean_probability"]["mean"] < gate1["rare_mean_probability"]["mean"]
    assert gate2["rare_median_probability"]["mean"] > gate1["rare_median_probability"]["mean"]
    diagnostics = _load("foundation-v34-determinism-and-rare.json")
    for stage in analysis.STAGES:
        for seed in analysis.SEEDS:
            rare = diagnostics["trajectory"][stage][str(seed)]["metrics"][
                "frequency_buckets"
            ]["rare_bottom_20_percent"]
            assert math.isclose(
                math.exp(-rare["cross_entropy"]),
                rare["geometric_mean_correct_token_probability"],
                rel_tol=1e-12,
            )


def test_existing_eos_recipe_control_supports_weight_1_5():
    recipe = _load("foundation-v34-recipe-comparison.json")
    assert not recipe["new_recipe_pilot_executed"]
    assert recipe["adoption_criterion_pass"]
    assert recipe["recommended_eos_weight"] == 1.5
    assert recipe["delta_eos_1_5_minus_standard"]["terminal_eos_probability"] > 0
    assert abs(recipe["delta_eos_1_5_minus_standard"]["validation_loss"]) < 0.001


def test_gate2_failure_is_seed_local_not_global_plateau():
    summary = _load("foundation-v34-lm-gate-review-summary.json")
    reproduction = summary["gate2_failure_reproduction"]
    assert reproduction["validation_loss_worse_seeds"] == [42]
    assert reproduction["top1_materially_worse_seeds"] == [42]
    assert reproduction["semantic_worse_seeds"] == [123]
    assert not reproduction["all_seed_common"]
    assert summary["plateau_classification"] == "HEALTHY_SHORT_TERM_VARIANCE"
    assert summary["next_phase_gate"] == "CONTINUE_SHORT_GPU_GATES_EOS_1_5"
    assert summary["recommended_next_interval_tokens"] == 256_000
    assert not summary["continue_20m_permission"]


def test_thermal_reason_parser_and_recorded_classification():
    assert active("Active")
    assert active("0x0000000000000020")
    assert not active("Not Active")
    assert not active("0x0000000000000000")
    thermal = _load("foundation-v34-thermal.json")
    assert thermal["summary"]["classification"] == "THERMAL_THROTTLING_OBSERVED"
    assert thermal["summary"]["thermal_throttling_observed"]
    assert thermal["checkpoint_unchanged"]
    assert not thermal["settings_changed"]
    assert thermal["optimizer_steps"] == 0
