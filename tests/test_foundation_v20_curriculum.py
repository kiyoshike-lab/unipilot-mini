from __future__ import annotations

import copy
import json
import random

import torch

from foundation.reference_transformer_v18 import ReferenceConfigV18, ReferenceTransformerV18
from foundation.synthetic_context_oracle_v31 import oracle_answer_v31
from foundation.synthetic_context_v3 import (
    counterfactual_relation,
    example_hash,
    key_lookup_example_v3,
    mapping_hash,
    removed_query,
    removed_relation,
    shuffled_relation,
    wrong_query,
)
from training.validate_foundation_v20_curriculum import (
    FORMAL_PARAMETERS,
    build_model,
    create_synthetic_optimizer,
    deterministic_examples,
    load_json,
    model_spec,
    reference_gate_ready,
    sample_complexity,
    train_update,
)


def test_formal_reference_has_required_parameter_count_and_optimizer_settings():
    settings = load_json("configs/unipilot-foundation-v20.json")
    model = build_model(settings, model_spec(settings, "reference_mha"), 43)
    assert model.parameter_count() == FORMAL_PARAMETERS == 19_514_880
    optimizer = create_synthetic_optimizer(model, settings)
    assert optimizer.defaults["lr"] == 3e-4
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["eps"] == 1e-8
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.01}


def test_controls_have_oracle_expected_semantics():
    original = key_lookup_example_v3(random.Random(311), 4, split="heldout")
    counterfactual = counterfactual_relation(original)
    wrong_new = wrong_query(original, target_new=True)
    shuffled = shuffled_relation(original)
    assert oracle_answer_v31(original[0]) == original[1]
    assert oracle_answer_v31(counterfactual[0]) == counterfactual[1] != original[1]
    assert oracle_answer_v31(wrong_new[0]) == wrong_new[1] != original[1]
    assert oracle_answer_v31(shuffled[0]) != shuffled[1]
    for ablated in (removed_query(original), removed_relation(original)):
        try:
            oracle_answer_v31(ablated[0])
        except ValueError:
            pass
        else:
            raise AssertionError("ablated context unexpectedly remained oracle-resolvable")


def test_canonical_any_split_can_hold_out_exact_mapping_combinations():
    rng = random.Random(314)
    training = [key_lookup_example_v3(rng, 3, split="any") for _ in range(500)]
    heldout = deterministic_examples(
        3,
        100,
        315,
        "any",
        excluded_hashes={example_hash(row) for row in training},
        excluded_mapping_hashes={mapping_hash(row) for row in training},
    )
    assert not ({mapping_hash(row) for row in training} & {mapping_hash(row) for row in heldout})


def test_sample_complexity_reports_only_observed_evaluation_points():
    curve = [
        {"accuracy": 0.49, "updates": 100, "examples_processed": 1600, "tokens_processed": 38400},
        {"accuracy": 0.91, "updates": 500, "examples_processed": 8000, "tokens_processed": 192000},
    ]
    measured = sample_complexity(curve)
    assert measured["0.50"] == {"updates": 500, "examples": 8000, "tokens": 192000}
    assert measured["0.90"] == measured["0.50"]
    assert measured["0.95"] is None


def test_reference_first_gate_requires_both_standalone_and_sequential_pass(tmp_path):
    (tmp_path / "reference_mha-l3-standalone.json").write_text(
        json.dumps({"pass": True}), encoding="utf-8"
    )
    (tmp_path / "reference_mha-l4-standalone.json").write_text(
        json.dumps({"pass": False}), encoding="utf-8"
    )
    (tmp_path / "reference_mha-sequential.json").write_text(
        json.dumps({"validity_gate": {"pass": True}}), encoding="utf-8"
    )
    assert reference_gate_ready(tmp_path) is False
    (tmp_path / "reference_mha-l4-standalone.json").write_text(
        json.dumps({"pass": True}), encoding="utf-8"
    )
    assert reference_gate_ready(tmp_path) is True


def tiny_reference(seed: int) -> ReferenceTransformerV18:
    torch.manual_seed(seed)
    return ReferenceTransformerV18(ReferenceConfigV18(
        model_name="v3.1 resume reproducibility",
        vocab_size=256,
        context_length=48,
        embedding_dim=32,
        n_layers=2,
        n_heads=4,
        ffn_dim=64,
        dropout=0.0,
        residual_projection_init_scale=0.5,
    ))


def test_resume_state_reproduces_next_update_exactly():
    model = tiny_reference(312)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    rng = random.Random(313)
    first = [key_lookup_example_v3(rng, 3, split="train") for _ in range(4)]
    train_update(model, optimizer, first, 1.0)
    snapshot = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "python_rng": rng.getstate(),
        "torch_rng": torch.get_rng_state().clone(),
    }
    continued_batch = [key_lookup_example_v3(rng, 3, split="train") for _ in range(4)]
    continued_loss, _ = train_update(model, optimizer, continued_batch, 1.0)

    restored = tiny_reference(999)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-3, betas=(0.9, 0.95))
    restored.load_state_dict(snapshot["model"], strict=True)
    restored_optimizer.load_state_dict(snapshot["optimizer"])
    resumed_rng = random.Random()
    resumed_rng.setstate(snapshot["python_rng"])
    torch.set_rng_state(snapshot["torch_rng"])
    resumed_batch = [key_lookup_example_v3(resumed_rng, 3, split="train") for _ in range(4)]
    resumed_loss, _ = train_update(restored, restored_optimizer, resumed_batch, 1.0)

    assert [example_hash(row) for row in continued_batch] == [example_hash(row) for row in resumed_batch]
    assert continued_loss == resumed_loss
    assert all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), restored.state_dict().values())
    )
