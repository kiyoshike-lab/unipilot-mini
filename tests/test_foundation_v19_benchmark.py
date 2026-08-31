from __future__ import annotations

import random

import pytest
import torch

from foundation.reference_transformer_v18 import ReferenceConfigV18, ReferenceTransformerV18
from foundation.synthetic_context_v3 import (
    ANSWER,
    LEVEL_PAIRS,
    REMOVED,
    actual_chance,
    ambiguity_audit,
    causal_learnability,
    counterfactual_relation,
    example_hash,
    key_lookup_example_v3,
    make_batch_v3,
    mapping_hash,
    removed_query,
    removed_relation,
    scan_dataset,
    shuffled_relation,
    wrong_query,
    weighted_relation_loss,
)
from training.validate_foundation_v19_benchmark import choose_active_level


@pytest.mark.parametrize("level", range(6))
@pytest.mark.parametrize("markers", [False, True])
def test_v3_is_causally_learnable_and_unambiguous(level, markers):
    example = key_lookup_example_v3(
        random.Random(100 + level), level, markers=markers, split="train"
    )
    assert causal_learnability(example)["pass"] is True
    assert ambiguity_audit(example)["ambiguous"] is False
    sequence, answer, metadata = example
    assert sequence[metadata["answer_position"]] in {ANSWER, 225}
    assert sequence[metadata["correct_value_position"]] == answer


@pytest.mark.parametrize("level", range(6))
def test_actual_chance_uses_unique_candidate_values(level):
    example = key_lookup_example_v3(random.Random(200 + level), level)
    chance = actual_chance(example)
    assert chance["actual_candidate_count"] == LEVEL_PAIRS[level]
    assert chance["candidate_choice_accuracy"] == 1 / LEVEL_PAIRS[level]
    assert chance["full_value_vocab_accuracy"] == 1 / 32


def test_answer_only_and_all_token_masks_are_aligned_to_answer_marker():
    examples = [key_lookup_example_v3(random.Random(seed), 2) for seed in range(4)]
    inputs, answer_targets, answer_weights = make_batch_v3(examples, "answer_only")
    assert torch.all(answer_targets[:, :-1] == -100)
    assert torch.equal(answer_targets[:, -1], torch.tensor([row[1] for row in examples]))
    assert torch.all(answer_weights[:, :-1] == 0)
    _, all_targets, all_weights = make_batch_v3(examples, "all_token", answer_weight=16)
    assert torch.equal(all_targets[:, :-1], inputs[:, 1:])
    assert torch.equal(all_targets[:, -1], answer_targets[:, -1])
    assert torch.all(all_weights[:, :-1] == 1)
    assert torch.all(all_weights[:, -1] == 16)


def test_counterfactual_changes_relation_and_target_together():
    original = key_lookup_example_v3(random.Random(301), 3)
    changed = counterfactual_relation(original)
    position = original[2]["correct_value_position"]
    assert changed[0][position] == changed[1]
    assert changed[1] != original[1]
    assert changed[0][:position] == original[0][:position]


def test_query_and_relation_ablations_change_only_intended_context():
    original = key_lookup_example_v3(random.Random(302), 3)
    wrong_original_target = wrong_query(original)
    wrong_new_target = wrong_query(original, target_new=True)
    assert wrong_original_target[0] == wrong_new_target[0]
    assert wrong_original_target[1] == original[1]
    assert wrong_new_target[1] != original[1]
    removed_q = removed_query(original)
    assert removed_q[0][original[2]["query_key_position"]] == REMOVED
    removed_r = removed_relation(original)
    assert removed_r[0][original[2]["correct_key_position"]] == REMOVED
    assert removed_r[0][original[2]["correct_value_position"]] == REMOVED
    shuffled = shuffled_relation(original)
    assert shuffled[1] == original[1]
    assert shuffled[0] != original[0]


def test_dataset_is_deterministic_and_train_test_exact_disjoint():
    def build(seed, split):
        rng = random.Random(seed)
        return [key_lookup_example_v3(rng, level, split=split) for level in range(1, 6) for _ in range(40)]
    train_a = build(900, "train")
    train_b = build(900, "train")
    heldout = build(901, "heldout")
    assert [example_hash(row) for row in train_a] == [example_hash(row) for row in train_b]
    assert not ({example_hash(row) for row in train_a} & {example_hash(row) for row in heldout})
    train_pairs = {pair for row in train_a for pair in map(tuple, row[2]["mapping"])}
    heldout_pairs = {pair for row in heldout for pair in map(tuple, row[2]["mapping"])}
    assert not (train_pairs & heldout_pairs)
    assert scan_dataset(train_a)["causal_failures"] == 0
    assert scan_dataset(train_a)["ambiguity_count"] == 0
    assert len({mapping_hash(row) for row in train_a}) > 100


def test_seen_and_unseen_token_generalization_pools_are_disjoint():
    seen = [
        key_lookup_example_v3(random.Random(seed), 2, split="any", token_split="seen")
        for seed in range(20)
    ]
    unseen = [
        key_lookup_example_v3(random.Random(seed), 2, split="any", token_split="unseen")
        for seed in range(20)
    ]
    seen_keys = {key for row in seen for key, _ in row[2]["mapping"]}
    unseen_keys = {key for row in unseen for key, _ in row[2]["mapping"]}
    seen_values = {value for row in seen for _, value in row[2]["mapping"]}
    unseen_values = {value for row in unseen for _, value in row[2]["mapping"]}
    assert not (seen_keys & unseen_keys)
    assert not (seen_values & unseen_values)


def test_unseen_token_pool_rejects_level_beyond_its_capacity():
    with pytest.raises(ValueError, match="exceed unseen token pool capacity"):
        key_lookup_example_v3(random.Random(1), 4, token_split="unseen")


def test_vocabulary_stages_preserve_task_grammar_and_expand_token_pool():
    small = [
        key_lookup_example_v3(random.Random(seed), 2, vocabulary_stage="small")
        for seed in range(50)
    ]
    full = [
        key_lookup_example_v3(random.Random(seed), 2, vocabulary_stage="full")
        for seed in range(50)
    ]
    assert all(row[2]["pairs"] == 2 and row[2]["markers"] for row in small + full)
    small_tokens = {token for row in small for pair in row[2]["mapping"] for token in pair}
    full_tokens = {token for row in full for pair in row[2]["mapping"] for token in pair}
    assert small_tokens < full_tokens


def test_tiny_reference_converges_on_minimal_fixed_relation_and_reloads_strictly():
    torch.manual_seed(77)
    config = ReferenceConfigV18(
        model_name="v3 tiny convergence",
        vocab_size=256,
        context_length=16,
        embedding_dim=32,
        n_layers=2,
        n_heads=4,
        ffn_dim=64,
        dropout=0.0,
        residual_projection_init_scale=1 / 2,
    )
    model = ReferenceTransformerV18(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    example = key_lookup_example_v3(random.Random(1), 0)
    inputs, targets, weights = make_batch_v3([example] * 8)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = weighted_relation_loss(model, inputs, targets, weights)
        loss.backward()
        optimizer.step()
    assert logits[:, -1].argmax(-1).eq(example[1]).all()
    restored = ReferenceTransformerV18(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.inference_mode():
        reloaded_logits, _ = restored(inputs)
    assert reloaded_logits[:, -1].argmax(-1).eq(example[1]).all()


def test_zero_replay_does_not_consume_dataset_rng_state():
    rng = random.Random(144)
    before = rng.getstate()
    curriculum = {
        "replay_probability": 0,
        "relation_replay_probability": 0,
        "relation_replay_start_update_by_level": [0] * 6,
    }
    assert choose_active_level(rng, 2, 100, curriculum) == 2
    assert rng.getstate() == before
