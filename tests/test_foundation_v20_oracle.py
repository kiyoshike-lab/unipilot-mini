from __future__ import annotations

import random

import pytest

from foundation.synthetic_context_oracle_v31 import (
    oracle_answer_v31,
    parse_key_lookup_v31,
)
from foundation.synthetic_context_v3 import (
    LEVEL_PAIRS,
    REMOVED,
    example_hash,
    key_lookup_example_v3,
    mapping_hash,
    removed_relation,
)


@pytest.mark.parametrize("level", range(6))
def test_independent_oracle_is_exact_for_every_level(level):
    rng = random.Random(31_000 + level)
    for _ in range(100):
        sequence, answer, metadata = key_lookup_example_v3(
            rng, level, markers=True, split="any"
        )
        parsed = parse_key_lookup_v31(sequence)
        assert parsed.answer == answer
        assert parsed.query == metadata["mapping"][metadata["selected_pair_index"]][0]
        assert oracle_answer_v31(sequence) == answer


def test_oracle_does_not_use_generator_metadata():
    sequence, answer, metadata = key_lookup_example_v3(random.Random(91), 4)
    metadata["mapping"] = [[999, 998]]
    metadata["selected_pair_index"] = 999
    assert oracle_answer_v31(sequence) == answer


def test_oracle_rejects_removed_relevant_relation():
    example = key_lookup_example_v3(random.Random(92), 3)
    removed = removed_relation(example)
    assert REMOVED in removed[0]
    with pytest.raises(ValueError, match="exactly one relation"):
        oracle_answer_v31(removed[0])


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_train_and_novel_mapping_partitions_are_disjoint(level):
    train_rng = random.Random(32_000 + level)
    test_rng = random.Random(33_000 + level)
    train = [key_lookup_example_v3(train_rng, level, split="train") for _ in range(200)]
    novel = [key_lookup_example_v3(test_rng, level, split="heldout") for _ in range(200)]
    assert not ({example_hash(row) for row in train} & {example_hash(row) for row in novel})
    assert not ({mapping_hash(row) for row in train} & {mapping_hash(row) for row in novel})
    train_relations = {tuple(pair) for row in train for pair in row[2]["mapping"]}
    novel_relations = {tuple(pair) for row in novel for pair in row[2]["mapping"]}
    assert not (train_relations & novel_relations)
    assert all(oracle_answer_v31(row[0]) == row[1] for row in train + novel)


def test_v31_keeps_v3_pair_counts_and_marked_sequence_format():
    for level, pairs in LEVEL_PAIRS.items():
        sequence, _, metadata = key_lookup_example_v3(random.Random(level), level)
        assert metadata["pairs"] == pairs
        assert len(sequence) == 1 + pairs * 5 + 3
