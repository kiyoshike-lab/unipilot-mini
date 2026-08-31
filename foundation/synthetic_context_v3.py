from __future__ import annotations

from collections import Counter
import hashlib
import json
import random

import torch
from torch.nn import functional as F


VOCAB_SIZE = 256
TASK_KEY_LOOKUP = 2
VALUES = tuple(range(32, 64))
KEYS = tuple(range(64, 80))
PAIR = 227
KEY = 228
VALUE = 229
QUERY = 230
ANSWER = 231
REMOVED = 232
LEGACY_QUERY = 224
LEGACY_ANSWER = 225

LEVEL_PAIRS = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
LEVEL_THRESHOLDS = {0: .99, 1: .99, 2: .98, 3: .95, 4: .90}


def token_label(token_id: int) -> str:
    markers = {
        TASK_KEY_LOOKUP: "<KEY_LOOKUP>",
        PAIR: "<PAIR>", KEY: "<KEY>", VALUE: "<VALUE>",
        QUERY: "<QUERY>", ANSWER: "<ANSWER>", REMOVED: "<REMOVED>",
        LEGACY_QUERY: "<QUERY_LEGACY>", LEGACY_ANSWER: "<ANSWER_LEGACY>",
    }
    if token_id in markers:
        return markers[token_id]
    if token_id in KEYS:
        return f"K{token_id - KEYS[0]:02d}"
    if token_id in VALUES:
        return f"V{token_id - VALUES[0]:02d}"
    return f"T{token_id}"


def mapping_partition(key: int, value: int) -> str:
    """Deterministic held-out pair combinations; every token appears in both splits."""
    return "heldout" if ((key - KEYS[0]) * 7 + value - VALUES[0]) % 5 == 0 else "train"


def _sample_mapping(
    rng: random.Random,
    pairs: int,
    split: str,
    fixed: bool,
    token_split: str,
    vocabulary_stage: str,
) -> list[tuple[int, int]]:
    if fixed:
        return [(KEYS[index], VALUES[index]) for index in range(pairs)]
    if token_split != "shared" and vocabulary_stage != "full":
        raise ValueError("vocabulary staging is only defined for the shared token split")
    if token_split == "shared":
        key_pool, value_pool = KEYS, VALUES
    elif token_split == "seen":
        key_pool, value_pool = KEYS[:12], VALUES[:24]
    elif token_split == "unseen":
        key_pool, value_pool = KEYS[12:], VALUES[24:]
    else:
        raise ValueError(f"unknown token split: {token_split}")
    if vocabulary_stage == "small":
        key_pool = key_pool[:max(pairs, 4)]
        value_pool = value_pool[:max(pairs, 8)]
    elif vocabulary_stage == "medium":
        key_pool = key_pool[:max(pairs, 8)]
        value_pool = value_pool[:max(pairs, 16)]
    elif vocabulary_stage != "full":
        raise ValueError(f"unknown vocabulary stage: {vocabulary_stage}")
    if pairs > len(key_pool) or pairs > len(value_pool):
        raise ValueError(
            f"{pairs} pairs exceed {token_split} token pool capacity "
            f"({len(key_pool)} keys, {len(value_pool)} values)"
        )
    keys = rng.sample(key_pool, pairs)
    values_left = list(value_pool)
    rng.shuffle(values_left)
    mapping = []
    for key in keys:
        candidates = [
            value for value in values_left
            if split == "any" or mapping_partition(key, value) == split
        ]
        if not candidates:
            # Retry the whole bipartite matching with a deterministic child stream.
            return _sample_mapping(
                rng, pairs, split, fixed, token_split, vocabulary_stage
            )
        value = rng.choice(candidates)
        values_left.remove(value)
        mapping.append((key, value))
    return mapping


def key_lookup_example_v3(
    rng: random.Random,
    level: int,
    *,
    markers: bool = True,
    split: str = "train",
    token_split: str = "shared",
    vocabulary_stage: str = "full",
) -> tuple[list[int], int, dict]:
    if level not in LEVEL_PAIRS:
        raise ValueError(f"unknown curriculum level: {level}")
    if split not in {"train", "heldout", "any"}:
        raise ValueError(f"unknown mapping split: {split}")
    if token_split not in {"shared", "seen", "unseen"}:
        raise ValueError(f"unknown token split: {token_split}")
    if vocabulary_stage not in {"small", "medium", "full"}:
        raise ValueError(f"unknown vocabulary stage: {vocabulary_stage}")
    pairs = LEVEL_PAIRS[level]
    mapping = _sample_mapping(
        rng, pairs, split, fixed=level == 0, token_split=token_split,
        vocabulary_stage=vocabulary_stage,
    )
    selected = 0 if level <= 1 else rng.randrange(pairs)
    sequence = [TASK_KEY_LOOKUP]
    key_positions = []
    value_positions = []
    pair_spans = []
    for key, value in mapping:
        start = len(sequence)
        if markers:
            sequence.extend((PAIR, KEY, key, VALUE, value))
            key_positions.append(start + 2)
            value_positions.append(start + 4)
        else:
            sequence.extend((key, value))
            key_positions.append(start)
            value_positions.append(start + 1)
        pair_spans.append((start, len(sequence)))
    query_marker = QUERY if markers else LEGACY_QUERY
    answer_marker = ANSWER if markers else LEGACY_ANSWER
    query_marker_position = len(sequence)
    sequence.extend((query_marker, mapping[selected][0], answer_marker))
    answer_position = len(sequence) - 1
    answer = mapping[selected][1]
    metadata = {
        "difficulty": f"level_{level}_pairs_{pairs}",
        "level": level,
        "pairs": pairs,
        "markers": markers,
        "mapping_split": split,
        "token_split": token_split,
        "vocabulary_stage": vocabulary_stage,
        "mapping": [[key, value] for key, value in mapping],
        "selected_pair_index": selected,
        "all_key_positions": key_positions,
        "all_value_positions": value_positions,
        "pair_spans": [list(span) for span in pair_spans],
        "correct_key_position": key_positions[selected],
        "correct_value_position": value_positions[selected],
        "query_marker_position": query_marker_position,
        "query_key_position": answer_position - 1,
        "answer_position": answer_position,
        "candidate_values": [value for _, value in mapping],
        "human_readable": " ".join(token_label(token) for token in sequence)
        + f" -> {token_label(answer)}",
    }
    return sequence, answer, metadata


def example_hash(example: tuple[list[int], int, dict]) -> str:
    sequence, answer, _ = example
    canonical = json.dumps(
        {"sequence": sequence, "answer": answer}, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def mapping_hash(example: tuple[list[int], int, dict]) -> str:
    mapping = example[2]["mapping"]
    return hashlib.sha256(
        json.dumps(mapping, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def make_batch_v3(
    examples: list[tuple[list[int], int, dict]],
    supervision: str = "answer_only",
    answer_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if supervision not in {"answer_only", "all_token"}:
        raise ValueError(supervision)
    lengths = {len(row[0]) for row in examples}
    if len(lengths) != 1:
        raise RuntimeError("one v3 batch must use a fixed curriculum level/length")
    inputs = torch.tensor([row[0] for row in examples], dtype=torch.long)
    targets = torch.full_like(inputs, -100)
    weights = torch.zeros_like(inputs, dtype=torch.float32)
    if supervision == "all_token":
        targets[:, :-1] = inputs[:, 1:]
        weights[:, :-1] = 1.0
    targets[:, -1] = torch.tensor([row[1] for row in examples])
    weights[:, -1] = float(answer_weight)
    return inputs, targets, weights


def weighted_relation_loss(
    model, inputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, _ = model(inputs)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(targets)
    active = targets != -100
    denominator = weights[active].sum().clamp_min(1e-12)
    return logits, (token_losses * weights).sum() / denominator


def causal_learnability(example: tuple[list[int], int, dict]) -> dict:
    _, _, metadata = example
    answer_position = metadata["answer_position"]
    required = {
        "correct_key": metadata["correct_key_position"],
        "correct_value": metadata["correct_value_position"],
        "query_marker": metadata["query_marker_position"],
        "query_key": metadata["query_key_position"],
    }
    checks = {name: position < answer_position for name, position in required.items()}
    return {
        "answer_prediction_position": answer_position,
        "required_context_positions": required,
        "checks": checks,
        "pass": all(checks.values()),
    }


def ambiguity_audit(example: tuple[list[int], int, dict]) -> dict:
    _, answer, metadata = example
    mapping = metadata["mapping"]
    keys = [row[0] for row in mapping]
    values = [row[1] for row in mapping]
    query_key = mapping[metadata["selected_pair_index"]][0]
    matching = [value for key, value in mapping if key == query_key]
    checks = {
        "duplicate_key": len(keys) != len(set(keys)),
        "duplicate_value": len(values) != len(set(values)),
        "duplicate_pair": len(mapping) != len({tuple(row) for row in mapping}),
        "multiple_answers": len(matching) != 1,
        "answer_mismatch": matching != [answer],
    }
    return {"issues": checks, "ambiguous": any(checks.values())}


def actual_chance(example: tuple[list[int], int, dict]) -> dict:
    candidates = set(example[2]["candidate_values"])
    return {
        "actual_candidate_count": len(candidates),
        "candidate_choice_accuracy": 1 / len(candidates),
        "full_value_vocab_accuracy": 1 / len(VALUES),
    }


def counterfactual_relation(
    example: tuple[list[int], int, dict]
) -> tuple[list[int], int, dict]:
    sequence, old_answer, metadata = example
    sequence = list(sequence)
    metadata = json.loads(json.dumps(metadata))
    blocked = set(metadata["candidate_values"])
    new_answer = next(value for value in VALUES if value not in blocked and value != old_answer)
    index = metadata["selected_pair_index"]
    sequence[metadata["correct_value_position"]] = new_answer
    metadata["mapping"][index][1] = new_answer
    metadata["candidate_values"][index] = new_answer
    metadata["counterfactual_from"] = old_answer
    metadata["counterfactual_to"] = new_answer
    metadata["human_readable"] = " ".join(token_label(token) for token in sequence) + f" -> {token_label(new_answer)}"
    return sequence, new_answer, metadata


def shuffled_relation(example: tuple[list[int], int, dict]):
    sequence, answer, metadata = example
    sequence = list(sequence)
    if metadata["pairs"] == 1:
        return counterfactual_relation(example)[0], answer, dict(metadata)
    values = list(metadata["candidate_values"])
    rotated = values[1:] + values[:1]
    for position, value in zip(metadata["all_value_positions"], rotated):
        sequence[position] = value
    changed = dict(metadata)
    changed["control"] = "shuffled_relation_original_answer_target"
    return sequence, answer, changed


def wrong_query(example: tuple[list[int], int, dict], *, target_new: bool = False):
    sequence, answer, metadata = example
    sequence = list(sequence)
    index = metadata["selected_pair_index"]
    wrong_index = (index + 1) % metadata["pairs"] if metadata["pairs"] > 1 else index
    if metadata["pairs"] == 1:
        wrong_key = next(key for key in KEYS if key != metadata["mapping"][index][0])
        wrong_answer = answer
    else:
        wrong_key, wrong_answer = metadata["mapping"][wrong_index]
    sequence[metadata["query_key_position"]] = wrong_key
    changed = dict(metadata)
    changed["control"] = "wrong_query_new_target" if target_new else "wrong_query_original_target"
    return sequence, (wrong_answer if target_new else answer), changed


def removed_query(example: tuple[list[int], int, dict]):
    sequence, answer, metadata = example
    sequence = list(sequence)
    sequence[metadata["query_key_position"]] = REMOVED
    changed = dict(metadata)
    changed["control"] = "removed_query"
    return sequence, answer, changed


def removed_relation(example: tuple[list[int], int, dict]):
    sequence, answer, metadata = example
    sequence = list(sequence)
    index = metadata["selected_pair_index"]
    sequence[metadata["all_key_positions"][index]] = REMOVED
    sequence[metadata["all_value_positions"][index]] = REMOVED
    changed = dict(metadata)
    changed["control"] = "removed_relevant_relation"
    return sequence, answer, changed


def scan_dataset(examples: list[tuple[list[int], int, dict]]) -> dict:
    ambiguities = [ambiguity_audit(row)["ambiguous"] for row in examples]
    causal = [causal_learnability(row)["pass"] for row in examples]
    exact = Counter(example_hash(row) for row in examples)
    mappings = Counter(mapping_hash(row) for row in examples)
    return {
        "examples": len(examples),
        "causal_failures": causal.count(False),
        "ambiguity_count": ambiguities.count(True),
        "duplicate_exact_sequences": sum(count - 1 for count in exact.values()),
        "unique_mapping_sets": len(mappings),
    }
