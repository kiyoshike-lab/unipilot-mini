from __future__ import annotations

from dataclasses import dataclass

from foundation.synthetic_context_v3 import (
    ANSWER,
    KEY,
    PAIR,
    QUERY,
    TASK_KEY_LOOKUP,
    VALUE,
)


@dataclass(frozen=True)
class OracleParseV31:
    mapping: tuple[tuple[int, int], ...]
    query: int
    answer: int


def parse_key_lookup_v31(sequence: list[int] | tuple[int, ...]) -> OracleParseV31:
    """Parse Benchmark v3 markers without using generator metadata or a model."""
    tokens = list(sequence)
    if not tokens or tokens[0] != TASK_KEY_LOOKUP:
        raise ValueError("not a Synthetic Context Benchmark key lookup sequence")

    cursor = 1
    mapping: list[tuple[int, int]] = []
    while cursor < len(tokens) and tokens[cursor] == PAIR:
        if cursor + 4 >= len(tokens):
            raise ValueError("truncated relation")
        if tokens[cursor + 1] != KEY or tokens[cursor + 3] != VALUE:
            raise ValueError("malformed relation markers")
        mapping.append((tokens[cursor + 2], tokens[cursor + 4]))
        cursor += 5

    if tokens[cursor:] and len(tokens) - cursor == 3:
        if tokens[cursor] != QUERY or tokens[cursor + 2] != ANSWER:
            raise ValueError("malformed query markers")
        query = tokens[cursor + 1]
    else:
        raise ValueError("missing or malformed query")

    matches = [value for key, value in mapping if key == query]
    if len(matches) != 1:
        raise ValueError(f"query must resolve to exactly one relation, found {len(matches)}")
    return OracleParseV31(tuple(mapping), query, matches[0])


def oracle_answer_v31(sequence: list[int] | tuple[int, ...]) -> int:
    return parse_key_lookup_v31(sequence).answer
