from __future__ import annotations

import heapq
import re


class FastBPEEncoder:
    """Exact, heap-based encoder for UniPilot's byte-level BPE format.

    The production tokenizer intentionally stays unchanged.  Foundation corpora are
    large enough that scanning every merge over every byte is prohibitively slow,
    so this research-only encoder applies the same ranked merges with a linked list.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.offset = len(tokenizer.special_tokens)
        self.special_pattern = re.compile(
            "(" + "|".join(re.escape(token) for token in tokenizer.special_tokens) + ")"
        )
        self.merge_map = tokenizer.merge_map

    def _encode_bytes(self, value: bytes) -> list[int]:
        if not value:
            return []
        tokens = [byte + self.offset for byte in value]
        size = len(tokens)
        previous = [index - 1 for index in range(size)]
        following = [index + 1 for index in range(size)]
        following[-1] = -1
        alive = [True] * size
        queue: list[tuple[int, int, int, int, int, int]] = []

        def queue_pair(left: int) -> None:
            if left < 0 or not alive[left]:
                return
            right = following[left]
            if right < 0 or not alive[right]:
                return
            merge = self.merge_map.get((tokens[left], tokens[right]))
            if merge is not None:
                rank, new_token = merge
                heapq.heappush(
                    queue, (rank, left, right, tokens[left], tokens[right], new_token)
                )

        for index in range(size - 1):
            queue_pair(index)

        while queue:
            _, left, right, expected_left, expected_right, new_token = heapq.heappop(queue)
            if (
                not alive[left]
                or not alive[right]
                or following[left] != right
                or tokens[left] != expected_left
                or tokens[right] != expected_right
            ):
                continue
            tokens[left] = new_token
            alive[right] = False
            next_index = following[right]
            following[left] = next_index
            if next_index >= 0:
                previous[next_index] = left
            queue_pair(previous[left])
            queue_pair(left)

        output: list[int] = []
        index = 0
        while index >= 0:
            if alive[index]:
                output.append(tokens[index])
            index = following[index]
        return output

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.tokenizer.bos_id] if add_bos else []
        for part in self.special_pattern.split(text):
            if not part:
                continue
            if part in self.tokenizer.special_to_id:
                ids.append(self.tokenizer.special_to_id[part])
            else:
                ids.extend(self._encode_bytes(part.encode("utf-8")))
        if add_eos:
            ids.append(self.tokenizer.eos_id)
        return ids
