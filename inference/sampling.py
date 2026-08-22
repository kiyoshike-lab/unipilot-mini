from __future__ import annotations

import torch


def sample_next_token(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.95) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities > top_p
        sorted_probabilities[remove] = 0
        sorted_probabilities /= sorted_probabilities.sum()
        return int(sorted_indices[torch.multinomial(sorted_probabilities, 1)])
    return int(torch.multinomial(probabilities, 1))


def apply_repetition_penalty(logits: torch.Tensor, previous_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty <= 1:
        return logits
    logits = logits.clone()
    token_ids = torch.tensor(list(set(previous_ids)), dtype=torch.long, device=logits.device)
    selected = logits[token_ids]
    logits[token_ids] = torch.where(selected > 0, selected / penalty, selected * penalty)
    return logits
