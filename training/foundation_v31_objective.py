"""Pure helpers for the PHASE 42 repetition-aware objective."""
from __future__ import annotations
import torch


JAPANESE_FUNCTION_WORDS = ("の", "に", "は", "を", "が", "と", "で", "て")


def repetition_negative_candidates(sequence: torch.Tensor, targets: torch.Tensor,
                                   ngrams: tuple[int, ...] = (3, 4)) -> list[tuple[int, int, int]]:
    """Return (batch, position, candidate), excluding the actual next-token target."""
    output: set[tuple[int, int, int]] = set()
    values = sequence.detach().cpu().tolist()
    truth = targets.detach().cpu().tolist()
    for batch, row in enumerate(values):
        seen = {width: {} for width in ngrams}
        for position in range(len(row)):
            for width in ngrams:
                prefix_width = width - 1
                completed_start = position - width + 1
                if completed_start >= 0:
                    prefix = tuple(row[completed_start:completed_start + prefix_width])
                    seen[width].setdefault(prefix, set()).add(row[position])
                if position + 1 < prefix_width:
                    continue
                suffix = tuple(row[position - prefix_width + 1:position + 1])
                for candidate in seen[width].get(suffix, ()):
                    if candidate != truth[batch][position]:
                        output.add((batch, position, candidate))
    return sorted(output)


def unlikelihood_loss(logits: torch.Tensor, negatives: list[tuple[int, int, int]]) -> torch.Tensor:
    if not negatives:
        return logits.sum() * 0.0
    index = torch.tensor(negatives, device=logits.device, dtype=torch.long)
    probabilities = torch.softmax(logits[index[:, 0], index[:, 1]], -1)
    selected = probabilities.gather(1, index[:, 2:3]).squeeze(1)
    return -torch.log1p(-selected.clamp(max=1.0 - 1e-6)).mean()


def weighted_lm_loss(logits: torch.Tensor, targets: torch.Tensor, eos_id: int,
                     eos_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_token = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).view_as(targets)
    eos = targets == eos_id
    weighted = per_token * torch.where(eos, eos_weight, 1.0)
    return weighted.mean(), per_token[eos].mean() if eos.any() else per_token.sum() * 0.0, per_token[~eos].mean()
