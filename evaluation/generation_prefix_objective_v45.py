"""PHASE57 objective prototype ONLY; no model, optimizer or training entry point.

Unlike PHASE42's teacher-forced3/4gram candidate search, this acts only on a
detached generated continuation after four identical contiguous cycles.
Reference-supported repetition is vetoed. This is not a general truth detector.
"""
from __future__ import annotations
import torch


def cycle_negatives(generated: list[int], reference: list[int],
                    special_ids: set[int]) -> list[tuple[int, int]]:
    """Return (generated next-token index, negative token). No prompt tokens.

    At t, four cycles must already exist before g[t]; smallest width1..8 wins.
    Exclude actual aligned target, specials and any repeated suffix+candidate
    that exists in the reference. Reference is original TRAIN continuation,
    never Diagnostic/Confirmatory/Reserve data. Absence is not proof of harm.
    """
    negatives=[]
    for t,candidate in enumerate(generated):
        if candidate in special_ids or t>=len(reference) or candidate==reference[t]:
            continue
        for width in range(1,9):
            if t<4*width:
                continue
            cycle=generated[t-width:t]
            if any(x in special_ids for x in cycle):
                continue
            if generated[t-4*width:t]!=cycle*4 or candidate!=cycle[0]:
                continue
            snippet=generated[t-4*width:t+1]
            if any(reference[j:j+len(snippet)]==snippet for j in range(len(reference)-len(snippet)+1)):
                break
            negatives.append((t,candidate));break
    return negatives


def generated_prefix_unlikelihood(logits: torch.Tensor,
                                 negatives: list[tuple[int,int]]) -> torch.Tensor:
    """logits[t] predicts generated[t] from prefix+generated[:t], shape[T,V]."""
    if logits.ndim!=2:raise ValueError('expected [generated steps, vocabulary]')
    if not negatives:return logits.sum()*0.0
    positions=torch.tensor([t for t,_ in negatives],device=logits.device)
    tokens=torch.tensor([v for _,v in negatives],device=logits.device)
    probability=logits[positions].softmax(-1).gather(1,tokens[:,None]).squeeze(1)
    return -torch.log1p(-probability.clamp(max=1-1e-6)).mean()
