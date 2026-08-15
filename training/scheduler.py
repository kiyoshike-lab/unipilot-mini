import math


def warmup_cosine_multiplier(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.1) -> float:
    if warmup_steps and step < warmup_steps:
        return max(1e-8, (step + 1) / warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
