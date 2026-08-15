import math


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))
