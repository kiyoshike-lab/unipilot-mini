"""Re-evaluate the fixed punctuation probe through the PHASE 37 budget."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import frequency_ranks, language_metrics, load_json


MILESTONES = (2_048_000, 2_560_000, 3_072_000, 3_584_000, 4_096_000, 4_608_000, 5_120_000)


def checkpoint(tokens: int) -> Path:
    if tokens == 2_048_000:
        return ROOT / "checkpoints/foundation-v25-current/current/seed-42/checkpoint-tokens-2048000.pt"
    return ROOT / f"checkpoints/foundation-v26-current/current/seed-42/checkpoint-tokens-{tokens}.pt"


def main() -> int:
    torch.set_num_threads(4)
    settings = load_json("configs/unipilot-foundation-v26.json")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    corpus = load_json(settings["corpus_manifest"])
    train = np.memmap(ROOT / corpus["splits"]["train"]["path"], dtype=np.uint16, mode="r")
    validation = np.memmap(ROOT / corpus["splits"]["validation"]["path"], dtype=np.uint16, mode="r")
    ranks = frequency_ranks(train, tokenizer.vocab_size)
    rows = []
    for tokens in MILESTONES:
        payload = torch.load(checkpoint(tokens), map_location="cpu", weights_only=False)
        model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
        model.load_state_dict(payload["model_state"], strict=True)
        metrics = language_metrics(model, tokenizer, validation, ranks, 8192)
        rows.append({
            "tokens": tokens,
            "punctuation": metrics["punctuation"],
            "period_comma_prediction_mass": metrics["period_comma_prediction_mass"],
        })
        del payload, model
    result = {
        "schema": "foundation-v26-nine-token-punctuation-probe-v1",
        "phase": 37,
        "representative_seed": 42,
        "validation_tokens": 8192,
        "rows": rows,
    }
    (ROOT / "evaluation/foundation-v26-punctuation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"milestones": len(rows), "tokens": list(rows[-1]["punctuation"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
