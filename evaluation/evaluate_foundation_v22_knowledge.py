"""Observational Base knowledge-completion tracking with the corrected v1.4 proxy."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_foundation_v13 import KNOWLEDGE_PROBES, PRIMARY_MODES
from evaluation.investigate_foundation_v14 import generate_ids
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import load_json


MILESTONES = (256_000, 320_000, 384_000, 448_000, 512_000)


def checkpoint_for(tokens: int) -> Path:
    if tokens == 256_000:
        return ROOT / "checkpoints/foundation-v21-ab/current/seed-42/checkpoint-tokens-256000.pt"
    return ROOT / f"checkpoints/foundation-v22-current/current/seed-42/checkpoint-tokens-{tokens}.pt"


def evaluate_checkpoint(tokenizer: FoundationTokenizer, tokens: int) -> dict:
    path = checkpoint_for(tokens)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    mode = next(row for row in PRIMARY_MODES if row["name"] == "greedy_no_penalty")
    items = []
    for index, (prompt, expected_keywords) in enumerate(KNOWLEDGE_PROBES):
        generated = generate_ids(
            model, tokenizer, tokenizer.encode(prompt, add_bos=True), mode,
            seed=33_000 + tokens + index, max_new_tokens=64,
        )
        hits = [keyword for keyword in expected_keywords if keyword in generated["text"]]
        items.append({
            "prompt": prompt,
            "expected_keywords": expected_keywords,
            "text": generated["text"],
            "keyword_hits": hits,
            "keyword_hit": bool(hits),
            "evaluation": {
                key: generated[key] for key in (
                    "character_valid", "natural_japanese_proxy", "semantic_coherence_proxy",
                    "completion_proxy", "eos_reached", "runaway", "repetition_rate",
                )
            },
        })
    count = len(items)
    result = {
        "tokens": tokens,
        "checkpoint": path.relative_to(ROOT).as_posix(),
        "mode": mode,
        "knowledge_completion_is_observational_only": True,
        "keyword_hit_rate": sum(row["keyword_hit"] for row in items) / count,
        "natural_japanese_proxy_rate": sum(row["evaluation"]["natural_japanese_proxy"] for row in items) / count,
        "character_valid_rate": sum(row["evaluation"]["character_valid"] for row in items) / count,
        "mean_repetition_rate": sum(row["evaluation"]["repetition_rate"] for row in items) / count,
        "items": items,
    }
    del payload, model
    return result


def main() -> int:
    settings = load_json("configs/unipilot-foundation-v22.json")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    rows = [evaluate_checkpoint(tokenizer, tokens) for tokens in MILESTONES]
    output = {
        "schema": "foundation-v22-knowledge-probes-v1",
        "phase": 33,
        "representative_seed": 42,
        "corrected_evaluator": "evaluation.investigate_foundation_v14.language_proxy",
        "final_blind_used": False,
        "rows": rows,
    }
    path = ROOT / "evaluation/foundation-v22-knowledge-probes.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"milestones": len(rows), "output": path.relative_to(ROOT).as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
