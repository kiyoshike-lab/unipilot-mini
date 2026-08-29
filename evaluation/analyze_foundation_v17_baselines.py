from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.measure_foundation_v17 import (
    architecture_probe,
    hidden_token_similarity,
    stats,
    validation_metrics,
)
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from training.investigate_foundation_v14 import macro_batch, macro_permutation


def load_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def v17_config(payload: dict, variant: dict) -> DiagnosticConfigV17:
    source = payload["config"]
    return DiagnosticConfigV17(
        model_name=f"Foundation v1.7 converted baseline {variant['name']}",
        vocab_size=source["vocab_size"],
        context_length=source["context_length"],
        embedding_dim=source["embedding_dim"],
        n_layers=source["n_layers"],
        n_heads=source["n_heads"],
        ffn_dim=source["ffn_dim"],
        dropout=source["dropout"],
        bias=source["bias"],
        norm=source["norm"],
        norm_epsilon=source["norm_epsilon"],
        activation=source["activation"],
        token_embedding_scale=variant["token_embedding_scale"],
        position_embedding_scale=variant["position_embedding_scale"],
        residual_projection_init_scale=1.0,
        weight_tying=source["weight_tying"],
    )


def main() -> int:
    settings = load_json("configs/unipilot-foundation-v17.json")
    corpus = load_json(settings["corpus_manifest"])
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    train = np.memmap(ROOT / corpus["splits"]["train"]["path"], dtype=np.uint16, mode="r")
    validation = np.memmap(
        ROOT / corpus["splits"]["validation"]["path"], dtype=np.uint16, mode="r"
    )
    audit_tokens = torch.from_numpy(
        np.asarray(validation[8192:8320], dtype=np.int64).copy()
    ).unsqueeze(0)
    results = {}
    for variant_name in ("current_unscaled", "sqrt_scaled_a"):
        variant = next(row for row in settings["variants"] if row["name"] == variant_name)
        results[variant_name] = []
        for seed in settings["seeds"]:
            checkpoint_path = (
                ROOT / "checkpoints/foundation-v16-reproduction"
                / f"{variant_name}-seed-{seed}.pt"
            )
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = v17_config(payload, variant)
            model = DiagnosticTransformerV17(config)
            model.load_state_dict(payload["model_state"], strict=True)
            seed_all(seed)
            initialized = DiagnosticTransformerV17(config)
            initial_position = initialized.embeddings.position.weight.detach()
            delta = model.embeddings.position.weight.detach() - initial_position
            permutation = macro_permutation((len(train) - 1) // 512, seed)
            x, y = macro_batch(train, [int(permutation[128])], 512)
            torch.manual_seed(28_000 + seed)
            model.train()
            model.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            gradient = model.embeddings.position.weight.grad
            if gradient is None:
                raise RuntimeError("converted baseline position gradient missing")
            model.eval()
            results[variant_name].append({
                "seed": seed,
                "source_checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
                "probe": architecture_probe(model, audit_tokens),
                "validation": validation_metrics(model, validation, 8192),
                "next_batch_loss": float(loss.detach()),
                "position_gradient": stats(gradient),
                "position_parameter_delta": stats(delta),
                "position_parameter_delta_relative_norm": (
                    float(torch.linalg.vector_norm(delta))
                    / max(float(torch.linalg.vector_norm(initial_position)), 1e-12)
                ),
                "hidden_token_similarity": hidden_token_similarity(
                    model, tokenizer, validation
                ),
                "final_norm": "PRESENT",
            })
    output = {
        "schema_version": "foundation-v17-converted-baseline-diagnostics-v1",
        "method": (
            "PHASE 27 final checkpoints loaded strictly into the parameter-identical "
            "v1.7 diagnostic wrapper; gradients use the same next macro batch and a "
            "fixed dropout seed within each seed pair"
        ),
        "results": results,
        "final_blind_used": False,
        "external_ai_api": "OFF",
        "production_changed": False,
    }
    path = ROOT / "evaluation/foundation-v17-baseline-diagnostics.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        variant: [
            {
                "seed": row["seed"],
                "position_ratio": row["probe"]["embedding"][
                    "effective_token_to_position_rms_ratio"
                ],
                "position_gradient_rms": row["position_gradient"]["rms"],
                "position_delta_rms": row["position_parameter_delta"]["rms"],
            }
            for row in rows
        ]
        for variant, rows in results.items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
