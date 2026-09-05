"""Recompute historical same-prefix loop confidence metrics omitted in PHASE 40."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.diagnose_foundation_v29_generation import (
    build_prefixes,
    document_ranges,
    generate_batch,
    summarize_generation,
)
from evaluation.evaluate_foundation_v32_maturity import MODES
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from training.train_foundation_v21_ab import file_sha256


CHECKPOINTS = {
    "5_120_000": (
        ROOT / "checkpoints/foundation-v26-current/current/seed-42/checkpoint-tokens-5120000.pt",
        "fec985e601851b7412c9ff8f874f0f07fc4174152a43354bf4495400de47530e",
    ),
    "7_168_000": (
        ROOT / "checkpoints/foundation-v26-current/current/seed-42/checkpoint-tokens-7168000.pt",
        "76b655e6692299b2c3da25374aaf5d4fd57679b1c53b5cb8c1b4b3e51fa5415b",
    ),
    "10_240_000": (
        ROOT / "checkpoints/foundation-v26-current/current/seed-42/checkpoint-tokens-10240000.pt",
        "e9d9322a1192a0253d6f4c944cb0ff87f4eec648f7fb30394aab5661f0b574aa",
    ),
}


def summarize_loop(rows: list[dict]) -> dict:
    summary = summarize_generation(rows)
    onsets = [row["loop"]["loop_onset"] for row in rows if row["loop"]["loop_onset"]]
    onset_steps = [
        step
        for row in rows
        for step in row["trace"]
        if row["loop"]["loop_onset"] and step["step"] == row["loop"]["loop_onset"]
    ]
    summary.update(
        {
            "median_loop_onset": float(np.median(onsets)) if onsets else None,
            "mean_loop_onset": float(np.mean(onsets)) if onsets else None,
            "loop_onset_distribution": {
                "entropy": float(np.mean([step["entropy"] for step in onset_steps])),
                "top1_probability": float(
                    np.mean([step["top5"][0]["probability"] for step in onset_steps])
                ),
                "top1_top2_margin": float(
                    np.mean([step["top1_top2_margin"] for step in onset_steps])
                ),
                "eos_probability": float(
                    np.mean([step["eos_probability"] for step in onset_steps])
                ),
            },
        }
    )
    return summary


def main() -> None:
    torch.set_num_threads(2)
    tokenizer = FoundationTokenizer.load(ROOT / "tokenizer/foundation-v11-base-4096.json")
    validation = np.memmap(
        ROOT / "data/foundation_v11/packed/vocab-4096/validation.bin",
        dtype=np.uint16,
        mode="r",
    )
    ranges = document_ranges(validation, tokenizer.bos_id, tokenizer.eos_id)
    prefixes = build_prefixes(validation, ranges, tokenizer)
    prompts = [row["prefix_ids"] for row in prefixes]
    output = {"schema": "foundation-v32-historical-loop-dynamics-v1", "phase": 43, "rows": {}}
    for label, (path, expected_hash) in CHECKPOINTS.items():
        before = file_sha256(path)
        if before != expected_hash:
            raise RuntimeError(f"historical checkpoint hash mismatch: {label}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        rows = generate_batch(
            model,
            tokenizer,
            prompts,
            MODES["greedy"],
            list(range(100)),
            128,
            trace=True,
        )
        output["rows"][label] = {
            "checkpoint_sha256": before,
            "checkpoint_unchanged": file_sha256(path) == before,
            **summarize_loop(rows),
        }
        print(json.dumps({"tokens": label, **output["rows"][label]}), flush=True)
    destination = ROOT / "evaluation/foundation-v32-historical-loop-dynamics.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
