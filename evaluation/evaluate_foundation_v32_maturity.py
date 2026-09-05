"""Sequential offline evaluation for the PHASE 43 maturity pilot."""
from __future__ import annotations

import argparse
import json
import math
import time
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
    ngram_repetition,
    summarize_generation,
    target_metrics,
)
from evaluation.audit_foundation_v15_architecture import context_ablation
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from training.train_foundation_v21_ab import frequency_ranks, language_metrics


HORIZONS = (1, 2, 4, 8, 16, 32)
FREE_RUNNING_HORIZONS = (1, 2, 4, 8, 16, 32, 64)
MODES = {
    "greedy": {
        "name": "greedy",
        "kind": "greedy",
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.0,
        "no_repeat_ngram": None,
        "eos_threshold": None,
    },
    "temperature_0.7": {
        "name": "temperature_0.7",
        "kind": "sampling",
        "temperature": 0.7,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.0,
        "no_repeat_ngram": None,
        "eos_threshold": None,
    },
    "temperature_0.8": {
        "name": "temperature_0.8",
        "kind": "sampling",
        "temperature": 0.8,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.0,
        "no_repeat_ngram": None,
        "eos_threshold": None,
    },
    "top_p_0.90": {
        "name": "top_p_0.90",
        "kind": "sampling",
        "temperature": 1.0,
        "top_k": None,
        "top_p": 0.90,
        "repetition_penalty": 1.0,
        "no_repeat_ngram": None,
        "eos_threshold": None,
    },
}


def load_model(path: Path) -> tuple[dict, DiagnosticTransformerV17]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return payload, model


def _reference(validation: np.memmap, prefix: dict, length: int = 64) -> list[int]:
    start = int(prefix["start"]) + int(prefix["prefix_length"])
    return np.asarray(validation[start : start + length], dtype=np.int64).tolist()


@torch.inference_mode()
def teacher_forced_horizons(
    model: DiagnosticTransformerV17,
    validation: np.memmap,
    prefixes: list[dict],
) -> dict:
    device = next(model.parameters()).device
    observations = {h: {"loss": [], "correct": [], "probability": []} for h in HORIZONS}
    grouped: dict[int, list[dict]] = {}
    for prefix in prefixes:
        grouped.setdefault(len(prefix["prefix_ids"]), []).append(prefix)
    for group in grouped.values():
        inputs = []
        targets = []
        prefix_length = len(group[0]["prefix_ids"])
        for prefix in group:
            reference = _reference(validation, prefix, max(HORIZONS))
            inputs.append(prefix["prefix_ids"] + reference[:-1])
            targets.append(reference)
        x = torch.tensor(inputs, dtype=torch.long, device=device)
        y = torch.tensor(targets, dtype=torch.long, device=device)
        logits, _ = model(x)
        horizon_logits = logits[:, prefix_length - 1 : prefix_length - 1 + max(HORIZONS)].float()
        probabilities = torch.softmax(horizon_logits, dim=-1)
        for horizon in HORIZONS:
            index = horizon - 1
            scores = horizon_logits[:, index]
            truth = y[:, index]
            assigned = probabilities[:, index].gather(1, truth[:, None]).squeeze(1)
            observations[horizon]["loss"].extend((-assigned.clamp_min(1e-30).log()).tolist())
            observations[horizon]["correct"].extend((scores.argmax(-1) == truth).tolist())
            observations[horizon]["probability"].extend(assigned.tolist())
    return {
        str(h): {
            "examples": len(observations[h]["loss"]),
            "loss": float(np.mean(observations[h]["loss"])),
            "accuracy": float(np.mean(observations[h]["correct"])),
            "correct_token_probability": float(np.mean(observations[h]["probability"])),
        }
        for h in HORIZONS
    }


def _topic_retention(generated: list[int], reference: list[int]) -> float:
    if not generated or not reference:
        return 0.0
    reference_set = set(reference)
    return sum(token in reference_set for token in generated) / len(generated)


def free_running_divergence(
    rows: list[dict], validation: np.memmap, prefixes: list[dict]
) -> dict:
    output = {}
    for horizon in FREE_RUNNING_HORIZONS:
        repetition = []
        topic = []
        token_agreement = []
        entropy = []
        margin = []
        for row, prefix in zip(rows, prefixes):
            generated = row["ids"][:horizon]
            reference = _reference(validation, prefix, horizon)
            width = min(len(generated), len(reference))
            repetition.append(ngram_repetition(generated))
            topic.append(_topic_retention(generated, reference))
            token_agreement.append(
                sum(a == b for a, b in zip(generated[:width], reference[:width])) / max(1, width)
            )
            trace = row["trace"][:horizon]
            entropy.append(float(np.mean([step["entropy"] for step in trace])) if trace else 0.0)
            margin.append(float(np.mean([step["top1_top2_margin"] for step in trace])) if trace else 0.0)
        output[str(horizon)] = {
            "mean_repetition_1": float(np.mean(repetition)),
            "topic_retention_proxy": float(np.mean(topic)),
            "exact_reference_agreement": float(np.mean(token_agreement)),
            "mean_entropy": float(np.mean(entropy)),
            "mean_top1_top2_margin": float(np.mean(margin)),
        }
    return output


def generation_summary(
    model: DiagnosticTransformerV17,
    tokenizer: FoundationTokenizer,
    validation: np.memmap,
    prefixes: list[dict],
) -> tuple[dict, dict]:
    prompt_ids = [row["prefix_ids"] for row in prefixes]
    summaries = {}
    greedy_rows = []
    for mode_index, (name, mode) in enumerate(MODES.items()):
        maximum = 128 if name == "greedy" else 64
        rows = generate_batch(
            model,
            tokenizer,
            prompt_ids,
            mode,
            [43_000 + mode_index * 10_000 + index for index in range(len(prefixes))],
            maximum,
            trace=name == "greedy",
        )
        references = [_reference(validation, prefix, maximum) for prefix in prefixes]
        summary = summarize_generation(rows)
        summary["topic_retention_proxy"] = float(
            np.mean([_topic_retention(row["ids"], ref) for row, ref in zip(rows, references)])
        )
        if name == "greedy":
            onsets = [row["loop"]["loop_onset"] for row in rows if row["loop"]["loop_onset"]]
            onset_steps = [
                step
                for row in rows
                for step in row["trace"]
                if row["loop"]["loop_onset"] and step["step"] == row["loop"]["loop_onset"]
            ]
            summary["median_loop_onset"] = float(np.median(onsets)) if onsets else None
            summary["mean_loop_onset"] = float(np.mean(onsets)) if onsets else None
            summary["loop_onset_distribution"] = {
                "entropy": float(np.mean([step["entropy"] for step in onset_steps])) if onset_steps else None,
                "top1_probability": float(
                    np.mean([step["top5"][0]["probability"] for step in onset_steps])
                ) if onset_steps else None,
                "top1_top2_margin": float(
                    np.mean([step["top1_top2_margin"] for step in onset_steps])
                ) if onset_steps else None,
                "eos_probability": float(
                    np.mean([step["eos_probability"] for step in onset_steps])
                ) if onset_steps else None,
            }
        summaries[name] = summary
        if name == "greedy":
            greedy_rows = rows
    return summaries, free_running_divergence(greedy_rows, validation, prefixes)


def evaluate(path: Path, label: str) -> dict:
    started = time.perf_counter()
    payload, model = load_model(path)
    tokenizer = FoundationTokenizer.load(ROOT / "tokenizer/foundation-v11-base-4096.json")
    train = np.memmap(
        ROOT / "data/foundation_v11/packed/vocab-4096/train.bin", dtype=np.uint16, mode="r"
    )
    validation = np.memmap(
        ROOT / "data/foundation_v11/packed/vocab-4096/validation.bin",
        dtype=np.uint16,
        mode="r",
    )
    ranks = frequency_ranks(train, tokenizer.vocab_size)
    ranges = document_ranges(validation, tokenizer.bos_id, tokenizer.eos_id)
    prefixes = build_prefixes(validation, ranges, tokenizer)
    terminal_unique = np.asarray(
        [int(position) for position in np.flatnonzero(validation == tokenizer.eos_id) if position >= 128]
    )
    terminal = np.resize(terminal_unique, 500)
    nonterminal = np.linspace(128, len(validation) - 2, 500, dtype=int)

    validation_metrics = language_metrics(model, tokenizer, validation, ranks, 8192)
    context_diagnostics = context_ablation(model, validation, probes=32)
    context = {
        str(length): context_diagnostics[str(length)]["loss"]
        for length in (512, 64, 16, 2, 1)
    }
    generation, divergence = generation_summary(model, tokenizer, validation, prefixes)
    result = {
        "schema": "foundation-v32-maturity-evaluation-v1",
        "phase": 43,
        "label": label,
        "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "tokens_processed": int(payload["tokens_processed"]),
        "validation": validation_metrics,
        "terminal_eos": target_metrics(model, validation, terminal, tokenizer.eos_id),
        "nonterminal_eos": target_metrics(model, validation, nonterminal, tokenizer.eos_id),
        "context_loss": context,
        "context_diagnostics": context_diagnostics,
        "full_context_advantage_vs_1": context_diagnostics["full_vs_last_1_loss_advantage"],
        "teacher_forced_horizons": teacher_forced_horizons(model, validation, prefixes),
        "generation": generation,
        "free_running_divergence": divergence,
        "all_finite": all(
            math.isfinite(value)
            for value in (
                validation_metrics["loss"],
                validation_metrics["perplexity"],
                *context.values(),
            )
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    destination = ROOT / f"evaluation/foundation-v32-{label}-evaluation.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", choices=("baseline", "pilot"), required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    result = evaluate(checkpoint, args.label)
    print(
        json.dumps(
            {
                "label": result["label"],
                "tokens": result["tokens_processed"],
                "loss": result["validation"]["loss"],
                "top1": result["validation"]["top_1_accuracy"],
                "wall_seconds": result["wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
