from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.diagnostic_transformer_v15 import DiagnosticConfig, DiagnosticTransformer
from training.optimizer import create_optimizer


TASKS = [
    "indexed_copy",
    "previous_key_lookup",
    "long_range_dependency",
    "pattern_continuation",
    "context_conditioned",
]
TASK_TOKEN = {name: 2 + index for index, name in enumerate(TASKS)}
ANSWER_TOKEN = 8
VALUE_TOKENS = list(range(16, 32))
KEY_TOKENS = list(range(32, 36))
FILLER_TOKENS = list(range(40, 56))


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def make_example(task: str, rng: random.Random, length: int = 32) -> tuple[list[int], int]:
    sequence = [rng.choice(FILLER_TOKENS) for _ in range(length)]
    sequence[0] = TASK_TOKEN[task]
    sequence[-1] = ANSWER_TOKEN
    if task == "indexed_copy":
        values = [rng.choice(VALUE_TOKENS) for _ in range(4)]
        query = rng.randrange(4)
        sequence[2:6] = values
        sequence[-3] = 36 + query
        sequence[-2] = 39
        answer = values[query]
    elif task == "previous_key_lookup":
        values = rng.sample(VALUE_TOKENS, 4)
        for index, (key, value) in enumerate(zip(KEY_TOKENS, values)):
            sequence[2 + index * 2] = key
            sequence[3 + index * 2] = value
        query = rng.randrange(4)
        sequence[-3] = 38
        sequence[-2] = KEY_TOKENS[query]
        answer = values[query]
    elif task == "long_range_dependency":
        answer = rng.choice(VALUE_TOKENS)
        sequence[1] = answer
        sequence[-3] = 37
        sequence[-2] = 39
    elif task == "pattern_continuation":
        first, second = rng.sample(VALUE_TOKENS, 2)
        for index in range(2, 18):
            sequence[index] = first if index % 2 == 0 else second
        choose_second = bool(rng.randrange(2))
        sequence[-3] = 38
        sequence[-2] = 33 if choose_second else 32
        answer = second if choose_second else first
    elif task == "context_conditioned":
        condition = rng.randrange(4)
        answer = VALUE_TOKENS[condition]
        sequence[1] = KEY_TOKENS[condition]
        sequence[-3] = 37
        sequence[-2] = 39
    else:
        raise KeyError(task)
    return sequence, answer


def make_batch(
    rng: random.Random, batch_size: int, *, task: str | None = None
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
    inputs = []
    targets = []
    names = []
    answers = []
    for index in range(batch_size):
        selected = task or TASKS[index % len(TASKS)]
        sequence, answer = make_example(selected, rng)
        target = [-100] * len(sequence)
        target[-1] = answer
        inputs.append(sequence)
        targets.append(target)
        names.append(selected)
        answers.append(answer)
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        names,
        answers,
    )


@torch.inference_mode()
def evaluate(
    model: DiagnosticTransformer, seed: int, examples_per_task: int,
    tasks: list[str] | None = None,
) -> dict:
    model.eval()
    by_task = {}
    all_correct = 0
    all_examples = 0
    output_counts = Counter()
    evaluated_tasks = tasks or TASKS
    for task_index, task in enumerate(evaluated_tasks):
        rng = random.Random(seed + 10_000 + task_index)
        correct = total = 0
        batch_size = 20
        for _ in range(0, examples_per_task, batch_size):
            size = min(batch_size, examples_per_task - total)
            inputs, _, _, answers = make_batch(rng, size, task=task)
            logits, _ = model(inputs)
            predictions = logits[:, -1].argmax(dim=-1).tolist()
            correct += sum(prediction == answer for prediction, answer in zip(predictions, answers))
            output_counts.update(answers)
            total += size
        by_task[task] = {
            "examples": total,
            "accuracy": correct / total,
        }
        all_correct += correct
        all_examples += total
    majority = max(output_counts.values()) / sum(output_counts.values())
    return {
        "examples": all_examples,
        "overall_accuracy": all_correct / all_examples,
        "by_task": by_task,
        "bigram_last_token_baseline_accuracy": majority,
        "last_input_token_is_constant": ANSWER_TOKEN,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v15.json")
    parser.add_argument("--output", default="evaluation/foundation-v15-synthetic-context.json")
    parser.add_argument("--variant", default="current_preln_gelu_tied")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--task", choices=TASKS)
    args = parser.parse_args()
    settings = load_json(args.config)
    synthetic = settings["synthetic"]
    torch.set_num_threads(int(settings["cpu_threads"]))
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    architecture = dict(settings["architecture"])
    variant = next(row for row in settings["ablations"] if row["name"] == args.variant)
    architecture.update(variant["changes"])
    architecture.update({
        "context_length": int(synthetic["context_length"]),
        "dropout": 0.1,
    })
    model = DiagnosticTransformer(DiagnosticConfig(
        model_name=f"Foundation v1.5 synthetic context audit {args.variant}",
        vocab_size=int(synthetic["vocab_size"]),
        **architecture,
    ))
    optimizer = create_optimizer(model, float(synthetic["learning_rate"]), 0.01)
    training_rng = random.Random(seed + 1)
    batch_size = int(synthetic["batch_size"])
    max_updates = args.max_updates or int(synthetic["max_updates"])
    target_accuracy = float(synthetic["target_accuracy"])
    evaluation_examples = int(synthetic["evaluation_examples_per_task"])
    evaluated_tasks = [args.task] if args.task else TASKS
    curve = []
    started = time.perf_counter()
    losses = []
    stopped = max_updates
    for update in range(1, max_updates + 1):
        inputs, targets, _, _ = make_batch(training_rng, batch_size, task=args.task)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite synthetic context loss")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        losses.append(float(loss.item()))
        if update in {1, 25, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000} or update == max_updates:
            measured = evaluate(model, seed, evaluation_examples, evaluated_tasks)
            row = {
                "update": update,
                "recent_loss": sum(losses[-25:]) / min(25, len(losses)),
                "gradient_norm": gradient_norm,
                **measured,
            }
            curve.append(row)
            print(json.dumps(row), flush=True)
            if update >= 100 and all(
                value["accuracy"] > target_accuracy
                for value in measured["by_task"].values()
            ):
                stopped = update
                break
    final = evaluate(model, seed, evaluation_examples, evaluated_tasks)
    context_pass = all(
        value["accuracy"] > target_accuracy for value in final["by_task"].values()
    )
    report = {
        "schema_version": "foundation-v15-synthetic-context-v1",
        "architecture": model.config.to_dict(),
        "variant": variant,
        "parameters": model.parameter_count(),
        "tasks": evaluated_tasks,
        "training_mode": "independent_task" if args.task else "mixed_tasks",
        "training": {
            "updates": stopped,
            "batch_size": batch_size,
            "supervised_targets_per_update": batch_size,
            "learning_rate": synthetic["learning_rate"],
            "wall_seconds": time.perf_counter() - started,
            "curve": curve,
        },
        "final": final,
        "target_accuracy": target_accuracy,
        "context_learning": "PASS" if context_pass else "FAIL",
        "bigram_cannot_solve_by_last_token": (
            final["bigram_last_token_baseline_accuracy"] < 0.20
        ),
        "external_ai_api": "OFF",
        "production_changed": False,
        "final_blind_used": False,
    }
    (ROOT / args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "context_learning": report["context_learning"],
        "final": final,
    }, indent=2))
    return 0 if context_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
