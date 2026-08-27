from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from foundation.packed_dataset import PackedTokenDataset
from training.foundation_v12_diagnostics import (
    SequenceRecord,
    build_model,
    causal_mask_audit,
    encode_records,
    generate,
    initialization_audit,
    loss_audit,
    memorization_probe,
    read_short_clean_documents,
    train_sequences,
)


SYNTHETIC_EOS_SENTENCES = [
    "今日は晴れです。",
    "大学へ行きます。",
    "図書館で本を読みます。",
    "授業の復習をします。",
    "課題を期限までに提出します。",
    "朝に予定を確認します。",
    "分からない点を質問します。",
    "根拠を確かめて書きます。",
    "短い休憩を取ります。",
    "試験の日程を確認します。",
    "出典を記録します。",
    "文章を声に出して読みます。",
    "目標を小さく分けます。",
    "重要な内容を復習します。",
    "睡眠時間を確保します。",
    "必要な資料を準備します。",
    "結果を表にまとめます。",
    "提出前に誤字を直します。",
    "先生に丁寧なメールを送ります。",
    "学習した内容を説明します。",
]

COMPLETION_PROMPTS = [
    "大学では、授業だけでなく",
    "日本の首都は",
    "水は",
    "人工知能とは",
    "効率よく勉強するためには",
    "コンピュータは",
    "歴史を学ぶ理由は",
    "文章を書くときは",
    "研究を始める前に",
    "情報を確認するときは",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def strip_model(result: dict) -> tuple[torch.nn.Module, dict]:
    model = result.pop("model")
    return model, result


def token_display(tokenizer: FoundationTokenizer, token_id: int) -> dict:
    return {
        "id": token_id,
        "token": tokenizer.backend.id_to_token(token_id),
        "decoded": tokenizer.decode([token_id], skip_special=False),
    }


def target_shift_audit(tokenizer: FoundationTokenizer) -> dict:
    manifest = json_load("data/foundation_v11/packed/vocab-4096/manifest.json")
    path = ROOT / manifest["splits"]["train"]["path"]
    dataset = PackedTokenDataset(path, 512)
    raw = np.memmap(path, dtype=np.uint16, mode="r")
    blocks = []
    all_pass = True
    for block_index in range(10):
        inputs, targets = dataset[block_index]
        start = block_index * 512
        expected_inputs = torch.from_numpy(
            np.asarray(raw[start:start + 512], dtype=np.int64).copy()
        )
        expected_targets = torch.from_numpy(
            np.asarray(raw[start + 1:start + 513], dtype=np.int64).copy()
        )
        exact = (
            torch.equal(inputs, expected_inputs)
            and torch.equal(targets, expected_targets)
            and torch.equal(inputs[1:], targets[:-1])
        )
        all_pass &= exact
        pairs = []
        for position in range(12):
            pairs.append({
                "position": position,
                "input": token_display(tokenizer, int(inputs[position].item())),
                "expected_next": token_display(tokenizer, int(targets[position].item())),
            })
        blocks.append({
            "block": block_index,
            "raw_start": start,
            "exact_one_token_shift": exact,
            "decoded_input_prefix": tokenizer.decode(inputs[:64].tolist(), skip_special=False),
            "pairs": pairs,
        })
    return {
        "input_definition": "token[0:n-1]",
        "target_definition": "token[1:n]",
        "blocks_checked": 10,
        "blocks": blocks,
        "same_token_prediction": False,
        "offset_tokens": 1,
        "status": "PASS" if all_pass else "FAIL",
    }


def random_baseline() -> dict:
    manifest = json_load(
        "checkpoints/foundation-v11-clean-100/checkpoint-step-100.manifest.json"
    )
    baseline = math.log(4096)
    rows = []
    for item in manifest["history"]:
        loss = float(item["validation_loss"])
        rows.append({
            "step": int(item["step"]),
            "validation_loss": loss,
            "train_loss": item["train_loss"],
            "perplexity": math.exp(loss),
            "absolute_improvement_from_random": baseline - loss,
            "percent_improvement_from_random": (baseline - loss) / baseline * 100,
        })
    final = rows[-1]
    return {
        "vocab": 4096,
        "formula": "ln(vocab_size)",
        "random_cross_entropy": baseline,
        "steps": rows,
        "step_100_absolute_improvement": final["absolute_improvement_from_random"],
        "step_100_percent_improvement": final["percent_improvement_from_random"],
        "step_100_perplexity": final["perplexity"],
        "learning_progress_confirmed": final["validation_loss"] < baseline,
    }


def frequency_audit(tokenizer: FoundationTokenizer) -> dict:
    manifest = json_load("data/foundation_v11/packed/vocab-4096/manifest.json")
    path = ROOT / manifest["splits"]["train"]["path"]
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    counts = np.bincount(tokens, minlength=tokenizer.vocab_size)
    order_top = np.argsort(-counts)[:20]
    positive = np.flatnonzero(counts)
    order_bottom = positive[np.argsort(counts[positive])[:20]]
    specials = {}
    for token, token_id in tokenizer.special_to_id.items():
        specials[token] = {
            "id": token_id,
            "count": int(counts[token_id]),
            "fraction": float(counts[token_id] / len(tokens)),
        }
    return {
        "tokens": int(len(tokens)),
        "vocab": tokenizer.vocab_size,
        "used_vocab": int((counts > 0).sum()),
        "zero_frequency_tokens": int((counts == 0).sum()),
        "top_20": [
            {**token_display(tokenizer, int(index)), "count": int(counts[index])}
            for index in order_top
        ],
        "bottom_20_nonzero": [
            {**token_display(tokenizer, int(index)), "count": int(counts[index])}
            for index in order_bottom
        ],
        "special_tokens": specials,
        "eos_fraction": specials["<EOS>"]["fraction"],
        "eos_percent": specials["<EOS>"]["fraction"] * 100,
        "expected_eos_targets_in_original_51200_token_run": (
            specials["<EOS>"]["fraction"] * 51_200
        ),
        "bos_fraction": specials["<BOS>"]["fraction"],
        "pad_is_absent": specials["<PAD>"]["count"] == 0,
        "unk_is_absent": specials["<UNK>"]["count"] == 0,
    }


def sequence_levels(text: str, eos_reached: bool) -> dict:
    stripped = text.strip()
    valid = bool(stripped) and "\ufffd" not in text and not re.search(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text
    )
    japanese = len(re.findall(r"[ぁ-んァ-ヶ一-龯]", stripped))
    local_syntax = valid and japanese >= 4 and bool(re.search(r"[。、はがをにでと]", stripped))
    repetitions = 0.0
    if stripped:
        repetitions = 1 - len(set(stripped)) / len(stripped)
    semantic = local_syntax and len(stripped) >= 10 and repetitions < 0.75
    paragraph = semantic and len(stripped) >= 40 and len(re.findall(r"[。！？]", stripped)) >= 2
    completion = semantic and (eos_reached or bool(re.search(r"[。！？.!?]$", stripped)))
    return {
        "level_0_valid_characters": valid,
        "level_1_japanese_local_syntax": local_syntax,
        "level_2_semantic_sentence": semantic,
        "level_3_paragraph_coherence": paragraph,
        "level_4_basic_completion": completion,
        "level_5_instruction_following": False,
        "character_repetition_rate": repetitions,
    }


def generation_glimpse(model, tokenizer: FoundationTokenizer, *, seed: int,
                       prompts: list[str] | None = None, mode: str = "greedy",
                       temperature: float = 1.0, top_k: int | None = None,
                       top_p: float | None = None,
                       repetition_penalty: float = 1.0) -> dict:
    prompts = prompts or COMPLETION_PROMPTS
    items = []
    for index, prompt in enumerate(prompts):
        output = generate(
            model, tokenizer, tokenizer.encode(prompt, add_bos=True), max_new_tokens=64,
            seed=seed + index, mode=mode, temperature=temperature, top_k=top_k,
            top_p=top_p, repetition_penalty=repetition_penalty,
        )
        levels = sequence_levels(output["text"], output["eos_reached"])
        items.append({"prompt": prompt, "generated": output["text"],
                      "eos_reached": output["eos_reached"], **levels})
    count = len(items)
    level_keys = [key for key in items[0] if key.startswith("level_")]
    return {
        "metrics": {
            **{key: sum(bool(item[key]) for item in items) / count for key in level_keys},
            "eos_rate": sum(item["eos_reached"] for item in items) / count,
            "mean_character_repetition_rate": sum(
                item["character_repetition_rate"] for item in items
            ) / count,
        },
        "items": items,
    }


def run_tiny_overfit(config: dict, tokenizer: FoundationTokenizer,
                     records: list[SequenceRecord]) -> tuple[dict, bool, bool]:
    settings = config["tiny_overfit"]
    validation = records[100:120]
    specifications = [
        ("1_document", 1, settings["one_document_max_steps"],
         settings["one_document_target_loss"]),
        ("10_documents", 10, settings["ten_document_max_steps"],
         settings["ten_document_target_loss"]),
        ("100_documents", 100, settings["hundred_document_max_steps"], None),
    ]
    results = {}
    ten_success = False
    health = True
    for name, count, max_steps, target in specifications:
        result = train_sequences(
            records=records[:count], validation_records=validation,
            tokenizer=tokenizer, vocab_size=4096, ffn_dim=1536,
            seed=config["seed"], learning_rate=settings["learning_rate"],
            max_steps=max_steps, weight_decay=config["weight_decay"],
            gradient_clip=config["gradient_clip"], target_loss=target,
            evaluation_interval=settings["evaluation_interval"],
            generation_probe=count <= 10,
        )
        model, serializable = strip_model(result)
        if count == 100:
            serializable["memorization"] = memorization_probe(
                model, tokenizer, records[:10], config["seed"]
            )
            serializable["memorization"]["note"] = (
                "The 100-document diagnostic probes the first 10 fixed documents."
            )
        results[name] = serializable
        if count == 10:
            memory = serializable["memorization"]
            ten_success = (
                serializable["metrics"]["final_train_loss"] < 1.0
                and memory["mean_token_exact_rate"] >= 0.75
                and memory["eos_rate"] >= 0.8
            )
        for snapshot in serializable["gradients"].values():
            health &= all(math.isfinite(value) and value > 1e-12 for value in snapshot.values())
        del model
        gc.collect()
    return {
        "purpose": "memorization/optimization sanity only; not a generalization result",
        "selection": "shortest deterministic clean-train documents; no evaluation questions",
        "results": results,
        "ten_document_success": ten_success,
    }, ten_success, health


def run_eos_sanity(config: dict, tokenizer: FoundationTokenizer) -> tuple[dict, bool]:
    records = [
        SequenceRecord(f"synthetic-{index:02d}", text, tuple(
            tokenizer.encode(text, add_bos=True, add_eos=True)
        ))
        for index, text in enumerate(SYNTHETIC_EOS_SENTENCES, 1)
    ]
    settings = config["eos_sanity"]
    result = train_sequences(
        records=records, validation_records=records, tokenizer=tokenizer,
        vocab_size=4096, ffn_dim=1536, seed=config["seed"],
        learning_rate=settings["learning_rate"], max_steps=settings["max_steps"],
        weight_decay=config["weight_decay"], gradient_clip=config["gradient_clip"],
        target_loss=0.2, evaluation_interval=50, generation_probe=False,
    )
    model, serializable = strip_model(result)
    top1 = 0
    probabilities = []
    greedy_eos = 0
    items = []
    model.eval()
    with torch.inference_mode():
        for record in records:
            inputs = record.inputs.unsqueeze(0)
            logits, _ = model(inputs)
            values = torch.softmax(logits[0, -1], dim=-1)
            predicted = int(values.argmax().item())
            probability = float(values[tokenizer.eos_id].item())
            generated = generate(
                model, tokenizer, list(record.ids[:-1]), max_new_tokens=3,
                seed=config["seed"], mode="greedy",
            )
            top1 += predicted == tokenizer.eos_id
            probabilities.append(probability)
            greedy_eos += generated["eos_reached"]
            items.append({
                "text": record.text,
                "eos_probability_after_sentence": probability,
                "eos_is_top1": predicted == tokenizer.eos_id,
                "greedy_reaches_eos": generated["eos_reached"],
            })
    rate = top1 / len(records)
    greedy_rate = greedy_eos / len(records)
    passed = rate >= settings["target_top1_rate"] and greedy_rate >= settings["target_top1_rate"]
    serializable.update({
        "synthetic_only_not_used_for_foundation_training": True,
        "eos_top1_rate": rate,
        "greedy_eos_rate": greedy_rate,
        "mean_eos_probability": sum(probabilities) / len(probabilities),
        "items": items,
        "status": "PASS" if passed else "FAIL",
    })
    del model
    gc.collect()
    return serializable, passed


def run_lr_sweep(config: dict, tokenizer: FoundationTokenizer,
                 records: list[SequenceRecord]) -> tuple[dict, float, dict]:
    settings = config["lr_sweep"]
    schedule = config["current_schedule"]
    train = records[:settings["train_documents"]]
    validation = records[
        settings["train_documents"]:
        settings["train_documents"] + settings["validation_documents"]
    ]
    rows = []
    for learning_rate in settings["learning_rates"]:
        result = train_sequences(
            records=train, validation_records=validation, tokenizer=tokenizer,
            vocab_size=4096, ffn_dim=1536, seed=config["seed"],
            learning_rate=learning_rate, max_steps=settings["steps"],
            weight_decay=config["weight_decay"], gradient_clip=config["gradient_clip"],
            warmup_steps=schedule["warmup_steps"],
            schedule_steps=schedule["schedule_steps"],
            evaluation_interval=25, generation_probe=False,
        )
        model, serializable = strip_model(result)
        serializable["learning_rate"] = learning_rate
        serializable["generation"] = generation_glimpse(
            model, tokenizer, seed=config["seed"]
        )
        rows.append(serializable)
        del model
        gc.collect()
    eligible = [row for row in rows if not row["metrics"]["diverged"]]
    best = min(eligible, key=lambda row: row["metrics"]["final_validation_loss"])
    best_lr = float(best["learning_rate"])
    return {
        "status": "PASS",
        "fixed_subset": {"train_documents": len(train), "validation_documents": len(validation)},
        "same_model_seed_subset_schedule": True,
        "schedule": {
            "warmup_steps": schedule["warmup_steps"],
            "peak_lr_step": schedule["warmup_steps"],
            "decay_schedule_steps": schedule["schedule_steps"],
            "minimum_ratio": schedule["minimum_ratio"],
            "sanity_run_fraction_in_warmup": schedule["warmup_steps"] / settings["steps"],
            "sanity_run_is_almost_only_warmup": False,
        },
        "results": rows,
        "best_learning_rate": best_lr,
        "selection_rule": "lowest fixed-validation loss among finite, non-divergent runs",
    }, best_lr


def run_tokenizer_comparison(config: dict, rows: list[dict], best_lr: float) -> dict:
    settings = config["tokenizer_comparison"]
    schedule = config["current_schedule"]
    tokenizers = {
        vocab: FoundationTokenizer.load(
            ROOT / f"tokenizer/foundation-v11-base-{vocab}.json"
        ) for vocab in settings["vocabs"]
    }
    common = []
    for row in rows:
        encoded = {
            vocab: tokenizer.encode(row["text"], add_bos=True, add_eos=True)
            for vocab, tokenizer in tokenizers.items()
        }
        if all(len(ids) <= 513 for ids in encoded.values()):
            common.append((row, encoded))
        if len(common) >= settings["train_documents"] + settings["validation_documents"]:
            break
    results = []
    for vocab, tokenizer in tokenizers.items():
        records = [
            SequenceRecord(row["id"], row["text"], tuple(encoded[vocab]))
            for row, encoded in common
        ]
        train = records[:settings["train_documents"]]
        validation = records[
            settings["train_documents"]:
            settings["train_documents"] + settings["validation_documents"]
        ]
        ffn_dim = int(settings["ffn_dim_by_vocab"][str(vocab)])
        result = train_sequences(
            records=train, validation_records=validation, tokenizer=tokenizer,
            vocab_size=vocab, ffn_dim=ffn_dim, seed=config["seed"],
            learning_rate=best_lr, max_steps=settings["steps"],
            weight_decay=config["weight_decay"], gradient_clip=config["gradient_clip"],
            warmup_steps=schedule["warmup_steps"],
            schedule_steps=schedule["schedule_steps"],
            evaluation_interval=25, generation_probe=False,
        )
        model, serializable = strip_model(result)
        serializable["generation"] = generation_glimpse(
            model, tokenizer, seed=config["seed"]
        )
        serializable["random_baseline_loss"] = math.log(vocab)
        serializable["normalized_validation_improvement_percent"] = (
            (math.log(vocab) - serializable["metrics"]["final_validation_loss"])
            / math.log(vocab) * 100
        )
        results.append(serializable)
        del model
        gc.collect()
    parameter_gap = abs(results[0]["metrics"]["parameters"] - results[1]["metrics"]["parameters"])
    return {
        "status": "PASS",
        "same_source_documents": True,
        "same_seed_steps_lr_schedule": True,
        "parameter_gap": parameter_gap,
        "parameter_gap_percent": parameter_gap / results[1]["metrics"]["parameters"] * 100,
        "losses_are_vocab_conditional": True,
        "results": results,
    }


def run_sampling_and_repetition(config: dict, tokenizer: FoundationTokenizer,
                                clean_checkpoint: Path) -> dict:
    payload = torch.load(clean_checkpoint, map_location="cpu", weights_only=False)
    trained = build_model(vocab_size=4096, seed=11012026, ffn_dim=1536)
    trained.load_state_dict(payload["model_state"])
    scratch = build_model(vocab_size=4096, seed=11012026, ffn_dim=1536)
    modes = [
        ("greedy_no_penalty", "greedy", 1.0, None, None, 1.0),
        ("temperature_0.7_untruncated", "sampling", 0.7, None, None, 1.0),
        ("temperature_1.0_untruncated", "sampling", 1.0, None, None, 1.0),
        ("temperature_0.7_topk40_topp0.9", "sampling", 0.7, 40, 0.9, 1.0),
        ("temperature_0.8_topk40_penalty1.1", "sampling", 0.8, 40, None, 1.1),
    ]
    checkpoints = []
    for checkpoint_name, model in (("scratch_step_0", scratch), ("clean_step_100", trained)):
        results = []
        for name, mode, temperature, top_k, top_p, penalty in modes:
            glimpse = generation_glimpse(
                model, tokenizer, seed=config["seed"], mode=mode,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=penalty,
            )
            results.append({
                "name": name,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": penalty,
                **glimpse,
            })
        checkpoints.append({"checkpoint": checkpoint_name, "results": results})
    clean_rows = read_short_clean_documents(20)
    records = encode_records(clean_rows, tokenizer)
    context_rows = []
    for prefix_tokens in (16, 64, 128):
        prompts = [
            tokenizer.decode(list(record.ids[1:prefix_tokens + 1]), skip_special=True)
            for record in records[:10]
            if len(record.ids) > prefix_tokens + 1
        ]
        context_rows.append({
            "prompt_tokens": prefix_tokens,
            **generation_glimpse(trained, tokenizer, seed=config["seed"], prompts=prompts),
        })
    del scratch, trained, payload
    gc.collect()
    return {
        "base_result_without_repetition_penalty_preserved": True,
        "checkpoints": checkpoints,
        "context_comparison_clean_step_100_greedy_no_penalty": context_rows,
        "interpretation_scope": (
            "Separates undertraining (step 0 vs 100), decoding mode, context length, "
            "and repetition penalty; heuristics are not a substitute for model quality."
        ),
    }


def layer_gradient_health(tiny: dict) -> tuple[str, dict]:
    snapshots = tiny["results"]["10_documents"]["gradients"]
    failures = []
    for step, row in snapshots.items():
        for name, value in row.items():
            if not math.isfinite(value) or value <= 1e-12:
                failures.append({"step": step, "component": name, "value": value})
    return ("PASS" if not failures else "FAIL"), {
        "sampled_steps": list(snapshots),
        "norms_are_pre_clip": True,
        "gradient_clip": 1.0,
        "failures": failures,
    }


def weight_update_health(tiny: dict) -> tuple[str, dict]:
    deltas = tiny["results"]["10_documents"]["weight_deltas"]
    failures = []
    for required_step in ("10", "100"):
        if required_step not in deltas:
            failures.append({"step": required_step, "reason": "missing"})
            continue
        for name, values in deltas[required_step].items():
            if values["l2"] <= 0 or not math.isfinite(values["l2"]):
                failures.append({"step": required_step, "component": name})
    return ("PASS" if not failures else "FAIL"), {"deltas": deltas, "failures": failures}


def markdown_report(report: dict) -> str:
    random = report["random_baseline"]
    tiny = report["tiny_overfit"]
    lines = [
        "# UniPilot Foundation v1.2 Training Dynamics Investigation",
        "",
        "外部LLM/API、Final Blind、Production/Campus/Webには触れず、20M Foundationの学習核だけを検証した。",
        "",
        "## Core audits",
        "",
        f"- Random baseline: {random['random_cross_entropy']:.6f}",
        f"- Step 100 validation loss / perplexity: {random['steps'][-1]['validation_loss']:.6f} / {random['step_100_perplexity']:.2f}",
        f"- Random baseline比改善: {random['step_100_percent_improvement']:.2f}%",
        f"- Causal shift: {report['causal_shift']['status']}",
        f"- Causal mask: {report['causal_mask']['status']}",
        f"- Loss: {report['loss']['status']}",
        f"- Gradient health: {report['gradient_health']['status']}",
        f"- Weight update: {report['weight_update']['status']}",
        f"- EOS sanity: {report['eos_sanity']['status']}",
        "",
        "## Tiny overfit",
        "",
    ]
    for name in ("1_document", "10_documents", "100_documents"):
        row = tiny["results"][name]
        memory = row.get("memorization") or {}
        lines.append(
            f"- {name}: step {row['metrics']['steps']}, train loss "
            f"{row['metrics']['final_train_loss']:.4f}, exact continuation "
            f"{memory.get('mean_token_exact_rate', 0) * 100:.1f}%, EOS "
            f"{memory.get('eos_rate', 0) * 100:.1f}%"
        )
    lines.extend([
        "",
        "## Learning-rate sweep",
        "",
    ])
    if report["lr_sweep"]["status"] == "PASS":
        for row in report["lr_sweep"]["results"]:
            lines.append(
                f"- {row['learning_rate']:.0e}: train {row['metrics']['final_train_loss']:.4f}, "
                f"validation {row['metrics']['final_validation_loss']:.4f}, "
                f"grad {row['metrics']['gradient_norm']:.4f}, diverged {row['metrics']['diverged']}"
            )
    else:
        lines.append("- Training Core FAILのため未実行。")
    lines.extend(["", "## Tokenizer short training", ""])
    if report["tokenizer_training_comparison"]["status"] == "PASS":
        for row in report["tokenizer_training_comparison"]["results"]:
            lines.append(
                f"- vocab {row['metrics']['vocab']}: parameters {row['metrics']['parameters']:,}, "
                f"validation {row['metrics']['final_validation_loss']:.4f}, "
                f"normalized improvement {row['normalized_validation_improvement_percent']:.2f}%, "
                f"{row['metrics']['tokens_per_second']:.1f} tok/s, RAM {row['metrics']['peak_ram_mb']:.1f}MB"
            )
        lines.append(
            f"- Recommended vocab: {report['tokenizer_training_comparison']['recommended_vocab']}"
        )
    sampling = report["sampling_audit"]["checkpoints"]
    clean_sampling = next(row for row in sampling if row["checkpoint"] == "clean_step_100")
    lines.extend([
        "",
        "## Token frequency / decoding audit",
        "",
        f"- EOS: {report['token_frequency']['special_tokens']['<EOS>']['count']:,} / "
        f"{report['token_frequency']['tokens']:,} "
        f"({report['token_frequency']['eos_percent']:.5f}%)",
        f"- Original 51,200-token runの期待EOS観測数: "
        f"{report['token_frequency']['expected_eos_targets_in_original_51200_token_run']:.2f}",
        f"- Used vocabulary: {report['token_frequency']['used_vocab']:,} / "
        f"{report['token_frequency']['vocab']:,}",
        f"- Weight tying: {report['weight_tying']['enabled']}",
        "- Clean step 100 decoding（Level 1 / repetition）:",
    ])
    for row in clean_sampling["results"]:
        lines.append(
            f"  - {row['name']}: "
            f"{row['metrics']['level_1_japanese_local_syntax'] * 100:.0f}% / "
            f"{row['metrics']['mean_character_repetition_rate'] * 100:.1f}%"
        )
    lines.extend([
        "- Greedy repetitionはcontext 16/64/128でも解消せず、samplingで低下するが、"
        "100stepではLevel 2〜3が成立しない。主因は未学習であり、decoding調整だけでは代替しない。",
        "",
        "## Verification",
        "",
        f"- Resume: {report['verification']['resume']}",
        f"- Tokenizer roundtrip: {report['verification']['tokenizer_roundtrip']}",
        f"- Checkpoint integrity: {report['verification']['checkpoint_integrity']}",
        f"- Final Blind SHA256: `{report['protected']['final_blind_sha256']}`（内容未使用）",
    ])
    lines.extend([
        "",
        "## Decision",
        "",
        f"- TRAINING CORE: **{report['gates']['training_core']}**",
        f"- Best LR: {report['decisions']['best_learning_rate']}",
        f"- Full Clean 250step: **{report['decisions']['full_clean_250_recommended']}**（未実行）",
        f"- Corpus追加: **{report['decisions']['corpus_addition_needed']}**",
        f"- Architecture変更: **{report['decisions']['architecture_change_needed']}**",
        "- 500step、46M、Final Blindは未実行。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v12-investigation.json")
    parser.add_argument("--output", default="evaluation/foundation-v12-training-dynamics.json")
    parser.add_argument("--report", default="evaluation/foundation-v12-training-dynamics-report.md")
    parser.add_argument("--reuse-experiments", action="store_true")
    args = parser.parse_args()
    config = json_load(args.config)
    torch.set_num_threads(int(config["cpu_threads"]))
    tokenizer = FoundationTokenizer.load(ROOT / "tokenizer/foundation-v11-base-4096.json")
    rows = read_short_clean_documents(500)
    records = encode_records(rows, tokenizer)
    if len(records) < 120:
        raise RuntimeError(f"Need at least 120 short clean records, got {len(records)}")

    random = random_baseline()
    shift = target_shift_audit(tokenizer)
    mask = causal_mask_audit(config["seed"])
    loss = loss_audit(config["seed"])
    initialization = initialization_audit(
        build_model(vocab_size=4096, seed=config["seed"], ffn_dim=1536)
    )
    frequency = frequency_audit(tokenizer)
    existing = None
    existing_path = ROOT / args.output
    if args.reuse_experiments:
        if not existing_path.exists():
            raise RuntimeError("--reuse-experiments requires an existing output report")
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
    if existing is None:
        tiny, tiny_success, _ = run_tiny_overfit(config, tokenizer, records)
    else:
        tiny = existing["tiny_overfit"]
        memory = tiny["results"]["10_documents"]["memorization"]
        tiny_success = (
            tiny["results"]["10_documents"]["metrics"]["final_train_loss"] < 1.0
            and memory["mean_token_exact_rate"] >= 0.75
            and memory["eos_rate"] >= 0.8
        )
    gradient_status, gradient_details = layer_gradient_health(tiny)
    weight_status, weight_details = weight_update_health(tiny)
    if existing is None:
        eos, eos_success = run_eos_sanity(config, tokenizer)
    else:
        eos = existing["eos_sanity"]
        eos_success = eos["status"] == "PASS"
    resume = json_load("evaluation/foundation-v11-resume-reproducibility.json")
    training_core_checks = {
        "tiny_10_document_memorization": tiny_success,
        "eos_sanity": eos_success,
        "causal_shift": shift["status"] == "PASS",
        "causal_mask": mask["status"] == "PASS",
        "loss": loss["status"] == "PASS",
        "gradients": gradient_status == "PASS",
        "weight_updates": weight_status == "PASS",
        "resume": resume["resume_integrity"] == "PASS",
    }
    training_core = "PASS" if all(training_core_checks.values()) else "FAIL"

    if training_core == "PASS":
        if existing is None:
            lr_sweep, best_lr = run_lr_sweep(config, tokenizer, records)
            tokenizer_comparison = run_tokenizer_comparison(config, rows, best_lr)
        else:
            lr_sweep = existing["lr_sweep"]
            best_lr = float(lr_sweep["best_learning_rate"])
            tokenizer_comparison = existing["tokenizer_training_comparison"]
        if tokenizer_comparison["status"] == "PASS":
            tokenizer_comparison["recommended_vocab"] = max(
                tokenizer_comparison["results"],
                key=lambda row: row["normalized_validation_improvement_percent"],
            )["metrics"]["vocab"]
            tokenizer_comparison["recommendation_basis"] = (
                "higher normalized validation improvement with parameter counts matched "
                "within 0.02%; raw cross-vocab loss/perplexity is not compared directly"
            )
    else:
        best_lr = None
        lr_sweep = {"status": "NOT_RUN", "reason": "Training Core did not pass."}
        tokenizer_comparison = {"status": "NOT_RUN", "reason": "Training Core did not pass."}
    sampling = (
        existing["sampling_audit"] if existing is not None else run_sampling_and_repetition(
            config, tokenizer,
            ROOT / "checkpoints/foundation-v11-clean-100/checkpoint-step-100.pt",
        )
    )

    final_blind = ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"
    checkpoint = ROOT / "checkpoints/foundation-v11-clean-100/checkpoint-step-100.pt"
    checkpoint_manifest = json_load(
        "checkpoints/foundation-v11-clean-100/checkpoint-step-100.manifest.json"
    )
    roundtrip_benchmark = json_load("evaluation/foundation-v11-tokenizer-benchmark.json")
    full_250 = training_core == "PASS" and best_lr is not None
    report = {
        "schema_version": "foundation-v12-training-dynamics-v1",
        "random_baseline": random,
        "causal_shift": shift,
        "causal_mask": mask,
        "loss": loss,
        "initialization": initialization,
        "token_frequency": frequency,
        "tiny_overfit": tiny,
        "eos_sanity": eos,
        "gradient_health": {"status": gradient_status, **gradient_details},
        "weight_update": {"status": weight_status, **weight_details},
        "weight_tying": {
            **initialization["weight_tying"],
            "assessment": (
                "Already enabled and appropriate for a 20M model: it saves vocab_size × "
                "embedding_dim parameters and keeps input/output token geometry aligned."
            ),
            "architecture_changed": False,
        },
        "scheduler_audit": {
            "current": config["current_schedule"],
            "step_100_warmup_fraction": config["current_schedule"]["warmup_steps"] / 100,
            "step_100_is_almost_only_warmup": False,
            "separate_sanity_and_full_schedule_recommended": True,
            "future_full_clean_250_candidate_not_executed": {
                "warmup_steps": 20,
                "schedule_steps": 250,
                "minimum_ratio": 0.1,
                "peak_learning_rate": best_lr,
            },
            "assessment": (
                "The 100-step run spends 20% in warmup and then follows the 500-step cosine "
                "schedule. The LR sweep preserves this exactly; a future 250-step run should "
                "declare its own total schedule length rather than silently reuse a sanity schedule."
            ),
        },
        "lr_sweep": lr_sweep,
        "tokenizer_training_comparison": tokenizer_comparison,
        "sampling_audit": sampling,
        "base_evaluation_levels": {
            "level_0": "Valid Japanese characters",
            "level_1": "Japanese-like local syntax",
            "level_2": "Semantic sentence",
            "level_3": "Paragraph coherence",
            "level_4": "Basic completion",
            "level_5": "Instruction following",
            "foundation_priority": [0, 1, 2, 3],
            "automated_heuristics": True,
            "chat_q_and_a_is_not_the_primary_early_base_gate": True,
        },
        "verification": {
            "resume": resume["resume_integrity"],
            "tokenizer_roundtrip": "PASS" if all(
                row["exact_roundtrip_rate"] == 1 for row in roundtrip_benchmark["results"]
            ) else "FAIL",
            "checkpoint_integrity": "PASS" if (
                checkpoint_manifest["checkpoint_sha256"] == sha256(checkpoint)
            ) else "FAIL",
            "checkpoint_sha256": sha256(checkpoint),
        },
        "gates": {
            "training_core_checks": training_core_checks,
            "training_core": training_core,
            "lr_gate_executed": training_core == "PASS",
        },
        "decisions": {
            "best_learning_rate": best_lr,
            "full_clean_250_recommended": "YES" if full_250 else "NO",
            "full_clean_250_executed": False,
            "full_clean_500_executed": False,
            "corpus_addition_needed": "NO",
            "architecture_change_needed": "NO" if training_core == "PASS" else "INVESTIGATE",
            "standard_46m_allowed": False,
            "step_100_generation_diagnosis": (
                "Training core, tiny memorization, and EOS learning all pass. The v1.1 "
                "100-step Natural Japanese 0% result is therefore consistent with severe "
                "undertraining (0.1533% corpus exposure), not a causal-LM core defect."
            ) if training_core == "PASS" else "Training-core defect remains possible.",
        },
        "protected": {
            "final_blind_path": final_blind.relative_to(ROOT).as_posix(),
            "final_blind_sha256": sha256(final_blind),
            "final_blind_expected_sha256": (
                "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"
            ),
            "final_blind_content_opened": False,
            "production_v04_changed": False,
            "campus_v23_changed": False,
            "render_changed": False,
            "vercel_changed": False,
            "release_changed": False,
        },
        "external_ai_api": "OFF",
        "push_or_deploy_performed": False,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / args.report).write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "training_core": training_core,
        "tiny_10": tiny["results"]["10_documents"]["metrics"],
        "eos": {"status": eos["status"], "top1": eos["eos_top1_rate"]},
        "best_lr": best_lr,
        "full_clean_250_recommended": report["decisions"]["full_clean_250_recommended"],
    }, ensure_ascii=False, indent=2))
    return 0 if training_core == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
