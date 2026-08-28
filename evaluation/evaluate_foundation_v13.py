from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from training.train_foundation_v10 import bigram_overlap, natural_text
from training.train_foundation_v13 import CHECKPOINT_FORMAT


KNOWLEDGE_PROBES = [
    ("日本の首都は", ["東京"]),
    ("1日は24", ["時間"]),
    ("水は液体の", ["一種", "状態"]),
    ("大学は高等教育", ["機関", "教育"]),
    ("地球は太陽の周りを", ["公転"]),
    ("春の次に来る季節は", ["夏"]),
    ("植物は光を使って", ["光合成"]),
    ("物体を落とすと", ["重力", "落下"]),
    ("人は言葉を使って", ["伝", "コミュニケーション"]),
    ("文章を書く前に", ["構成", "目的", "考"]),
]

PRIMARY_MODES = [
    {
        "name": "greedy_no_penalty",
        "kind": "greedy",
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.0,
    },
    {
        "name": "sampling_t07_topk40_topp09_no_penalty",
        "kind": "sampling",
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
    },
]

STEP_250_EXTRA_MODES = [
    {
        "name": "sampling_t07_untruncated_no_penalty",
        "kind": "sampling",
        "temperature": 0.7,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.0,
    },
    {
        "name": "sampling_t10_untruncated_no_penalty",
        "kind": "sampling",
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.0,
    },
    {
        "name": "sampling_t10_topk40_topp09_no_penalty",
        "kind": "sampling",
        "temperature": 1.0,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
    },
]


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_scratch_model(settings: dict, corpus: dict) -> UniPilotTransformer:
    seed_all(int(settings["seed"]))
    config = ModelConfig(
        model_name=settings["model_name"],
        vocab_size=int(corpus["vocab"]),
        **settings["model"],
    )
    model = UniPilotTransformer(config)
    model.eval()
    return model


def load_checkpoint_model(path: Path) -> tuple[UniPilotTransformer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Unexpected checkpoint format: {path}")
    model = UniPilotTransformer(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def sample_token(scores: torch.Tensor, mode: dict, generator: torch.Generator) -> int:
    if mode["kind"] == "greedy":
        return int(scores.argmax().item())
    adjusted = scores.float() / float(mode["temperature"])
    top_k = mode["top_k"]
    if top_k is not None and 0 < top_k < adjusted.numel():
        threshold = torch.topk(adjusted, top_k).values[-1]
        adjusted[adjusted < threshold] = -torch.inf
    top_p = mode["top_p"]
    if top_p is not None and top_p < 1.0:
        ordered, indices = torch.sort(adjusted, descending=True)
        probabilities = torch.softmax(ordered, dim=-1)
        remove = torch.cumsum(probabilities, dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        adjusted[indices[remove]] = -torch.inf
    probabilities = torch.softmax(adjusted, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


@torch.inference_mode()
def generate(model: UniPilotTransformer, tokenizer: FoundationTokenizer, prompt: str,
             mode: dict, seed: int, max_new_tokens: int = 64) -> dict:
    ids = tokenizer.encode(prompt, add_bos=True)
    generated: list[int] = []
    eos_probabilities: list[float] = []
    past = None
    generator = torch.Generator().manual_seed(seed)
    forbidden = [
        token_id for token, token_id in tokenizer.special_to_id.items()
        if token not in {"<EOS>"}
    ]
    started = time.perf_counter()
    first_token_seconds = None
    for _ in range(max_new_tokens):
        current_ids = ids[-model.config.context_length:] if past is None else [ids[-1]]
        logits, _, past = model(
            torch.tensor([current_ids], dtype=torch.long),
            past_key_values=past,
            use_cache=True,
        )
        raw_scores = logits[0, -1].float()
        eos_probabilities.append(float(torch.softmax(raw_scores, dim=-1)[tokenizer.eos_id].item()))
        scores = raw_scores.clone()
        scores[forbidden] = -torch.inf
        next_id = sample_token(scores, mode, generator)
        if first_token_seconds is None:
            first_token_seconds = time.perf_counter() - started
        ids.append(next_id)
        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    elapsed = time.perf_counter() - started
    return {
        "text": tokenizer.decode(generated, skip_special=True),
        "ids": generated,
        "tokens": len(generated),
        "eos_reached": bool(generated and generated[-1] == tokenizer.eos_id),
        "first_step_eos_probability": eos_probabilities[0],
        "mean_eos_probability": sum(eos_probabilities) / len(eos_probabilities),
        "maximum_eos_probability": max(eos_probabilities),
        "first_token_seconds": first_token_seconds or 0.0,
        "total_seconds": elapsed,
        "tokens_per_second": len(generated) / max(elapsed, 1e-9),
    }


def valid_characters(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and "\ufffd" not in text and not re.search(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text
    )


def score_generation(prompt_row: dict, generation: dict) -> dict:
    text = generation["text"]
    valid = valid_characters(text)
    natural, repetition = natural_text(text)
    keyword_hit = any(
        keyword in text for keyword in prompt_row.get("expected_keywords", [])
    )
    reference = prompt_row.get("reference", "")
    overlap = bigram_overlap(text, reference) if reference else 0.0
    aligned = keyword_hit or (
        prompt_row.get("kind") == "heldout_continuation" and overlap >= 0.02
    )
    stripped = text.rstrip()
    sentence_ends = len(re.findall(r"[。！？.!?]", stripped))
    complete = valid and natural and bool(
        generation["eos_reached"] or re.search(r"[。！？.!?]$", stripped)
    )
    semantic = natural and repetition < 0.35 and (sentence_ends > 0 or aligned)
    paragraph = semantic and len(stripped) >= 60 and sentence_ends >= 2 and repetition < 0.25
    level_flags = {
        "level_0_valid_characters": valid,
        "level_1_japanese_local_syntax": valid and natural,
        "level_2_semantic_sentence": valid and semantic,
        "level_3_paragraph_coherence": valid and paragraph,
        "level_4_prompt_aligned_completion": valid and semantic and aligned and complete,
        "level_5_instruction_following": False,
    }
    highest = max(
        [index for index in range(6) if level_flags[list(level_flags)[index]]] or [-1]
    )
    return {
        **prompt_row,
        "generated": text,
        "character_valid": valid,
        "natural_japanese": valid and natural,
        "semantic_coherence": valid and semantic,
        "completion": complete,
        "prompt_aligned": aligned,
        "keyword_hit": keyword_hit,
        "reference_bigram_overlap": overlap,
        "repetition_rate": repetition,
        "runaway": generation["tokens"] >= 64 and not generation["eos_reached"],
        "highest_level": highest,
        **level_flags,
        **{key: value for key, value in generation.items() if key != "ids"},
    }


def aggregate(items: list[dict]) -> dict:
    count = len(items)
    level_keys = [
        "level_0_valid_characters",
        "level_1_japanese_local_syntax",
        "level_2_semantic_sentence",
        "level_3_paragraph_coherence",
        "level_4_prompt_aligned_completion",
        "level_5_instruction_following",
    ]
    distribution = Counter(row["highest_level"] for row in items)
    return {
        "prompts": count,
        "character_validity_rate": sum(row["character_valid"] for row in items) / count,
        "natural_japanese_rate": sum(row["natural_japanese"] for row in items) / count,
        "semantic_coherence_rate": sum(row["semantic_coherence"] for row in items) / count,
        "completion_rate": sum(row["completion"] for row in items) / count,
        "prompt_alignment_rate": sum(row["prompt_aligned"] for row in items) / count,
        "eos_rate": sum(row["eos_reached"] for row in items) / count,
        "runaway_rate": sum(row["runaway"] for row in items) / count,
        "mean_repetition_rate": sum(row["repetition_rate"] for row in items) / count,
        "mean_first_step_eos_probability": sum(
            row["first_step_eos_probability"] for row in items
        ) / count,
        "mean_generation_eos_probability": sum(
            row["mean_eos_probability"] for row in items
        ) / count,
        "mean_maximum_eos_probability": sum(
            row["maximum_eos_probability"] for row in items
        ) / count,
        "mean_first_token_seconds": sum(row["first_token_seconds"] for row in items) / count,
        "mean_tokens_per_second": sum(row["tokens_per_second"] for row in items) / count,
        "level_pass_rates": {
            key: sum(row[key] for row in items) / count for key in level_keys
        },
        "highest_level_distribution": {
            str(level): {"count": int(distribution.get(level, 0)),
                         "rate": distribution.get(level, 0) / count}
            for level in range(-1, 6)
        },
    }


def evaluate_mode(model: UniPilotTransformer, tokenizer: FoundationTokenizer,
                  prompts: list[dict], mode: dict, seed: int) -> dict:
    items = []
    for index, row in enumerate(prompts):
        generated = generate(model, tokenizer, row["prompt"], mode, seed + index)
        items.append(score_generation(row, generated))
    return {"settings": mode, "metrics": aggregate(items), "items": items}


def validation_eos_documents(tokenizer: FoundationTokenizer, limit: int = 32) -> list[list[int]]:
    rows = []
    path = ROOT / "data/foundation_v11/documents/validation.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text, add_bos=True, add_eos=False)
            if len(ids) > 1:
                rows.append(ids[-512:])
    rows.sort(key=len)
    return rows[:limit]


@torch.inference_mode()
def eos_document_probe(model: UniPilotTransformer, tokenizer: FoundationTokenizer,
                       documents: list[list[int]]) -> dict:
    probabilities = []
    top1 = 0
    model.eval()
    for ids in documents:
        logits, _ = model(torch.tensor([ids], dtype=torch.long))
        scores = torch.softmax(logits[0, -1].float(), dim=-1)
        probabilities.append(float(scores[tokenizer.eos_id].item()))
        top1 += int(scores.argmax().item()) == tokenizer.eos_id
    return {
        "documents": len(documents),
        "mean_eos_probability_after_complete_document": sum(probabilities) / len(probabilities),
        "minimum_eos_probability": min(probabilities),
        "maximum_eos_probability": max(probabilities),
        "eos_top1_rate": top1 / len(documents),
    }


def knowledge_probe(model: UniPilotTransformer, tokenizer: FoundationTokenizer,
                    mode: dict, seed: int) -> dict:
    items = []
    for index, (prompt, keywords) in enumerate(KNOWLEDGE_PROBES):
        generated = generate(model, tokenizer, prompt, mode, seed + index)
        natural, repetition = natural_text(generated["text"])
        hits = [keyword for keyword in keywords if keyword in generated["text"]]
        items.append({
            "prompt": prompt,
            "expected_keywords": keywords,
            "generated": generated["text"],
            "keyword_hits": hits,
            "keyword_hit": bool(hits),
            "character_valid": valid_characters(generated["text"]),
            "natural_japanese": valid_characters(generated["text"]) and natural,
            "repetition_rate": repetition,
            "eos_reached": generated["eos_reached"],
        })
    return {
        "mode": mode["name"],
        "prompts": len(items),
        "keyword_hit_rate": sum(row["keyword_hit"] for row in items) / len(items),
        "natural_japanese_rate": sum(row["natural_japanese"] for row in items) / len(items),
        "items": items,
        "interpretation": "emergence observation only; not a memorization or promotion gate",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v13.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/foundation-v13-clean-250")
    parser.add_argument("--output", default="evaluation/foundation-v13-generation.json")
    parser.add_argument("--rescore-existing", action="store_true")
    args = parser.parse_args()
    if args.rescore_existing:
        output = ROOT / args.output
        report = json.loads(output.read_text(encoding="utf-8"))
        for result in report["results"]:
            mode_groups = [result["modes"]]
            if "sampling_observation" in result:
                mode_groups.append(result["sampling_observation"])
            for modes in mode_groups:
                for mode_result in modes.values():
                    rescored_items = []
                    for item in mode_result["items"]:
                        prompt_row = {
                            key: item[key] for key in (
                                "id", "kind", "prompt", "expected_keywords", "reference"
                            )
                        }
                        generation = {
                            "text": item["generated"],
                            "tokens": item["tokens"],
                            "eos_reached": item["eos_reached"],
                            "first_step_eos_probability": item["first_step_eos_probability"],
                            "mean_eos_probability": item["mean_eos_probability"],
                            "maximum_eos_probability": item["maximum_eos_probability"],
                            "first_token_seconds": item["first_token_seconds"],
                            "total_seconds": item["total_seconds"],
                            "tokens_per_second": item["tokens_per_second"],
                            "ids": [],
                        }
                        rescored_items.append(score_generation(prompt_row, generation))
                    mode_result["items"] = rescored_items
                    mode_result["metrics"] = aggregate(rescored_items)
            for probe in result["knowledge_probe"].values():
                for item in probe["items"]:
                    natural, repetition = natural_text(item["generated"])
                    item["character_valid"] = valid_characters(item["generated"])
                    item["natural_japanese"] = item["character_valid"] and natural
                    item["repetition_rate"] = repetition
                probe["keyword_hit_rate"] = sum(
                    item["keyword_hit"] for item in probe["items"]
                ) / len(probe["items"])
                probe["natural_japanese_rate"] = sum(
                    item["natural_japanese"] for item in probe["items"]
                ) / len(probe["items"])
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "rescored": True,
            "steps": [row["step"] for row in report["results"]],
        }, ensure_ascii=False, indent=2))
        return 0
    settings = load_json(args.config)
    torch.set_num_threads(int(settings["cpu_threads"]))
    corpus = load_json(settings["corpus_manifest"])
    tokenizer = FoundationTokenizer.load(ROOT / corpus["tokenizer"])
    prompt_data = load_json(settings["completion_prompts"])
    prompts = prompt_data["items"]
    if len(prompts) < 50:
        raise RuntimeError("Foundation v1.3 requires at least 50 fixed completion prompts")
    prompts = prompts[:50]
    eos_documents = validation_eos_documents(tokenizer)
    steps = [0, 50, 100, 150, 200, 250]
    results = []
    process = psutil.Process(os.getpid())
    for step in steps:
        payload = None
        if step == 0:
            model = build_scratch_model(settings, corpus)
            source = "deterministic scratch initialization"
        else:
            checkpoint = ROOT / args.checkpoint_dir / f"checkpoint-step-{step}.pt"
            model, payload = load_checkpoint_model(checkpoint)
            if int(payload["global_step"]) != step:
                raise RuntimeError(f"checkpoint step mismatch: {checkpoint}")
            source = checkpoint.relative_to(ROOT).as_posix()
        modes = {}
        for mode_index, mode in enumerate(PRIMARY_MODES):
            modes[mode["name"]] = evaluate_mode(
                model, tokenizer, prompts, mode,
                seed=int(settings["seed"]) + mode_index * 100_000,
            )
        knowledge = {
            mode["name"]: knowledge_probe(
                model, tokenizer, mode,
                seed=int(settings["seed"]) + 500_000 + mode_index * 100_000,
            )
            for mode_index, mode in enumerate(PRIMARY_MODES)
        }
        result = {
            "step": step,
            "source": source,
            "parameters": model.parameter_count(),
            "modes": modes,
            "eos_document_probe": eos_document_probe(model, tokenizer, eos_documents),
            "knowledge_probe": knowledge,
            "process_rss_mb": process.memory_info().rss / 1024**2,
        }
        if step == 250:
            result["sampling_observation"] = {
                mode["name"]: evaluate_mode(
                    model, tokenizer, prompts, mode,
                    seed=int(settings["seed"]) + 900_000 + index * 100_000,
                )
                for index, mode in enumerate(STEP_250_EXTRA_MODES)
            }
        results.append(result)
        del model, payload

    report = {
        "schema_version": "foundation-v13-generation-evaluation-v1",
        "steps": steps,
        "prompts": len(prompts),
        "prompt_source": settings["completion_prompts"],
        "prompt_style": "base text completion, not instruction Q&A",
        "primary_modes": PRIMARY_MODES,
        "repetition_penalty_used_in_base_evaluation": False,
        "max_new_tokens": 64,
        "results": results,
        "level_definitions": {
            "0": "valid Japanese characters",
            "1": "Japanese-like local syntax",
            "2": "semantic sentence",
            "3": "multi-sentence semantic coherence",
            "4": "prompt-aligned completion",
            "5": "instruction following (not a Foundation v1.3 primary target)",
        },
        "automated_heuristics": True,
        "final_blind_used": False,
        "external_ai_api": "OFF",
        "production_changed": False,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "steps": steps,
        "results": [
            {
                "step": row["step"],
                "modes": {
                    name: value["metrics"] for name, value in row["modes"].items()
                },
                "eos_document_probe": row["eos_document_probe"],
            }
            for row in results
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
