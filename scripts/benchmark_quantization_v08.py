from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile
import time

import psutil
import torch

from inference.generate import generate_text, load_model


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def generation_probe(model, tokenizer) -> dict:
    prompt = (
        "<BOS><SYSTEM>\nあなたは大学生活支援に特化した完全ローカルのUniPilot Standardです。\n"
        "<CONTEXT>\n<USER>\n試験と課題の期限が重なりました。優先順位を教えてください。\n<ASSISTANT>\n"
    )
    started = time.perf_counter()
    text, metrics = generate_text(model, tokenizer, prompt, 64, temperature=0.0, top_k=40,
                                  top_p=0.9, repetition_penalty=1.1, stop_on_eos=False)
    return {"text": text, "seconds": time.perf_counter() - started, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/standard-v08-scratch/unipilot-standard-v08-a100-inference.pt")
    parser.add_argument("--tokenizer", default="tokenizer/vocab-standard-v08-1024.json")
    parser.add_argument("--output", default="evaluation/quantization-benchmark-standard-v08.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, tokenizer, _, _ = load_model(args.checkpoint, args.tokenizer, "cpu")
    fp32_rss = rss_mb()
    fp32_probe = generation_probe(model, tokenizer)
    fixed = torch.tensor([tokenizer.encode("大学生活の計画を立てる")], dtype=torch.long)
    with torch.inference_mode():
        fp32_logits, _ = model(fixed)
        fp32_top = fp32_logits.argmax(-1)
    try:
        from torch.ao.quantization import quantize_dynamic
        quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False).eval()
        implementation = "torch.ao.quantization.quantize_dynamic"
    except Exception as error:
        report = {"supported": False, "error": repr(error), "adopted": False}
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return
    del model
    gc.collect()
    int8_rss = rss_mb()
    int8_probe = generation_probe(quantized, tokenizer)
    with torch.inference_mode():
        int8_logits, _ = quantized(fixed)
        int8_top = int8_logits.argmax(-1)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dynamic-int8-state.pt"
        torch.save({"model_state": quantized.state_dict(), "dynamic_int8": True}, path)
        quantized_bytes = path.stat().st_size
    fp32_bytes = Path(args.checkpoint).stat().st_size
    top_agreement = (fp32_top == int8_top).float().mean().item()
    report = {
        "supported": True, "implementation": implementation,
        "fp32": {"checkpoint_bytes": fp32_bytes, "rss_after_load_mb": fp32_rss, "probe": fp32_probe},
        "dynamic_int8": {"state_dict_bytes": quantized_bytes, "rss_after_fp32_deleted_mb": int8_rss, "probe": int8_probe},
        "size_reduction": 1 - quantized_bytes / fp32_bytes,
        "rss_reduction": 1 - int8_rss / fp32_rss,
        "tokens_per_second_ratio": int8_probe["tokens_per_sec"] / max(fp32_probe["tokens_per_sec"], 1e-9),
        "fixed_prompt_top1_agreement": top_agreement,
        "generated_text_exact_match": fp32_probe["text"] == int8_probe["text"],
        "adopted": False,
        "decision": "Do not adopt from a failed Stage-A quality checkpoint; repeat quantization quality evaluation after a usable FP32 checkpoint exists.",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
