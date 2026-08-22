from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


KEEP_KEYS = (
    "model_state", "config", "tokenizer_version", "step", "loss",
    "v02_manifest", "v03_manifest", "v04_manifest", "v05_manifest",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a training checkpoint without optimizer/training state.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    if source.resolve() == destination.resolve():
        raise ValueError("output must differ from the original checkpoint")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    exported = {key: payload[key] for key in KEEP_KEYS if key in payload}
    required = {"model_state", "config"}
    if not required.issubset(exported):
        raise ValueError(f"checkpoint is missing required keys: {sorted(required - exported.keys())}")
    exported["inference_only"] = True
    exported["source_checkpoint"] = str(source).replace("\\", "/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, destination)
    report = {
        "source": str(source).replace("\\", "/"),
        "output": str(destination).replace("\\", "/"),
        "source_size_mb": source.stat().st_size / 1024**2,
        "output_size_mb": destination.stat().st_size / 1024**2,
        "kept_keys": sorted(exported),
        "removed_keys": sorted(set(payload) - set(exported)),
    }
    destination.with_suffix(".export.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
