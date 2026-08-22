from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request


V04_INFERENCE_URL = "https://github.com/kiyoshike-lab/unipilot-mini/releases/download/v0.4-model/unipilot-v04-inference-step-2000.pt"
V04_INFERENCE_SHA256 = "72ccad96d4d1fe75d55cae94240308ef2cf6372132c73dac6f3ac5a010ae4a03"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the production v0.4 inference checkpoint.")
    parser.add_argument("--url", default=V04_INFERENCE_URL)
    parser.add_argument("--sha256", default=V04_INFERENCE_SHA256)
    parser.add_argument("--output", default="checkpoints/v04-eos15/unipilot-mini-v04-inference.pt")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = args.sha256.lower()
    if output.exists() and sha256(output) == expected:
        print(f"checkpoint already verified: {output}")
        return
    temporary = output.with_suffix(output.suffix + ".download")
    request = urllib.request.Request(args.url, headers={"User-Agent": "UniPilot-Mini-Render-build/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
        actual = sha256(temporary)
        if actual != expected:
            raise RuntimeError(f"checkpoint SHA256 mismatch: expected {expected}, got {actual}")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"checkpoint downloaded and verified: {output}")


if __name__ == "__main__":
    main()
