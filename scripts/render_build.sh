#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install "torch==2.12.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-prod.txt
python -m scripts.download_production_checkpoint
