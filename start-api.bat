@echo off
cd /d "%~dp0"
set UNIPILOT_CHECKPOINT=checkpoints/v03-scratch-001/stage-c/checkpoint-step-5000.pt
set UNIPILOT_TOKENIZER=tokenizer/vocab-v02-512.json
set UNIPILOT_DEV_MODE=1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
