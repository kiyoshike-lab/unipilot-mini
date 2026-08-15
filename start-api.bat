@echo off
cd /d "%~dp0"
set UNIPILOT_CHECKPOINT=checkpoints/v04-eos15/checkpoint-step-2000.pt
set UNIPILOT_TOKENIZER=tokenizer/vocab-v02-512.json
set UNIPILOT_DEV_MODE=1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
