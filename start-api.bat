@echo off
cd /d "%~dp0"
set UNIPILOT_CHECKPOINT=checkpoints/sanity-100/checkpoint-step-100.pt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
