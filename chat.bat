@echo off
cd /d "%~dp0"
python chat.py --checkpoint checkpoints/unipilot-v02-step-1000/checkpoint-step-1000.pt --tokenizer tokenizer/vocab-v02-512.json
