@echo off
cd /d "%~dp0"
python -m training.train_v02 --max-steps 500 --resume checkpoints/unipilot-v02-step-100/checkpoint-step-100.pt --output-dir checkpoints/unipilot-v02-step-500
