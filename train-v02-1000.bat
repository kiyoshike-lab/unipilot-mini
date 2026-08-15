@echo off
cd /d "%~dp0"
python -m training.train_v02 --max-steps 1000 --resume checkpoints/unipilot-v02-step-500/checkpoint-step-500.pt --output-dir checkpoints/unipilot-v02-step-1000
