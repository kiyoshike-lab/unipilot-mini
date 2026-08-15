@echo off
cd /d "%~dp0"
python -m training.train_v02 --max-steps 100 --output-dir checkpoints/unipilot-v02-step-100
