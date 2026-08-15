@echo off
python -m training.train_v03 --config configs\unipilot-v03-long.json --resume-run checkpoints\v03-scratch-001\stage-c\checkpoint-step-5000.pt --max-steps 10000 --output-dir checkpoints\v03-long-001
