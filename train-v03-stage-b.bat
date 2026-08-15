@echo off
python -m training.train_v03 --resume-run checkpoints\v03-scratch-001\stage-a\checkpoint-step-1000.pt --max-steps 2000 --output-dir checkpoints\v03-scratch-001
