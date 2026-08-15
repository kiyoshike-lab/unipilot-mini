@echo off
python -m training.train_v03 --resume-run checkpoints\v03-scratch-001\stage-b\checkpoint-step-2000.pt --max-steps 5000 --output-dir checkpoints\v03-scratch-001
