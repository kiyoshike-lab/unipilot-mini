@echo off
python -m evaluation.evaluate_v03 --checkpoint checkpoints\v03-scratch-001\stage-c\checkpoint-step-5000.pt --output evaluation\results-v03-5000.json --prompt-count 300 --max-new-tokens 128 --temperature 0.7 --top-k 40 --top-p 0.9 --repetition-penalty 1.0
