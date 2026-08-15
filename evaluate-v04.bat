@echo off
python -m evaluation.evaluate_v04 --checkpoint checkpoints\v04-eos15\checkpoint-step-2000.pt --output evaluation\results-v04-best-2000.json --prompt-count 300 --max-new-tokens 96 --repetition-penalty 1.1
