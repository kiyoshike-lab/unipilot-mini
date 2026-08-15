@echo off
python chat.py --checkpoint checkpoints\v03-scratch-001\stage-c\checkpoint-step-5000.pt --tokenizer tokenizer\vocab-v02-512.json --max-new-tokens 128 --temperature 0.7 --top-k 40 --top-p 0.9 --repetition-penalty 1.0
