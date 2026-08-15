@echo off
python chat.py --checkpoint checkpoints\v04-eos15\checkpoint-step-2000.pt --tokenizer tokenizer\vocab-v02-512.json --max-new-tokens 96 --temperature 0.7 --top-k 40 --top-p 0.9 --repetition-penalty 1.1
