@echo off
python -m training.train_v04 --experiment-id v04-eos15 --eos-weight 1.5 --max-steps 2000 --output-dir checkpoints\v04-eos15
