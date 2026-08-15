@echo off
cd /d "%~dp0"
python -m tokenizer.train_tokenizer --input "data/**/*.jsonl" "data/**/*.txt" --vocab-size 512 --output tokenizer/vocab.json
python -m training.train --config configs/v0.1.json --dataset data/conversations --tokenizer tokenizer/vocab.json --epochs 10 --batch-size 4 --learning-rate 0.0003 --mixed-precision --output-dir checkpoints/v0.1
