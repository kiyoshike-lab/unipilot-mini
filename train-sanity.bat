@echo off
cd /d "%~dp0"
python -m scripts.generate_dataset
python -m tokenizer.train_tokenizer --input "data/**/*.jsonl" "data/**/*.txt" --vocab-size 384 --output tokenizer/vocab.json
python -m training.train --config configs/sanity.json --dataset data/conversations --epochs 3 --batch-size 8 --learning-rate 0.001 --warmup-steps 10 --max-steps 100 --output-dir checkpoints/sanity-100
