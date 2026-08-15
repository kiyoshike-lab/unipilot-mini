@echo off
python -m scripts.prepare_dataset_v04
python -m pytest tests\test_v04.py -q
