@echo off
python scripts\prepare_dataset_v03.py
python scripts\check_dataset_v03.py
python scripts\verify_v03_readiness.py
