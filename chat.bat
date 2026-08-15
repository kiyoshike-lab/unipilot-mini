@echo off
cd /d "%~dp0"
python chat.py --checkpoint checkpoints/sanity-100/checkpoint-step-100.pt
