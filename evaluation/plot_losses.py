import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--history", default="checkpoints/training_history.csv")
parser.add_argument("--output", default="evaluation/loss_curve.png")
args = parser.parse_args()
with open(args.history, encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
steps = [int(row["step"]) for row in rows]
train_rows = [row for row in rows if row["train_loss"]]
plt.plot([int(row["step"]) for row in train_rows], [float(row["train_loss"]) for row in train_rows], "o-", label="train")
plt.plot(steps, [float(row["validation_loss"]) for row in rows], label="validation")
plt.xlabel("step"); plt.ylabel("cross-entropy loss"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
Path(args.output).parent.mkdir(parents=True, exist_ok=True); plt.savefig(args.output, dpi=160)
