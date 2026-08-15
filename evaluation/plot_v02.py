import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(); parser.add_argument("logs", nargs="+"); parser.add_argument("--output", default="evaluation/v02-loss.png")
args = parser.parse_args(); rows = []
for log in args.logs:
    with open(log, encoding="utf-8") as file: rows.extend(csv.DictReader(file))
rows.sort(key=lambda row: int(row["step"])); trained = [row for row in rows if row["train_loss"]]
figure, loss_axis = plt.subplots(figsize=(8, 4.8)); learning_axis = loss_axis.twinx()
loss_axis.plot([int(row["step"]) for row in trained], [float(row["train_loss"]) for row in trained], "o-", label="train loss")
loss_axis.plot([int(row["step"]) for row in rows], [float(row["validation_loss"]) for row in rows], "o-", label="validation loss")
learning_axis.plot([int(row["step"]) for row in rows], [float(row["learning_rate"]) for row in rows], "--", color="gray", label="learning rate")
loss_axis.set(xlabel="step", ylabel="loss"); learning_axis.set_ylabel("learning rate"); loss_axis.grid(alpha=.25)
lines = loss_axis.lines + learning_axis.lines; loss_axis.legend(lines, [line.get_label() for line in lines]); figure.tight_layout()
Path(args.output).parent.mkdir(parents=True, exist_ok=True); figure.savefig(args.output, dpi=160)
