"""Render runs/lp_lr_sweep/*/summary.json + metrics.csv as a markdown table (val split)."""
import json, sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/lp_lr_sweep")
rows = []
for d in sorted(root.iterdir()):
    if not (d / "summary.json").exists():
        continue
    s = json.load(open(d / "summary.json")); m = pd.read_csv(d / "metrics.csv")
    c = json.load(open(d / "config.json"))["config"]
    rows.append({"optimizer": c["optim"]["name"], "lr": c["optim"]["lr"],
                 "val_acc_final": s["final_test_acc"], "val_acc_best": m.test_acc.max(),
                 "train_acc_final": m.train_acc.iloc[-1], "train_loss_final": m.train_loss.iloc[-1],
                 "train_split": c["data"]["train_split"], "eval_split": c["data"]["test_split"]})
df = pd.DataFrame(rows).sort_values(["optimizer", "lr"], ascending=[False, True])
assert (df.eval_split == "val").all(), "sweep must be evaluated on val, not test"
print("| optimizer | lr | val acc (last ep) | val acc (best ep) | train acc (last) | train loss (last) |")
print("|---|---|---|---|---|---|")
for r in df.itertuples():
    print(f"| {r.optimizer} | {r.lr:g} | {r.val_acc_final:.2f} | {r.val_acc_best:.2f} | {r.train_acc_final:.2f} | {r.train_loss_final:.3f} |")
