"""Collect runs/*/summary.json into one table (mean ± std over seeds per method).

    python scripts/aggregate.py                # prints markdown table, writes results/summary_by_method.csv
    python scripts/aggregate.py --runs runs    # custom runs dir
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    rows = []
    for f in sorted(Path(a.runs).glob("*/summary.json")):
        if "smoke" in f.parent.name:
            continue
        s = json.load(open(f))
        s["run"] = f.parent.name
        rows.append(s)
    if not rows:
        print("no summaries found"); return
    df = pd.DataFrame(rows)
    cols = ["method", "seed", "final_test_acc", "params_trainable", "params_method_excl_head", "trainable_pct",
            "peak_gpu_mem_mb", "epoch_time_s_mean", "train_time_s", "train_throughput_img_s", "epochs"]
    df = df[cols + ["run"]]

    Path(a.out).mkdir(exist_ok=True)
    df.to_csv(Path(a.out) / "summary_per_run.csv", index=False)
    g = df.groupby("method")
    agg = pd.DataFrame({
        "seeds": g["seed"].count(),
        "acc_mean": g["final_test_acc"].mean(), "acc_std": g["final_test_acc"].std(),
        "trainable_params": g["params_trainable"].first(),
        "method_params": g["params_method_excl_head"].first(),
        "trainable_pct": g["trainable_pct"].first(),
        "peak_mem_mb": g["peak_gpu_mem_mb"].mean(),
        "epoch_time_s": g["epoch_time_s_mean"].mean(),
        "train_time_min": g["train_time_s"].mean() / 60,
    })
    order = ["linear_probe", "bitfit", "lora", "vpt_deep", "full_ft"]
    agg = agg.reindex([m for m in order if m in agg.index] + [m for m in agg.index if m not in order])
    agg.to_csv(Path(a.out) / "summary_by_method.csv")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print()
    print(agg.round(2).to_markdown())


if __name__ == "__main__":
    main()
