"""M5 analysis: figures and LaTeX tables from runs/ (100-epoch grid) and runs/ablation_epochs/ep50/.

    python scripts/analyze.py --out latex/final

Writes <out>/figures/*.pdf (+ .png previews) and <out>/tables/*.tex.
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["linear_probe", "bitfit", "lora", "vpt_deep", "full_ft"]
LABEL = {"linear_probe": "Linear probe", "bitfit": "BitFit", "lora": "LoRA (r=8)", "vpt_deep": "VPT-deep (p=50)", "full_ft": "Full fine-tuning"}
# fixed categorical order (validated reference palette): blue, orange, aqua, yellow, magenta
COLOR = {"linear_probe": "#2a78d6", "bitfit": "#eb6834", "lora": "#1baf7a", "vpt_deep": "#eda100", "full_ft": "#e87ba4"}
MARKER = {"linear_probe": "o", "bitfit": "s", "lora": "D", "vpt_deep": "^", "full_ft": "v"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "ytick.labelcolor": INK, "xtick.labelcolor": INK, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def load_grid(root):
    rows, curves = [], {}
    for m in METHODS:
        for d in sorted(Path(root).glob(f"{m}_s*")):
            if not (d / "summary.json").exists():
                continue
            s = json.load(open(d / "summary.json")); s["method"] = m; rows.append(s)
            curves.setdefault(m, []).append(pd.read_csv(d / "metrics.csv"))
    return pd.DataFrame(rows), curves


def agg(df):
    g = df.groupby("method")
    out = pd.DataFrame({"acc_mean": g.final_test_acc.mean(), "acc_std": g.final_test_acc.std(),
                        "method_params": g.params_method_excl_head.first(), "trainable": g.params_trainable.first(),
                        "pct": g.trainable_pct.first(), "mem_gb": g.peak_gpu_mem_mb.mean() / 1024,
                        "epoch_s": g.epoch_time_s_mean.mean(), "train_min": g.train_time_s.mean() / 60,
                        "seeds": g.seed.count()})
    return out.reindex(METHODS)


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight"); fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_acc_vs_params(a, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    full = a.loc["full_ft"]
    ax.axhline(full.acc_mean, color=COLOR["full_ft"], lw=1, ls="--", zorder=1)
    ax.axhspan(full.acc_mean - full.acc_std, full.acc_mean + full.acc_std, color=COLOR["full_ft"], alpha=0.12, lw=0)
    for m in METHODS:
        r = a.loc[m]; x = max(r.trainable, 1)
        ax.errorbar(x, r.acc_mean, yerr=r.acc_std, fmt=MARKER[m], color=COLOR[m], ms=6, mec="white", mew=0.8,
                    capsize=2, elinewidth=0.8, zorder=3)
        place = {"linear_probe": (8, -3, "left", "top"), "bitfit": (-8, -6, "right", "top"), "lora": (0, 8, "center", "bottom"),
                 "vpt_deep": (8, -6, "left", "top"), "full_ft": (-8, 6, "right", "bottom")}[m]
        ax.annotate(f"{LABEL[m]}, {r.acc_mean:.1f}", (x, r.acc_mean), textcoords="offset points", xytext=place[:2], ha=place[2], va=place[3], fontsize=6.5, color=INK)
    ax.set_xscale("log"); ax.set_xlabel("Trainable parameters (incl. 76.9k head)"); ax.set_ylabel("Test top-1 accuracy (%)")
    ax.set_ylim(48, 80); ax.grid(axis="y"); ax.set_axisbelow(True)
    save(fig, out, "fig_acc_vs_params")


def fig_curves(curves, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for m in METHODS:
        cs = [c[c.epoch % 5 == 0] for c in curves[m]]; ep = cs[0].epoch.values     # test acc is evaluated every 5 epochs
        acc = np.stack([c.test_acc.values for c in cs]); mu, sd = acc.mean(0), acc.std(0)
        ax.plot(ep, mu, color=COLOR[m], lw=1.4, marker=MARKER[m], ms=2.5, label=LABEL[m]); ax.fill_between(ep, mu - sd, mu + sd, color=COLOR[m], alpha=0.18, lw=0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test top-1 accuracy (%)"); ax.set_xlim(0, 100); ax.set_ylim(0, 80)
    ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(frameon=False, loc="lower right", ncol=1)
    save(fig, out, "fig_curves")


def fig_cost(a, out):
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.1))
    full = a.loc["full_ft"]
    for ax, col, title, ref in [(axes[0], "mem_gb", "Peak GPU memory (GB)", full.mem_gb), (axes[1], "epoch_s", "Time per epoch (s)", full.epoch_s)]:
        y = np.arange(len(METHODS))[::-1]
        vals = [a.loc[m, col] for m in METHODS]
        ax.barh(y, vals, color=[COLOR[m] for m in METHODS], height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.1f}  ({100*v/ref:.0f}%)", va="center", fontsize=6.3, color=INK)
        ax.set_yticks(y); ax.set_yticklabels([LABEL[m].split(" (")[0] for m in METHODS]); ax.set_title(title, fontsize=7.5)
        ax.set_xlim(0, max(vals) * 1.55); ax.grid(axis="x"); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    axes[1].set_yticklabels([]); fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig_cost")


def fig_budget(a50, a100, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    y = np.arange(len(METHODS))[::-1]
    for yi, m in zip(y, METHODS):
        x0, x1 = a50.loc[m].acc_mean, a100.loc[m].acc_mean
        ax.plot([x0, x1], [yi, yi], color=COLOR[m], lw=2, solid_capstyle="round", zorder=2)
        ax.plot(x0, yi, "o", color="white", mec=COLOR[m], mew=1.4, ms=5.5, zorder=3)
        ax.plot(x1, yi, "o", color=COLOR[m], ms=5.5, zorder=3)
        ax.text(x1 + 0.6, yi, f"+{x1-x0:.1f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[m].split(" (")[0] for m in METHODS]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Test top-1 accuracy (%), mean of 3 seeds"); ax.set_xlim(50, 78); ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.plot([], [], "o", color="white", mec=MUTED, mew=1.4, label="50 epochs"); ax.plot([], [], "o", color=MUTED, label="100 epochs")
    ax.legend(frameon=False, loc="lower left")
    save(fig, out, "fig_budget")


def tables(df, a100, a50, out):
    def row(m):
        r = a100.loc[m]; r5 = a50.loc[m]
        bold = r.acc_mean == a100.acc_mean.max()
        acc = f"$\\mathbf{{{r.acc_mean:.2f}\\pm{r.acc_std:.2f}}}$" if bold else f"${r.acc_mean:.2f}\\pm{r.acc_std:.2f}$"
        return f"{LABEL[m]} & {acc} & {int(r.method_params):,} & {100*r.method_params/85798656:.2f} & {r.mem_gb:.1f} & {r.epoch_s:.1f} & ${r5.acc_mean:.2f}\\pm{r5.acc_std:.2f}$ \\\\"
    (out / "main_results.tex").write_text("\n".join(row(m) for m in METHODS))   # no trailing newline: the file is \input inside a tabular
    per = df.pivot(index="method", columns="seed", values="final_test_acc").reindex(METHODS)
    (out / "per_seed.tex").write_text("\n".join(f"{LABEL[m]} & " + " & ".join(f"{per.loc[m, s]:.2f}" for s in sorted(per.columns)) + f" & {per.loc[m].mean():.2f} & {per.loc[m].std():.2f} \\\\" for m in METHODS))
    rel = pd.DataFrame({"acc_rel": 100 * a100.acc_mean / a100.loc["full_ft"].acc_mean, "mem_rel": 100 * a100.mem_gb / a100.loc["full_ft"].mem_gb, "time_rel": 100 * a100.epoch_s / a100.loc["full_ft"].epoch_s})
    (out / "relative_to_full.tex").write_text("\n".join(f"{LABEL[m]} & {rel.loc[m].acc_rel:.1f} & {rel.loc[m].mem_rel:.0f} & {rel.loc[m].time_rel:.0f} \\\\" for m in METHODS))
    return rel


def main():
    p = argparse.ArgumentParser(); p.add_argument("--runs", default="runs"); p.add_argument("--ep50", default="runs/ablation_epochs/ep50"); p.add_argument("--out", default="latex/final")
    args = p.parse_args()
    figs, tabs = Path(args.out) / "figures", Path(args.out) / "tables"; figs.mkdir(parents=True, exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)
    df100, curves = load_grid(args.runs); df50, _ = load_grid(args.ep50)
    a100, a50 = agg(df100), agg(df50)
    fig_acc_vs_params(a100, figs); fig_curves(curves, figs); fig_cost(a100, figs); fig_budget(a50, a100, figs)
    rel = tables(df100, a100, a50, tabs)
    pd.set_option("display.width", 200); print(a100.round(2)); print(); print(rel.round(1))
    # convergence: last-20-epoch gain per method (mean over seeds)
    print("\nmean test-acc gain over the last 20 epochs (epoch 80 -> 100):")
    for m in METHODS:
        g = np.mean([c.test_acc.iloc[-1] - c.test_acc.iloc[79] for c in curves[m]]); tr = np.mean([c.train_acc.iloc[-1] for c in curves[m]])
        print(f"  {m:13s} {g:+.2f}   final train acc {tr:.1f}")


if __name__ == "__main__":
    main()
