"""Ablation tables/figure for the final report. Tolerates missing runs (prints what exists).

    python scripts/analyze_ablations.py --out latex/final
"""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

COLOR = {"vpt100": "#eda100", "vpt200": "#6250d6"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": "#898781", "axes.linewidth": 0.6, "grid.color": "#e1e0d9", "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def load(pattern):
    out = []
    for d in sorted(Path(".").glob(pattern)):
        if (d / "summary.json").exists():
            s = json.load(open(d / "summary.json")); s["dir"] = str(d); s["curve"] = pd.read_csv(d / "metrics.csv"); out.append(s)
    return out


def fmt(runs):
    if not runs: return "--", "--", "--", "--"
    a = np.array([r["final_test_acc"] for r in runs])
    acc = f"${a.mean():.2f}\\pm{a.std(ddof=1):.2f}$" if len(a) > 1 else f"${a[0]:.2f}$"
    return acc, f"{int(runs[0].get('params_method_effective', runs[0]['params_method_excl_head'])):,}", f"{np.mean([r['peak_gpu_mem_mb'] for r in runs])/1024:.1f}", f"{np.mean([r['epoch_time_s_mean'] for r in runs]):.1f}"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", default="latex/final"); a = p.parse_args()
    tabs = Path(a.out) / "tables"; figs = Path(a.out) / "figures"; tabs.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    rows = [
        ("VPT $p{=}50$, 100 ep (main)", load("runs/vpt_deep_s*")),
        ("VPT $p{=}50$, 200 ep", load("runs/ablation_epochs/ep200/vpt_deep_s*")),
        ("BitFit, all biases (main)", load("runs/bitfit_s*")),
        ("BitFit, $b_q,b_2$ only", load("runs/ablation_bitfit_subset/bitfit_qfc2_s*")),
        ("LoRA $r{=}2$ (seed 0)", load("runs/ablation_lora_rank/lora_r2_s0")),
        ("LoRA $r{=}8$ (main, seed 0)", [r for r in load("runs/lora_s0")]),
        ("LoRA $r{=}32$ (seed 0)", load("runs/ablation_lora_rank/lora_r32_s0")),
        ("VPT $p{=}10$ (seed 0)", load("runs/ablation_vpt_tokens/vpt_p10_s0")),
        ("VPT $p{=}50$ (main, seed 0)", load("runs/vpt_deep_s0")),
        ("VPT $p{=}200$ (seed 0)", load("runs/ablation_vpt_tokens/vpt_p200_s0")),
    ]
    lines = []
    for name, runs in rows:
        acc, params, mem, t = fmt(runs); n = len(runs)
        lines.append(f"{name} & {acc} & {params} & {mem} & {t} \\\\"); print(f"{name:40s} n={n}  acc={acc:22s} params={params:>12s} mem={mem} GB  epoch={t} s")
    (tabs / "ablations.tex").write_text("\n".join(lines))
    # figure: VPT 100 vs 200 epochs curves (mean over seeds, evaluated epochs only)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for key, pat, lab in [("vpt100", "runs/vpt_deep_s*", "100-epoch schedule"), ("vpt200", "runs/ablation_epochs/ep200/vpt_deep_s*", "200-epoch schedule")]:
        runs = load(pat)
        if not runs: continue
        cs = [r["curve"][r["curve"].epoch % 5 == 0] for r in runs]; n = min(len(c) for c in cs); cs = [c.iloc[:n] for c in cs]
        ep = cs[0].epoch.values; acc = np.stack([c.test_acc.values for c in cs]); mu, sd = acc.mean(0), acc.std(0)
        ax.plot(ep, mu, color=COLOR[key], lw=1.4, label=f"{lab} (n={len(runs)})"); ax.fill_between(ep, mu - sd, mu + sd, color=COLOR[key], alpha=0.18, lw=0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test top-1 accuracy (%)"); ax.set_ylim(0, 80); ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(frameon=False, loc="lower right")
    fig.savefig(figs / "fig_vpt_budget.pdf", bbox_inches="tight"); fig.savefig(figs / "fig_vpt_budget.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
