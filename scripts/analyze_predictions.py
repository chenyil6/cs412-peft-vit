"""Per-class gains over the linear probe and between-method agreement, from results/predictions.npz.

    python scripts/analyze_predictions.py --out latex/final
Writes figures/fig_perclass.pdf/.png and prints the numbers quoted in the report.
"""
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["linear_probe", "bitfit", "lora", "vpt_deep", "full_ft"]
LABEL = {"linear_probe": "Linear probe", "bitfit": "BitFit", "lora": "LoRA", "vpt_deep": "VPT-deep", "full_ft": "Full FT"}
COLOR = {"linear_probe": "#2a78d6", "bitfit": "#eb6834", "lora": "#1baf7a", "vpt_deep": "#eda100", "full_ft": "#e87ba4"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": "#898781",
                     "axes.linewidth": 0.6, "grid.color": "#e1e0d9", "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

ap = argparse.ArgumentParser(); ap.add_argument("--npz", default="results/predictions.npz"); ap.add_argument("--out", default="latex/final"); a = ap.parse_args()
z = np.load(a.npz, allow_pickle=True); y = z["labels"]; classes = z["classes"]; P = {m: z[m] for m in METHODS}
K = len(classes); acc = {m: 100 * (P[m] == y).mean() for m in METHODS}
percls = {m: np.array([100 * (P[m][y == c] == c).mean() for c in range(K)]) for m in METHODS}
print("overall:", {m: round(acc[m], 2) for m in METHODS})

# 1. gain over the linear probe by quintile of linear-probe class accuracy (hardest -> easiest)
order = np.argsort(percls["linear_probe"]); Q = 5; groups = np.array_split(order, Q)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.55, 1]})
ax = axes[0]; ms = ["bitfit", "lora", "vpt_deep", "full_ft"]; wbar = 0.19
for k, m in enumerate(ms):
    g = [np.mean(percls[m][grp] - percls["linear_probe"][grp]) for grp in groups]
    xs = np.arange(Q) + (k - 1.5) * wbar
    ax.bar(xs, g, width=wbar - 0.02, color=COLOR[m], label=LABEL[m])
    for x_, v in zip(xs, g): ax.text(x_, v + 0.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.5, color="#0b0b0b")
labels = [f"{percls['linear_probe'][grp].min():.0f}\u2013{percls['linear_probe'][grp].max():.0f}%" for grp in groups]
ax.set_xticks(range(Q)); ax.set_xticklabels(labels); ax.set_xlabel("Classes grouped by linear-probe accuracy (20 classes each, hardest to easiest)")
ax.set_ylabel("Mean gain over linear probe (points)"); ax.set_ylim(0, 42); ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(frameon=False, ncol=2, loc="upper right")
# 2. agreement matrix
ax = axes[1]; M = np.array([[100 * (P[a_] == P[b_]).mean() for b_ in METHODS] for a_ in METHODS])
im = ax.imshow(M, cmap="Blues", vmin=50, vmax=100)
for i in range(5):
    for j in range(5): ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 80 else "#0b0b0b")
ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels([LABEL[m] for m in METHODS], rotation=35, ha="right"); ax.set_yticklabels([LABEL[m] for m in METHODS]); ax.set_title("Prediction agreement (% of test images)", fontsize=8)
for sp in ax.spines.values(): sp.set_visible(False)
fig.tight_layout(w_pad=1.5); out = Path(a.out) / "figures"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig_perclass.pdf", bbox_inches="tight"); fig.savefig(out / "fig_perclass.png", dpi=300, bbox_inches="tight")

# 3. numbers for the text
third = K // 3; hard, easy = order[:third], order[-third:]
for m in ["bitfit", "lora", "vpt_deep", "full_ft"]:
    g = percls[m] - percls["linear_probe"]
    print(f"{m:13s} gain: hardest third {g[hard].mean():+.1f}  easiest third {g[easy].mean():+.1f}  classes worse than LP: {(g < 0).sum()}")
peft = ["bitfit", "lora", "vpt_deep"]
all_correct = np.all([P[m] == y for m in METHODS], axis=0).mean() * 100; any_correct = np.any([P[m] == y for m in METHODS], axis=0).mean() * 100
print(f"all five correct on {all_correct:.1f}% of test images; at least one correct on {any_correct:.1f}% (oracle)")
print(f"LoRA vs full FT agree on {M[METHODS.index('lora'), METHODS.index('full_ft')]:.1f}%; both correct {100*np.mean((P['lora']==y)&(P['full_ft']==y)):.1f}%; only full FT {100*np.mean((P['lora']!=y)&(P['full_ft']==y)):.1f}%; only LoRA {100*np.mean((P['lora']==y)&(P['full_ft']!=y)):.1f}%")
maj = np.array([np.bincount([P[m][i] for m in peft], minlength=K).argmax() for i in range(len(y))]); print(f"majority vote of the three PEFT methods: {100*(maj==y).mean():.2f}")
