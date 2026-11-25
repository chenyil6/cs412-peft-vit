"""Per-image test predictions of each trained model (seed 0), for the per-class / agreement analysis.

    python scripts/eval_predictions.py --ckpt linear_probe=runs/linear_probe_s0 bitfit=runs/bitfit_s0 ...
Writes results/predictions.npz with labels, and one int array of predicted classes per method.
"""
import argparse, json
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from peft_vit.data import FGVCAircraft, build_transforms
from peft_vit.methods import apply_method
from peft_vit.model import PEFTClassifier, load_backbone, load_trainable

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", nargs="+", required=True); ap.add_argument("--out", default="results/predictions.npz"); a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); out = {}
for spec in a.ckpt:
    method, d = spec.split("="); cfg = json.load(open(f"{d}/config.json"))["config"]
    bb, norm = load_backbone(cfg["backbone"], weights=cfg.get("backbone_weights")); model = PEFTClassifier(bb, 100); apply_method(model, method, cfg)
    load_trainable(model, f"{d}/trainable.pt"); model.to(dev).eval()
    _, eval_tf = build_transforms(norm["mean"], norm["std"], cfg["data"]["resize"], cfg["data"]["crop"])
    ds = FGVCAircraft(cfg["data"]["root"], cfg["data"]["test_split"], eval_tf); dl = DataLoader(ds, batch_size=256, num_workers=8)
    preds, labels = [], []
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for x, y in dl: preds.append(model(x.to(dev)).argmax(1).cpu()); labels.append(y)
    out[method] = torch.cat(preds).numpy(); out["labels"] = torch.cat(labels).numpy(); out["classes"] = np.array(ds.classes)
    print(f"{method:13s} test acc {100*(out[method]==out['labels']).mean():.2f}", flush=True)
np.savez(a.out, **out); print("wrote", a.out)
