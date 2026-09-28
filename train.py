"""Single entry point for every method.

    python train.py --config configs/linear_probe.yaml --seed 0
    python train.py --config configs/linear_probe.yaml --smoke          # tiny CPU/GPU sanity run
    python train.py --config configs/lora.yaml --set sched.epochs=30 optim.lr=1e-3

Outputs (in <out_dir>/<method>_s<seed>/):
    config.json            resolved config actually used
    trainable_params.txt   every trainable tensor and its shape (for parameter-count verification)
    metrics.csv            one row per epoch
    summary.json           final accuracy, parameter counts, peak memory, wall-clock
    trainable.pt           state dict of trainable tensors only (optional)
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from peft_vit.data import FGVCAircraft, build_transforms
from peft_vit.methods import apply_method
from peft_vit.model import PEFTClassifier, load_backbone
from peft_vit.utils import (RunLogger, Timer, cosine_warmup_lambda, count_params, git_commit,
                            set_seed, trainable_param_names)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None, help="overrides train.seed")
    p.add_argument("--out_dir", default=None, help="overrides <cfg.out_dir>/<method>_s<seed>")
    p.add_argument("--smoke", action="store_true", help="64 train / 64 test images, 1 epoch; runs on CPU too")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="dotted config overrides")
    return p.parse_args()


def apply_overrides(cfg, kvs):
    for kv in kvs:
        key, val = kv.split("=", 1)
        d = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            d = d.setdefault(k, {})
        d[parts[-1]] = yaml.safe_load(val)
    return cfg


def build_optimizer(params, ocfg):
    name = ocfg["name"].lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=ocfg["lr"], momentum=ocfg.get("momentum", 0.9),
                               weight_decay=ocfg.get("weight_decay", 0.0), nesterov=ocfg.get("nesterov", False))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=ocfg["lr"], betas=tuple(ocfg.get("betas", (0.9, 0.999))),
                                 weight_decay=ocfg.get("weight_decay", 0.0))
    raise ValueError(f"unknown optimizer {name}")


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
        loss_sum += crit(logits.float(), y).item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return loss_sum / n, 100.0 * correct / n


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg = apply_overrides(cfg, args.set)
    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    cfg["train"]["seed"] = seed
    if args.smoke:
        cfg["sched"]["epochs"], cfg["sched"]["warmup_epochs"] = 1, 0
        cfg["data"]["batch_size"], cfg["data"]["workers"] = 16, 2
        cfg["train"]["eval_every"], cfg["train"]["save_ckpt"] = 1, False

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    run_name = f"{cfg['method']}_s{seed}" + ("_smoke" if args.smoke else "")
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.get("out_dir", "runs")) / run_name
    logger = RunLogger(out_dir)
    print(f"[run] {run_name} -> {out_dir} | device={device} amp={use_amp}", flush=True)

    # ---- model -------------------------------------------------------------------------------
    backbone, norm = load_backbone(cfg["backbone"], pretrained=cfg.get("pretrained", True),
                                   weights=cfg.get("backbone_weights"))
    dcfg = cfg["data"]
    train_tf, eval_tf = build_transforms(norm["mean"], norm["std"], dcfg["resize"], dcfg["crop"])
    train_ds = FGVCAircraft(dcfg["root"], dcfg["train_split"], train_tf)
    test_ds = FGVCAircraft(dcfg["root"], dcfg["test_split"], eval_tf)
    num_classes = len(train_ds.classes)
    if args.smoke:
        # annotation files are sorted by class, so take a random subset rather than the first 64
        g = torch.Generator().manual_seed(0)
        train_ds = Subset(train_ds, torch.randperm(len(train_ds), generator=g)[:64].tolist())
        test_ds = Subset(test_ds, torch.randperm(len(test_ds), generator=g)[:64].tolist())

    model = PEFTClassifier(backbone, num_classes)
    n_base = count_params(model)                     # pre-trained backbone + shared head, before any method is applied
    method_info = apply_method(model, cfg["method"], cfg)
    model.to(device)

    n_total = count_params(model)                    # after the method (LoRA / VPT add parameters)
    n_added = n_total - n_base
    n_trainable = count_params(model, trainable_only=True)
    n_head = count_params(model.head)
    n_method = n_trainable - n_head
    with open(out_dir / "trainable_params.txt", "w") as f:
        for n, p in model.named_parameters():
            if p.requires_grad:
                f.write(f"{n}\t{tuple(p.shape)}\t{p.numel()}\n")
        f.write(f"\n# trainable={n_trainable}  head={n_head}  method(excl. head)={n_method}  "
                f"added={n_added}  base(backbone+head)={n_base}\n")
    print(f"[params] base={n_base:,} added={n_added:,} trainable={n_trainable:,} (head {n_head:,} + method {n_method:,}) "
          f"= {100*n_trainable/n_base:.3f}% of backbone+head | {method_info}", flush=True)

    # ---- data --------------------------------------------------------------------------------
    train_loader = DataLoader(train_ds, batch_size=dcfg["batch_size"], shuffle=True, num_workers=dcfg["workers"],
                              pin_memory=True, drop_last=False, persistent_workers=dcfg["workers"] > 0)
    test_loader = DataLoader(test_ds, batch_size=dcfg.get("eval_batch_size", 256), shuffle=False,
                             num_workers=dcfg["workers"], pin_memory=True, persistent_workers=dcfg["workers"] > 0)

    # ---- optimisation ------------------------------------------------------------------------
    scfg = cfg["sched"]
    epochs = scfg["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps, warmup_steps = epochs * steps_per_epoch, scfg.get("warmup_epochs", 0) * steps_per_epoch
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(trainable, cfg["optim"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_warmup_lambda(total_steps, warmup_steps, scfg.get("min_lr_ratio", 0.0)))
    criterion = nn.CrossEntropyLoss()

    logger.write_json("config.json", {"config": cfg, "argv": sys.argv, "git_commit": git_commit(),
                                      "torch": torch.__version__, "python": platform.python_version(),
                                      "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                                      "trainable_param_names": trainable_param_names(model)})

    # ---- train -------------------------------------------------------------------------------
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    eval_every = cfg["train"].get("eval_every", 5)
    wall = Timer()
    train_time_total = 0.0
    step = 0
    test_loss, test_acc = float("nan"), float("nan")
    for epoch in range(epochs):
        model.train()
        et = Timer()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["train"].get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(trainable, cfg["train"]["grad_clip"])
            optimizer.step()
            scheduler.step()
            step += 1
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.numel()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = et.elapsed()
        train_time_total += epoch_time

        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            test_loss, test_acc = evaluate(model, test_loader, device, use_amp)

        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"], "train_loss": loss_sum / seen,
               "train_acc": 100.0 * correct / seen, "test_loss": test_loss, "test_acc": test_acc,
               "epoch_time_s": epoch_time, "elapsed_s": wall.elapsed()}
        logger.log_epoch(row)
        print(f"[ep {epoch+1:3d}/{epochs}] lr={row['lr']:.2e} train_loss={row['train_loss']:.4f} "
              f"train_acc={row['train_acc']:.2f} test_acc={test_acc:.2f} ({epoch_time:.1f}s)", flush=True)

    # ---- summary -----------------------------------------------------------------------------
    peak_mem_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    n_train = len(train_ds)
    summary = {
        "method": cfg["method"], "seed": seed, "backbone": cfg["backbone"], "epochs": epochs,
        "final_test_acc": test_acc, "final_test_loss": test_loss,
        "params_base": n_base, "params_added": n_added, "params_total": n_total,
        "params_trainable": n_trainable, "params_head": n_head,
        "params_method_excl_head": n_method, "trainable_pct": 100 * n_trainable / n_base,
        "peak_gpu_mem_mb": peak_mem_mb,
        "train_time_s": train_time_total, "wall_time_s": wall.elapsed(),
        "epoch_time_s_mean": train_time_total / epochs,
        "train_throughput_img_s": n_train * epochs / train_time_total,
        "batch_size": dcfg["batch_size"], "amp": use_amp,
    }
    logger.write_json("summary.json", summary)
    if cfg["train"].get("save_ckpt", True):
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                   out_dir / "trainable.pt")
    logger.close()
    print("[done] " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}),
          flush=True)


if __name__ == "__main__":
    main()
