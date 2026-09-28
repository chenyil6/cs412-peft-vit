import csv
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def trainable_param_names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def cosine_warmup_lambda(total_steps, warmup_steps, min_lr_ratio=0.0):
    """Per-step multiplier: linear warmup from 0, then cosine decay to min_lr_ratio."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return f


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).resolve().parent.parent).decode().strip()
    except Exception:
        return None


class RunLogger:
    """Writes per-epoch rows to metrics.csv and the final summary to summary.json."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "metrics.csv"
        self._csv_file = None
        self._writer = None

    def log_epoch(self, row: dict):
        if self._writer is None:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._csv_file.flush()

    def write_json(self, name, obj):
        with open(self.out_dir / name, "w") as f:
            json.dump(obj, f, indent=2, default=str)

    def close(self):
        if self._csv_file:
            self._csv_file.close()


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.t0
