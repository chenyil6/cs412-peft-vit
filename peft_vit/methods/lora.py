"""LoRA (Hu et al., ICLR 2022) on the attention projections, implemented from scratch.

timm's ViT computes q, k, v with ONE fused linear layer  qkv: dim -> 3*dim  whose output is split as
[q | k | v] along the last axis. We wrap that layer: the frozen fused projection runs unchanged, and for
every selected target t in {q, k, v} a low-rank path

    delta_t(x) = (alpha / r) * B_t A_t dropout(x),      A_t: r x dim,  B_t: dim x r

is added to the corresponding dim-wide slice of the output. B_t is zero-initialised, so at step 0 the
model is exactly the pre-trained one; A_t uses the Kaiming-uniform init of the official loralib
(the paper says "random Gaussian"; the released code uses kaiming_uniform(a=sqrt(5))).
`merge()` folds (alpha / r) B_t A_t into the fused weight, so the adapted model has zero extra
inference cost; `unmerge()` restores it. Trainable parameters per layer and target: 2 * r * dim.
"""
import math

import torch
import torch.nn as nn


class LoRAQKV(nn.Module):
    SLOTS = ("q", "k", "v")

    def __init__(self, qkv: nn.Linear, dim: int, r: int, alpha: float, targets=("q", "v"), dropout: float = 0.0):
        super().__init__()
        assert qkv.in_features == dim and qkv.out_features == 3 * dim, "expected timm's fused qkv Linear"
        for t in targets:
            assert t in self.SLOTS, f"unknown LoRA target {t!r}"
        self.qkv = qkv                     # frozen pre-trained projection (weights untouched)
        self.dim, self.r = dim, r
        self.scaling = alpha / r
        self.targets = list(targets)
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for t in self.targets:
            self.lora_A[t] = nn.Parameter(torch.empty(r, dim))
            self.lora_B[t] = nn.Parameter(torch.zeros(dim, r))
            nn.init.kaiming_uniform_(self.lora_A[t], a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def delta_weight(self, t):
        """(alpha / r) * B_t A_t, shape dim x dim, in the dtype of the frozen weight."""
        return (self.lora_B[t] @ self.lora_A[t]) * self.scaling

    def forward(self, x):
        out = self.qkv(x)
        if self.merged:
            return out
        xd = self.dropout(x)
        parts = []
        for slot in self.SLOTS:
            if slot in self.targets:
                parts.append(((xd @ self.lora_A[slot].t()) @ self.lora_B[slot].t()) * self.scaling)
            else:
                parts.append(torch.zeros_like(out[..., : self.dim]))
        return out + torch.cat(parts, dim=-1)

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        for t in self.targets:
            off = self.SLOTS.index(t) * self.dim
            self.qkv.weight[off: off + self.dim] += self.delta_weight(t).to(self.qkv.weight.dtype)
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        for t in self.targets:
            off = self.SLOTS.index(t) * self.dim
            self.qkv.weight[off: off + self.dim] -= self.delta_weight(t).to(self.qkv.weight.dtype)
        self.merged = False

    def extra_repr(self):
        return f"dim={self.dim}, r={self.r}, scaling={self.scaling:g}, targets={self.targets}"


def apply_lora(model, cfg):
    lcfg = cfg["lora"]
    r, alpha = int(lcfg["r"]), float(lcfg["alpha"])
    targets, dropout = list(lcfg.get("targets", ["q", "v"])), float(lcfg.get("dropout", 0.0))
    for p in model.backbone.parameters():
        p.requires_grad = False
    dim = model.backbone.embed_dim
    n_lora = 0
    for blk in model.backbone.blocks:
        assert isinstance(blk.attn.qkv, nn.Linear), "LoRA already applied?"
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, dim, r, alpha, targets, dropout)
        n_lora += sum(p.numel() for p in blk.attn.qkv.parameters() if p.requires_grad)
    for p in model.head.parameters():
        p.requires_grad = True
    model.backbone_no_grad = False
    return {"trainable": f"LoRA r={r} alpha={alpha:g} on {targets} in {len(model.backbone.blocks)} blocks + head",
            "lora_params": n_lora}


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRAQKV)]


def merge_lora(model):
    for m in lora_modules(model):
        m.merge()


def unmerge_lora(model):
    for m in lora_modules(model):
        m.unmerge()
