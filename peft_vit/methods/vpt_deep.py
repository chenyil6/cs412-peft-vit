"""VPT-deep (Jia et al., ECCV 2022), implemented from scratch on timm's ViT.

p learnable prompt tokens P_i (p x dim) are inserted between the [CLS] token and the patch tokens at the
INPUT of every transformer block i. Following the official implementation, the prompt outputs of block i
are discarded and replaced by a fresh P_{i+1} before block i+1, so each layer has its own prompts:

    block 0 input:   [CLS, P_0, x_1 .. x_196]
    block i input:   [CLS_out, P_i, patches_out]          (prompt outputs of block i-1 dropped)

Prompts carry no positional embedding; the backbone is frozen; the [CLS] output feeds the shared head.
Init: uniform(-v, v) with v = sqrt(6 / (3 * patch_size^2 + dim)) ("xavier uniform" in the paper / code,
v = 0.0625 for ViT-B/16). Dropout 0.1 on the prompts, as the paper uses for VPT-deep.
Trainable parameters: depth * p * dim  (12 * 50 * 768 = 460,800 for p = 50).
"""
import math

import torch
import torch.nn as nn


class VPTDeepBlocks(nn.Module):
    """Drop-in replacement for `VisionTransformer.blocks` (an nn.Sequential) that adds deep prompts."""

    def __init__(self, blocks, num_tokens, dim, patch_size=16, dropout=0.1, init="xavier_uniform"):
        super().__init__()
        self.blocks = blocks                       # the frozen original blocks
        self.num_tokens = num_tokens
        self.prompts = nn.Parameter(torch.zeros(len(blocks), num_tokens, dim))
        if init == "xavier_uniform":
            val = math.sqrt(6.0 / (3 * patch_size * patch_size + dim))
            nn.init.uniform_(self.prompts, -val, val)
        elif init == "normal":
            nn.init.normal_(self.prompts, std=0.02)
        else:
            raise ValueError(f"unknown prompt init {init!r}")
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        for i, blk in enumerate(self.blocks):
            p = self.dropout(self.prompts[i].unsqueeze(0).expand(B, -1, -1)).to(x.dtype)
            skip = self.num_tokens if i > 0 else 0              # drop the previous layer's prompt outputs
            x = torch.cat([x[:, :1], p, x[:, 1 + skip:]], dim=1)
            x = blk(x)
        return x

    def __len__(self):
        return len(self.blocks)

    def extra_repr(self):
        return f"num_tokens={self.num_tokens}, layers={len(self.blocks)}"


def apply_vpt_deep(model, cfg):
    vcfg = cfg["vpt"]
    p, dropout, init = int(vcfg["num_tokens"]), float(vcfg.get("dropout", 0.1)), vcfg.get("init", "xavier_uniform")
    assert vcfg.get("deep", True), "only VPT-deep is implemented"
    bb = model.backbone
    assert bb.num_prefix_tokens == 1 and bb.global_pool == "token", "expects a ViT with a single [CLS] token"
    assert not isinstance(bb.blocks, VPTDeepBlocks), "VPT already applied?"
    for q in bb.parameters():
        q.requires_grad = False
    bb.blocks = VPTDeepBlocks(bb.blocks, p, bb.embed_dim, bb.patch_embed.patch_size[0], dropout, init)
    for q in model.head.parameters():
        q.requires_grad = True
    model.backbone_no_grad = False
    return {"trainable": f"VPT-deep p={p} in {len(bb.blocks)} layers + head", "prompt_params": bb.blocks.prompts.numel()}
