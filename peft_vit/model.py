"""Backbone loading and the shared classifier wrapper.

All methods share exactly the same backbone checkpoint and the same freshly initialised
linear head; they differ only in which parameters `apply_method` leaves trainable (and, for
LoRA / VPT, which modules it injects).
"""
import timm
import torch
import torch.nn as nn

# ViT-B/16 pre-trained on ImageNet-21k (original Google checkpoint, the one used by the VPT paper).
BACKBONES = {
    "vit_b16_in21k": "vit_base_patch16_224.orig_in21k",
    "vit_b16_in21k_augreg": "vit_base_patch16_224.augreg_in21k",
}


def load_backbone(name="vit_b16_in21k", pretrained=True, weights=None):
    """`weights`: optional path to a local safetensors/pth file. When given, timm loads from it
    directly and never touches the Hugging Face cache (keeps weights in /work/nvme, not $HOME)."""
    kwargs = {"pretrained_cfg_overlay": {"file": weights}} if weights else {}
    model = timm.create_model(BACKBONES[name], pretrained=pretrained, num_classes=0, **kwargs)
    cfg = model.pretrained_cfg
    return model, {"mean": cfg["mean"], "std": cfg["std"], "input_size": cfg["input_size"]}


class PEFTClassifier(nn.Module):
    """frozen-or-not ViT backbone + linear head on the [CLS] feature."""

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.num_features, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        # If True, the backbone forward runs under no_grad (linear probe): saves the activation
        # memory and the backward pass through 12 blocks.
        self.backbone_no_grad = False

    def forward(self, x):
        if self.backbone_no_grad:
            with torch.no_grad():
                feats = self.backbone(x)
        else:
            feats = self.backbone(x)
        return self.head(feats)
