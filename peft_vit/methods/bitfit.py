"""BitFit (Ben Zaken, Goldberg, Ravfogel; ACL 2022): train only the bias terms of the frozen backbone,
plus the shared head. No parameters are added.

On timm's ViT-B/16 the bias terms are
    per block: norm1 (768) + attn.qkv (2304) + attn.proj (768) + norm2 (768) + mlp.fc1 (3072) + mlp.fc2 (768) = 8,448
    x 12 blocks = 101,376, plus patch_embed.proj (768) and the final norm (768)  ->  102,912  (0.12% of the backbone)
`cls_token` and `pos_embed` are not biases and stay frozen.
"""


def apply_bitfit(model, cfg):
    bcfg = cfg.get("bitfit") or {}
    exclude = tuple(bcfg.get("exclude", []))   # optional name prefixes to keep frozen, e.g. ["patch_embed"]
    n_bias = 0
    for name, p in model.backbone.named_parameters():
        train = name.endswith(".bias") and not name.startswith(exclude)
        p.requires_grad = train
        n_bias += p.numel() if train else 0
    for p in model.head.parameters():
        p.requires_grad = True
    model.backbone_no_grad = False       # gradients must flow through the whole network to reach every bias
    return {"trainable": "all backbone bias terms + head", "bias_params": n_bias}
