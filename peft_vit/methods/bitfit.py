"""BitFit (Ben Zaken, Goldberg, Ravfogel; ACL 2022): train only the bias terms of the frozen backbone,
plus the shared head. No parameters are added.

On timm's ViT-B/16 the bias terms are
    per block: norm1 (768) + attn.qkv (2304) + attn.proj (768) + norm2 (768) + mlp.fc1 (3072) + mlp.fc2 (768) = 8,448
    x 12 blocks = 101,376, plus patch_embed.proj (768) and the final norm (768)  ->  102,912  (0.12% of the backbone)
`cls_token` and `pos_embed` are not biases and stay frozen.

Subset ablation (`bitfit.subset: q_fc2`): BitFit reports that the query bias and the second-MLP bias alone
(0.04% of BERT) recover most of the gain. On the fused qkv bias the k and v slices are kept fixed with a
gradient mask; the effective count is 12 * (768 + 768) = 18,432.
"""
import torch



def apply_bitfit(model, cfg):
    bcfg = cfg.get("bitfit") or {}
    exclude = tuple(bcfg.get("exclude", []))   # optional name prefixes to keep frozen, e.g. ["patch_embed"]
    subset = bcfg.get("subset", "all")         # "all" | "q_fc2": only the query bias b_q and the second MLP bias b_m2
    n_bias, n_eff = 0, 0
    dim = model.backbone.embed_dim
    for name, p in model.backbone.named_parameters():
        train = name.endswith(".bias") and not name.startswith(exclude)
        if subset == "q_fc2":
            train = train and (name.endswith("attn.qkv.bias") or name.endswith("mlp.fc2.bias"))
        p.requires_grad = train
        if not train:
            continue
        n_bias += p.numel()
        if subset == "q_fc2" and name.endswith("attn.qkv.bias"):
            # the fused qkv bias is one tensor [b_q | b_k | b_v]; keep b_k, b_v fixed by zeroing their gradient
            mask = torch.zeros_like(p); mask[:dim] = 1.0
            p.register_hook(lambda g, mask=mask: g * mask.to(g.device))   # mask is built before model.to(device)
            n_eff += dim
        else:
            n_eff += p.numel()
    for p in model.head.parameters():
        p.requires_grad = True
    model.backbone_no_grad = False       # gradients must flow through the whole network to reach every bias
    desc = "all backbone bias terms + head" if subset == "all" else "query bias + MLP fc2 bias only (BitFit's b_q, b_m2 subset) + head"
    return {"trainable": desc, "bias_params": n_bias, "effective_params": n_eff}
