"""Full fine-tuning: every backbone parameter and the head are trainable. Upper-bound reference."""


def apply_full_ft(model, cfg):
    for p in model.parameters():
        p.requires_grad = True
    model.backbone_no_grad = False
    return {"trainable": "all parameters"}
