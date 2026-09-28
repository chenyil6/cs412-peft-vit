"""Linear probe: backbone fully frozen, only the classification head is trained.
Lower-bound baseline. Any gain another method shows over this is attributable to its
adaptation mechanism, since the head is identical across methods."""


def apply_linear_probe(model, cfg):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    model.backbone_no_grad = True
    return {"trainable": "head only"}
