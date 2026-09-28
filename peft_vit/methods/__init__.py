"""Method registry. Each `apply_<method>(model, cfg)` mutates the PEFTClassifier in place:
sets requires_grad flags, injects modules if needed, and returns a dict describing what is trainable."""
from .linear_probe import apply_linear_probe
from .full_ft import apply_full_ft

METHODS = {
    "linear_probe": apply_linear_probe,
    "full_ft": apply_full_ft,
    # "bitfit": apply_bitfit,      # M2
    # "lora": apply_lora,          # M2
    # "vpt_deep": apply_vpt_deep,  # M3
}


def apply_method(model, name, cfg):
    if name not in METHODS:
        raise KeyError(f"unknown method '{name}'. available: {sorted(METHODS)}")
    return METHODS[name](model, cfg)
