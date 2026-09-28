"""Method registry. Each `apply_<method>(model, cfg)` mutates the PEFTClassifier in place:
sets requires_grad flags, injects modules if needed, and returns a dict describing what is trainable."""
from .bitfit import apply_bitfit
from .full_ft import apply_full_ft
from .linear_probe import apply_linear_probe
from .lora import apply_lora
from .vpt_deep import apply_vpt_deep

METHODS = {
    "linear_probe": apply_linear_probe,
    "bitfit": apply_bitfit,
    "lora": apply_lora,
    "vpt_deep": apply_vpt_deep,
    "full_ft": apply_full_ft,
}


def apply_method(model, name, cfg):
    if name not in METHODS:
        raise KeyError(f"unknown method '{name}'. available: {sorted(METHODS)}")
    return METHODS[name](model, cfg)
