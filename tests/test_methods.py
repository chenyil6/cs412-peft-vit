"""Unit tests for the from-scratch PEFT methods (CPU, random-init backbone, ~20 s).

    python tests/test_methods.py        # or: pytest tests/

Checks: exact trainable-parameter counts, that only the intended tensors receive gradients,
that LoRA is an identity at initialisation (B = 0), and that merging LoRA into the fused qkv weight
reproduces the un-merged forward pass.
"""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from peft_vit.methods import apply_method                      # noqa: E402
from peft_vit.methods.lora import LoRAQKV, merge_lora, unmerge_lora  # noqa: E402
from peft_vit.methods.vpt_deep import VPTDeepBlocks               # noqa: E402
from peft_vit.model import PEFTClassifier, load_backbone       # noqa: E402

DIM, DEPTH, NUM_CLASSES = 768, 12, 100
HEAD = DIM * NUM_CLASSES + NUM_CLASSES                         # 76,900
BIAS = DEPTH * (768 + 2304 + 768 + 768 + 3072 + 768) + 768 + 768   # 102,912
LORA_CFG = {"lora": {"r": 8, "alpha": 8, "targets": ["q", "v"], "dropout": 0.0}}
LORA = DEPTH * 2 * (2 * 8 * DIM)                               # 294,912
VPT_CFG = {"vpt": {"num_tokens": 50, "deep": True, "dropout": 0.1, "init": "xavier_uniform"}}
VPT = DEPTH * 50 * DIM                                          # 460,800


def base_model(seed=0):
    torch.manual_seed(seed)
    backbone, _ = load_backbone("vit_b16_in21k", pretrained=False)
    return PEFTClassifier(backbone, NUM_CLASSES).eval()


def trainable(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def test_linear_probe_counts():
    m = base_model(); apply_method(m, "linear_probe", {})
    t = trainable(m)
    assert set(t) == {"head.weight", "head.bias"}
    assert sum(p.numel() for p in t.values()) == HEAD


def test_bitfit_counts_and_names():
    m = base_model(); apply_method(m, "bitfit", {})
    t = trainable(m)
    backbone_names = [n for n in t if n.startswith("backbone.")]
    assert all(n.endswith(".bias") for n in backbone_names), backbone_names[:3]
    assert sum(t[n].numel() for n in backbone_names) == BIAS
    assert sum(p.numel() for p in t.values()) == BIAS + HEAD
    assert "backbone.cls_token" not in t and "backbone.pos_embed" not in t


def test_lora_counts_and_names():
    m = base_model(); apply_method(m, "lora", LORA_CFG)
    t = trainable(m)
    lora_names = [n for n in t if ".lora_" in n]
    assert len(lora_names) == DEPTH * 2 * 2                    # A and B for q and v in every block
    assert all(n.endswith((".lora_A.q", ".lora_B.q", ".lora_A.v", ".lora_B.v")) for n in lora_names)
    assert sum(t[n].numel() for n in lora_names) == LORA
    assert sum(p.numel() for p in t.values()) == LORA + HEAD
    assert all(isinstance(b.attn.qkv, LoRAQKV) for b in m.backbone.blocks)


def test_lora_is_identity_at_init():
    ref = base_model(); lo = copy.deepcopy(ref); apply_method(lo, "lora", LORA_CFG)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(ref(x), lo(x), atol=1e-6)


def test_lora_merge_matches_unmerged():
    m = base_model(); apply_method(m, "lora", LORA_CFG)
    with torch.no_grad():                                      # give B non-zero values so the merge does something
        for mod in m.modules():
            if isinstance(mod, LoRAQKV):
                for t in mod.targets:
                    mod.lora_B[t].normal_(std=0.02)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y_unmerged = m(x)
        w0 = m.backbone.blocks[0].attn.qkv.qkv.weight.clone()
        merge_lora(m); y_merged = m(x)
        assert all(mod.merged for mod in m.modules() if isinstance(mod, LoRAQKV))
        unmerge_lora(m); y_back = m(x)
    assert torch.allclose(y_unmerged, y_merged, atol=1e-4), (y_unmerged - y_merged).abs().max()
    assert torch.allclose(y_unmerged, y_back, atol=1e-5)
    assert torch.allclose(m.backbone.blocks[0].attn.qkv.qkv.weight, w0, atol=1e-6)
    assert not torch.allclose(y_unmerged, base_model()(x))    # sanity: the perturbed B changed the output


def test_vpt_deep_counts_and_names():
    m = base_model(); apply_method(m, "vpt_deep", VPT_CFG)
    t = trainable(m)
    assert set(t) == {"backbone.blocks.prompts", "head.weight", "head.bias"}
    assert t["backbone.blocks.prompts"].shape == (DEPTH, 50, DIM)
    assert sum(p.numel() for p in t.values()) == VPT + HEAD
    v = (6.0 / (3 * 16 * 16 + DIM)) ** 0.5
    assert t["backbone.blocks.prompts"].abs().max() <= v and t["backbone.blocks.prompts"].abs().max() > 0.9 * v
    assert isinstance(m.backbone.blocks, VPTDeepBlocks) and len(m.backbone.blocks) == DEPTH


def test_vpt_deep_prompts_replaced_at_every_layer():
    """Every block must see [CLS, P_i, 196 patches] at its input, with P_i the prompts of THAT layer."""
    m = base_model(); apply_method(m, "vpt_deep", VPT_CFG); m.eval()      # eval: prompt dropout off
    seen = {}
    hooks = [blk.register_forward_pre_hook(lambda mod, inp, i=i: seen.__setitem__(i, inp[0].detach()))
             for i, blk in enumerate(m.backbone.blocks.blocks)]
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = m(x)
    for h in hooks:
        h.remove()
    assert out.shape == (2, NUM_CLASSES)
    prompts = m.backbone.blocks.prompts
    for i in range(DEPTH):
        inp = seen[i]
        assert inp.shape == (2, 1 + 50 + 196, DIM), inp.shape
        assert torch.allclose(inp[:, 1:51], prompts[i].expand(2, -1, -1)), f"layer {i}: prompt slice is not P_{i}"
        if i > 0:
            assert not torch.allclose(inp[:, 1:51], prompts[i - 1].expand(2, -1, -1))


def test_gradients_reach_only_trainable_tensors():
    for method, cfg in [("bitfit", {}), ("lora", LORA_CFG), ("vpt_deep", VPT_CFG), ("linear_probe", {})]:
        m = base_model(); apply_method(m, method, cfg); m.train()
        if method == "lora":                                   # with B = 0 the gradient of A is exactly zero (dL/dA = B^T ...)
            with torch.no_grad():
                for mod in m.modules():
                    if isinstance(mod, LoRAQKV):
                        for t in mod.targets:
                            mod.lora_B[t].normal_(std=0.02)
        x = torch.randn(2, 3, 224, 224)
        loss = m(x).logsumexp(-1).mean(); loss.backward()
        for n, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and p.grad.abs().sum() > 0, f"{method}: no gradient for {n}"
            else:
                assert p.grad is None, f"{method}: frozen tensor {n} received a gradient"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"PASS  {t.__name__}")
    print(f"{len(tests)} tests passed")
