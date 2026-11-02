"""Inference cost of each adapted model (the deployment side of the comparison).

Measures images/s and ms/batch at batch 128, bf16, after warm-up, for:
  plain ViT-B/16 + head          (= linear probe, BitFit, full fine-tuning: same architecture at inference)
  LoRA r=8 on q,v, un-merged     (extra low-rank matmuls in every block)
  LoRA r=8 on q,v, merged        (folded into the fused qkv weight: should equal plain)
  VPT-deep p=50                  (50 extra tokens in every block)
Writes results/inference_profile.json.
"""
import json, time
from pathlib import Path
import torch, yaml
from peft_vit.model import PEFTClassifier, load_backbone
from peft_vit.methods import apply_method
from peft_vit.methods.lora import merge_lora

cfg = yaml.safe_load(open("configs/lora.yaml")); vcfg = yaml.safe_load(open("configs/vpt_deep.yaml"))
dev = torch.device("cuda"); B = 128; x = torch.randn(B, 3, 224, 224, device=dev)

def build(method, c):
    bb, _ = load_backbone(c["backbone"], weights=c.get("backbone_weights")); m = PEFTClassifier(bb, 100); apply_method(m, method, c); return m.to(dev).eval()

@torch.no_grad()
def bench(model, iters=50):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(10): model(x)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(iters): model(x)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / iters
    return {"ms_per_batch": 1000 * dt, "img_per_s": B / dt, "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}

res = {}
res["plain"] = bench(build("linear_probe", cfg))
lora = build("lora", cfg); res["lora_unmerged"] = bench(lora); merge_lora(lora); res["lora_merged"] = bench(lora)
res["vpt_deep_p50"] = bench(build("vpt_deep", vcfg))
res["gpu"] = torch.cuda.get_device_name(0); res["batch"] = B
Path("results").mkdir(exist_ok=True); json.dump(res, open("results/inference_profile.json", "w"), indent=2)
for k, v in res.items():
    if isinstance(v, dict): print(f"{k:16s} {v['ms_per_batch']:7.1f} ms/batch  {v['img_per_s']:7.0f} img/s")
