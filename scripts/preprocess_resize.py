"""Pre-resize FGVC-Aircraft images once so training does not decode ~1000x700 JPEGs every epoch.

Steps per image:
  1. Crop the 20-pixel copyright banner at the bottom (documented in the dataset README).
  2. Resize so the shorter side is `--short` pixels (default 256), preserving aspect ratio.
  3. Save as JPEG quality 95 under <root>-r<short>/data/images/.
Annotation files (images_*.txt, variants.txt, ...) are copied unchanged, so the resized
folder has the same layout as the original and can be used as a drop-in `data.root`.

Both Resize(256) in the training pipeline and this script resize the shorter side, so the
eval pipeline (Resize 256 -> CenterCrop 224) is unaffected by pre-resizing.
"""
import argparse
import shutil
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

BANNER_PX = 20


def process(args):
    src, dst, short = args
    if dst.exists():
        return 0
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        im = im.crop((0, 0, w, h - BANNER_PX))
        w, h = im.size
        scale = short / min(w, h)
        if scale < 1.0:
            im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        im.save(dst, quality=95)
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="path to fgvc-aircraft-2013b")
    p.add_argument("--short", type=int, default=256)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()

    root = Path(a.root)
    out = root.parent / f"{root.name}-r{a.short}"
    (out / "data" / "images").mkdir(parents=True, exist_ok=True)
    for f in (root / "data").glob("*.txt"):
        shutil.copy(f, out / "data" / f.name)

    srcs = sorted((root / "data" / "images").glob("*.jpg"))
    jobs = [(s, out / "data" / "images" / s.name, a.short) for s in srcs]
    with Pool(a.workers) as pool:
        done = sum(pool.imap_unordered(process, jobs, chunksize=64))
    print(f"resized {done} / {len(jobs)} images -> {out}")


if __name__ == "__main__":
    main()
