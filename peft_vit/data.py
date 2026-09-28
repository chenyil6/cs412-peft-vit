"""FGVC-Aircraft (variant level, 100 classes) dataset and transforms.

Directory layout expected under `root` (same as the official tarball, or the pre-resized copy
produced by scripts/preprocess_resize.py):
    root/data/images/<id>.jpg
    root/data/images_variant_{train,val,trainval,test}.txt   lines: "<id> <variant name>"
    root/data/variants.txt                                    one variant name per line
"""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class FGVCAircraft(Dataset):
    def __init__(self, root, split="trainval", transform=None):
        data_dir = Path(root) / "data"
        self.classes = [l.strip() for l in open(data_dir / "variants.txt") if l.strip()]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        with open(data_dir / f"images_variant_{split}.txt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_id, variant = line.split(" ", 1)
                self.samples.append((data_dir / "images" / f"{img_id}.jpg", self.class_to_idx[variant]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, label


def build_transforms(mean, std, resize=256, crop=224):
    """Train: Resize(shorter side) -> RandomCrop -> HFlip. Eval: Resize -> CenterCrop.
    This is the FGVC recipe used by the VPT paper. `mean`/`std` come from the backbone's
    pretrained config so every method sees identically normalised inputs."""
    norm = T.Normalize(mean, std)
    train_tf = T.Compose([
        T.Resize(resize),
        T.RandomCrop(crop),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        norm,
    ])
    eval_tf = T.Compose([
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf
