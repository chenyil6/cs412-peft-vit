# A Unified Empirical Comparison of PEFT Methods for Vision Transformers

CS-412 term project (Fall 2026). Linear probe / BitFit / LoRA / VPT-deep / full fine-tuning on
FGVC-Aircraft with a single frozen ImageNet-21k ViT-B/16 backbone, identical data pipeline,
identical head, identical training loop. See `CS_412_proposal.pdf`.

## Layout
```
train.py                  single entry point for every method
peft_vit/data.py          FGVC-Aircraft dataset + VPT-style transforms
peft_vit/model.py         backbone loading (timm) + shared linear head
peft_vit/methods/         one file per method; each sets requires_grad / injects modules
peft_vit/utils.py         seeding, param counting, cosine schedule, logging
configs/<method>.yaml     hyper-parameters per method
scripts/train.sbatch      Slurm job (Delta, 1x A100)
scripts/preprocess_resize.py   one-off: crop banner + resize shorter side to 256
runs/<method>_s<seed>/    metrics.csv, summary.json, trainable_params.txt
```

## Setup (NCSA Delta)
```bash
conda activate /u/hchen56/conda-envs/cs412
# data: tarball in /projects/bilw/hchen56/datasets, extracted to /work/nvme/bilw/hchen56/datasets
python scripts/preprocess_resize.py --root /work/nvme/bilw/hchen56/datasets/fgvc-aircraft-2013b
```

## Run
```bash
python train.py --config configs/linear_probe.yaml --smoke         # sanity check (CPU ok)
sbatch scripts/train.sbatch configs/linear_probe.yaml --seed 0     # 1x A100
```

## Protocol notes
* Train on `trainval` (6,667 images), report **last-epoch** top-1 on `test` (3,333). No early
  stopping and no hyper-parameter search, so the test set is never used for selection.
* All methods share: backbone checkpoint, input normalisation (from the timm pretrained config),
  augmentation (Resize 256 -> RandomCrop 224 -> HFlip), head init, epochs, cosine schedule with warmup.
* Reported per run: last-epoch accuracy, trainable parameters (with and without the head),
  peak GPU memory (`torch.cuda.max_memory_allocated`), mean epoch time and total training time.
* Hyper-parameter selection, when any is done (e.g. the linear-probe lr, which no paper fixes),
  uses the official `train` (3,334) / `val` (3,333) split: `scripts/lp_lr_sweep.sbatch` with
  `data.train_split=train data.test_split=val`. The test split is only ever used for the final
  trainval-trained runs.
* Per-method optimizer / lr / weight decay follow each method's original paper (sources and the
  reasoning are in `report.md`, section 3-4); shared budget: 50 epochs, 5 warmup, cosine, batch 128.
  The linear-probe lr (3.0, SGD) is the only value selected on the val split.
