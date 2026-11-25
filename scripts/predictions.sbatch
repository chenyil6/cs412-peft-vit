#!/bin/bash
#SBATCH --job-name=preds
#SBATCH --account=bilw-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/%x-%j.out
# 1) retrain full FT seed 0 with the checkpoint kept (the main run saved none), 2) dump test predictions of all 5 models.
set -euo pipefail
source /u/hchen56/miniconda3/etc/profile.d/conda.sh; conda activate /u/hchen56/conda-envs/cs412
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=/u/hchen56/CS-412; cd /u/hchen56/CS-412
[ -f runs/_pred/full_ft_s0/trainable.pt ] || python train.py --config configs/full_ft.yaml --seed 0 --out_dir runs/_pred/full_ft_s0 --set train.save_ckpt=true | grep -E "^\[(params|done)"
python scripts/eval_predictions.py --ckpt linear_probe=runs/linear_probe_s0 bitfit=runs/bitfit_s0 lora=runs/lora_s0 vpt_deep=runs/vpt_deep_s0 full_ft=runs/_pred/full_ft_s0
