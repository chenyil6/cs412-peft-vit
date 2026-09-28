#!/bin/bash
#SBATCH --job-name=lp_sweep
#SBATCH --account=bilw-delta-gpu
#SBATCH --partition=gpuA100x4
# Billing per GPU-hour (TRESBillingWeights): gpuA100x4 = 1x, gpuA100x4-preempt = 0.5x (may be requeued),
# gpuA100x4-interactive = 2x (max 1 h, short queue). Override with: sbatch --partition=... 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:59:00
#SBATCH --output=slurm_logs/%x-%j.out
# Linear-probe lr sensitivity: several short runs sequentially in one allocation (~3 min each).
#   sbatch scripts/lp_lr_sweep.sbatch sgd_lr1.0:optim.lr=1.0 adamw_lr1e-2:"optim.name=adamw optim.lr=0.01"
# Each argument is <tag>:<space-separated --set overrides>. Output goes to runs/lp_lr_sweep/<tag>.
set -uo pipefail
source /u/hchen56/miniconda3/etc/profile.d/conda.sh
conda activate /u/hchen56/conda-envs/cs412
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
cd /u/hchen56/CS-412
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) job=$SLURM_JOB_ID"
for spec in "$@"; do
  tag=${spec%%:*}; overrides=${spec#*:}
  echo "##### $tag  ($overrides)"
  python train.py --config configs/linear_probe.yaml --seed 0 --out_dir runs/lp_lr_sweep/$tag --set $overrides 2>&1 \
    | grep -E "^\[(ep +(10|20|30|40|50)/|done|params)|Traceback|Error"
done
echo "SWEEP_DONE"
