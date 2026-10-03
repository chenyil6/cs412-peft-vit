#!/bin/bash
#SBATCH --job-name=peft
#SBATCH --account=bilw-delta-gpu
#SBATCH --partition=gpuA100x4
# Billing per GPU-hour (TRESBillingWeights): gpuA100x4 = 1x, gpuA100x4-preempt = 0.5x (may be requeued),
# gpuA100x4-interactive = 2x (max 1 h, short queue). Override with: sbatch --partition=... 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# usage:  sbatch scripts/train.sbatch configs/linear_probe.yaml [--seed 0] [--set k=v ...]
#         sbatch --job-name=lp_s1 scripts/train.sbatch configs/linear_probe.yaml --seed 1
set -euo pipefail
CONFIG=$1; shift

source /u/hchen56/miniconda3/etc/profile.d/conda.sh
conda activate /u/hchen56/conda-envs/cs412
export HF_HUB_OFFLINE=1          # backbone weights are pre-downloaded on the login node
export OMP_NUM_THREADS=4

cd /u/hchen56/CS-412
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) job=$SLURM_JOB_ID"
srun python train.py --config "$CONFIG" "$@"
