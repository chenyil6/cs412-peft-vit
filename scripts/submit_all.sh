#!/bin/bash
# Submit the full M4 grid: every method x every seed, one Slurm job each, to the 1x-billed batch partition.
#   scripts/submit_all.sh                    # submit 5 methods x seeds 0 1 2
#   DRY_RUN=1 scripts/submit_all.sh          # print the sbatch commands only
#   METHODS="bitfit lora" SEEDS="0" scripts/submit_all.sh
#   PARTITION=gpuA100x4-preempt EXTRA="--requeue" scripts/submit_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
METHODS=${METHODS:-"linear_probe bitfit lora vpt_deep full_ft"}
SEEDS=${SEEDS:-"0 1 2"}
PARTITION=${PARTITION:-gpuA100x4}
EXTRA=${EXTRA:-}
for m in $METHODS; do
  for s in $SEEDS; do
    if [ -f "runs/${m}_s${s}/summary.json" ]; then echo "skip ${m}_s${s}: summary.json exists"; continue; fi
    cmd="sbatch --job-name=${m}_s${s} --partition=${PARTITION} ${EXTRA} scripts/train.sbatch configs/${m}.yaml --seed ${s}"
    if [ -n "${DRY_RUN:-}" ]; then echo "$cmd"; else echo -n "${m}_s${s}: "; $cmd; fi
  done
done
