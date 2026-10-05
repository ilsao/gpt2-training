#!/bin/bash
#SBATCH --job-name=gpt2-upload
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./job_env.sh
python push_to_hub.py --run-dir "$1" --repo-id "$HF_HUB_REPO"
