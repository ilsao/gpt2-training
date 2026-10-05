#!/bin/bash
#SBATCH --job-name=gpt2-data
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./job_env.sh

# Preparation downloads/tokenizes data in its own allocation, before training.
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
mkdir -p "$HF_HOME"
echo "Dataset output directory: $GPT2_DATA_DIR"
python prepare_dataset.py --output-dir "$GPT2_DATA_DIR" "$@"
