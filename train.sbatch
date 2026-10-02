#!/bin/bash
#SBATCH --job-name=gpt2
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge
set -euo pipefail

# Use the same Python module used to create .venv.
: "${NANO4_WORK_DIR:?Set NANO4_WORK_DIR to your writable /work directory}"
: "${NANO4_PYTHON_MODULE:?Set NANO4_PYTHON_MODULE to an available Python module}"
module load "$NANO4_PYTHON_MODULE"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# Cache downloads on /work; keep credentials in your private home directory.
export HF_HOME="$NANO4_WORK_DIR/huggingface"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export HF_HUB_REPO="${HF_HUB_REPO:-cerulean-works/gpt2-c4}"
export GPT2_OUTPUT_ROOT="$NANO4_WORK_DIR/gpt2-pretrain"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "$HF_HOME" "$GPT2_OUTPUT_ROOT"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2; assert torch.cuda.is_bf16_supported(); print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)'

echo "Start time: $(date -Iseconds)"
echo "Model output root: $GPT2_OUTPUT_ROOT"
echo "HF repository for later upload: $HF_HUB_REPO"
torchrun --standalone --nproc_per_node=2 train.py
echo "End time: $(date -Iseconds)"
