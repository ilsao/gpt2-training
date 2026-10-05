#!/bin/bash
# Shared environment for submission, data preparation, training, and upload jobs.
module purge
export NANO4_WORK_DIR="${NANO4_WORK_DIR:-/work/asciibase64}"
export NANO4_PYTHON_MODULE="${NANO4_PYTHON_MODULE:-miniconda3/26.1.1}"
module load "$NANO4_PYTHON_MODULE"

source .venv/bin/activate
export HF_HOME="${HF_HOME:-$NANO4_WORK_DIR/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HF_HUB_REPO="${HF_HUB_REPO:-asciibase64/gpt2-c4}"
export GPT2_DATA_DIR="${GPT2_DATA_DIR:-$NANO4_WORK_DIR/gpt2-data/c4-1024}"
export GPT2_OUTPUT_ROOT="${GPT2_OUTPUT_ROOT:-$NANO4_WORK_DIR/gpt2-pretrain}"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
