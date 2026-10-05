#!/bin/bash
#SBATCH --job-name=gpt2
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./job_env.sh

# Download/tokenize/pack with prepare_dataset.py before submitting this job.
# This allocation only reads the published local dataset and model assets.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "$HF_HOME" "$GPT2_OUTPUT_ROOT"
run_dir="$GPT2_OUTPUT_ROOT/gpt2-$SLURM_JOB_ID"
mkdir -p "$run_dir"
wait_interrupted=0
request_stop() {
    wait_interrupted=1
    touch "$run_dir/stop-requested"
}
trap request_stop USR1 TERM INT
# Older Slurm installations may not export the projected end timestamp.
export GPT2_TRAIN_DEADLINE="${SLURM_JOB_END_TIME:-$((${SLURM_JOB_START_TIME:-$(date +%s)} + 1800))}"

nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2; assert torch.cuda.is_bf16_supported(); print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)'

echo "Start time: $(date -Iseconds)"
echo "Model output root: $GPT2_OUTPUT_ROOT"
echo "HF repository for later upload: $HF_HUB_REPO"
torchrun --standalone --nproc_per_node=2 train.py &
training_pid=$!
while true; do
    wait_interrupted=0
    if wait "$training_pid"; then status=0; else status=$?; fi
    # A trapped signal interrupts wait; wait again to collect the child's real status.
    if [[ "$wait_interrupted" == 0 ]]; then break; fi
done
echo "End time: $(date -Iseconds)"
exit "$status"
