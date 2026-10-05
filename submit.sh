#!/bin/bash
# Register the uploader before allowing training to start.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./job_env.sh

python -c 'import os; from prepare_dataset import validate_prepared_dataset; validate_prepared_dataset(os.environ["GPT2_DATA_DIR"])'
mkdir -p logs "$HF_HOME" "$GPT2_OUTPUT_ROOT"

train_job_id=""
upload_job_id=""
released=0
cleanup() {
    if [[ "$released" == 0 ]]; then
        if [[ -n "$upload_job_id" ]]; then scancel "$upload_job_id" || true; fi
        if [[ -n "$train_job_id" ]]; then scancel "$train_job_id" || true; fi
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

response=$(sbatch --parsable --hold train.sbatch)
train_job_id=${response%%;*}
[[ "$train_job_id" =~ ^[0-9]+$ ]] || { train_job_id=""; echo "Invalid training job ID: $response" >&2; exit 1; }
run_dir="$GPT2_OUTPUT_ROOT/gpt2-$train_job_id"
response=$(sbatch --parsable --dependency="afterany:$train_job_id" upload.sbatch "$run_dir")
upload_job_id=${response%%;*}
[[ "$upload_job_id" =~ ^[0-9]+$ ]] || { upload_job_id=""; echo "Invalid upload job ID: $response" >&2; exit 1; }
scontrol release "$train_job_id"
released=1
echo "Training job: $train_job_id; upload job: $upload_job_id"
echo "Run directory: $run_dir"
