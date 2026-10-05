#!/bin/bash
# Register the uploader before allowing training to start.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./job_env.sh

python -c 'import os; from prepare_dataset import validate_prepared_dataset; validate_prepared_dataset(os.environ["GPT2_DATA_DIR"])'
mkdir -p logs "$HF_HOME" "$GPT2_OUTPUT_ROOT"

train_job_id=""
upload_job_id=""
ready_tmp=""
registered=0
phase="submitting training job"
cleanup() {
    status=$?
    if [[ "$registered" == 0 ]]; then
        echo "Submission stopped while $phase; cancelling jobs created by this submission." >&2
        if [[ -n "$upload_job_id" ]]; then scancel "$upload_job_id" || true; fi
        if [[ -n "$train_job_id" ]]; then scancel "$train_job_id" || true; fi
    fi
    if [[ -n "$ready_tmp" ]]; then rm -f "$ready_tmp"; fi
    return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

response=$(sbatch --parsable train.sbatch --wait-for-upload)
train_job_id=${response%%;*}
[[ "$train_job_id" =~ ^[0-9]+$ ]] || { train_job_id=""; echo "Invalid training job ID: $response" >&2; exit 1; }
run_dir="$GPT2_OUTPUT_ROOT/gpt2-$train_job_id"
echo "Training job: $train_job_id"
echo "Run directory: $run_dir"
phase="registering upload job"
response=$(sbatch --parsable --dependency="afterany:$train_job_id" upload.sbatch "$run_dir")
upload_job_id=${response%%;*}
[[ "$upload_job_id" =~ ^[0-9]+$ ]] || { upload_job_id=""; echo "Invalid upload job ID: $response" >&2; exit 1; }
echo "Upload job: $upload_job_id (afterany:$train_job_id)"
phase="publishing upload registration"
mkdir -p "$run_dir"
ready_file="$run_dir/upload-registered"
[[ ! -e "$ready_file" ]] || { echo "Upload registration marker already exists: $ready_file" >&2; exit 1; }
ready_tmp=$(mktemp "$run_dir/.upload-registered.XXXXXX")
printf '%s\n' "$upload_job_id" > "$ready_tmp"
mv "$ready_tmp" "$ready_file"
ready_tmp=""
registered=1
echo "Upload registered; training may start."
