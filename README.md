# gpt2-training

Offline C4 preparation, two-GPU GPT-2 training, and a separate Hugging Face upload
job. Dropout is disabled and GeLU uses PyTorch's native tanh approximation.

## Prepare data once

On the training machine, load the Python module used to create `.venv`, install
`requirements.txt`, and run these commands from this repository:

```bash
module load miniconda3/26.1.1
source .venv/bin/activate
pip install -r requirements.txt
export NANO4_WORK_DIR=/work/asciibase64
export HF_HOME="$NANO4_WORK_DIR/huggingface"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export GPT2_DATA_DIR="$NANO4_WORK_DIR/gpt2-data/c4-1024"
mkdir -p logs
sbatch prepare.sbatch
```

Wait for the preparation job to finish successfully and print `Prepared ...` in
`logs/gpt2-data-<job-id>.out`, then run `bash submit.sh` to submit training/upload.
`prepare.sbatch` has its own 4-hour limit and uses the existing `8gpus` partition,
2 GPU allocation, 12 CPUs, and 128 GB RAM. The preparation program uses only CPU;
this allocation is independent of the training job's 30-minute limit.

Preparation runs separately from the 30-minute GPU training allocation. It streams
English `allenai/c4` into disk-backed Arrow shards: 1,280,000 training blocks and
1024 validation blocks, each containing 1024 tokens. This covers 5000 steps at
2 GPUs × batch size 128 without repeating an epoch. Token IDs and labels occupy
about 10.5 GB before Arrow overhead; allow additional space for the temporary
generator cache and Hugging Face downloads during preparation.

All dataset downloads, tokenization, and packing finish before training is
submitted, so none of that work consumes the training job's 30 minutes. The
training allocation sets `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` and reads
only the prepared local files. Missing or invalid prepared data fails validation
before submission; training never falls back to preparing or downloading it.

Use `sbatch prepare.sbatch --train-blocks N --eval-blocks N` for a smaller dataset.
If training exhausts a smaller dataset, Trainer starts another epoch until the
step/time limit.
Packing retains document EOS tokens and labels, seed 42, shuffle buffer 10000,
and drops the partial block at the end of each 1000-document batch.

The published directory contains `dataset/`, `tokenizer/`, `config/`, and
`metadata.json`. It appears only when preparation and validation succeed.
An existing output directory is never overwritten; use a new path to rebuild.
`train.py` reads this directory through `GPT2_DATA_DIR`, without downloading or
tokenizing C4 or fetching model/tokenizer files. The dataset and output paths
must be on shared storage accessible to both jobs.

## Submit training and upload

Run `hf auth login` with a token that has write access to the existing model
repository `asciibase64/gpt2-c4`, or set `HF_TOKEN`. Use the same `HF_TOKEN_PATH`
as above; never commit credentials. Override the target with `HF_HUB_REPO`.

```bash
bash submit.sh
```

Submission validates the local data, submits a held training job, registers an
upload job with `afterany:<training-job-id>`, then releases training. Registration
or release failures cancel the jobs created by the submission. Use this entrypoint
for automatic uploading; submitting `train.sbatch` alone does not register an
uploader. The script prints both job IDs and the exact run directory.

Both jobs use account `ACD115198`, partition `8gpus`, 2 GPUs, 12 CPUs, and 128 GB
RAM. Training uses `torchrun` with two ranks and a **30-minute allocation limit**.
The uploader runs one Python process with a **1-hour limit**; its allocation
still reserves the chosen GPU resources. Jobs do not automatically requeue.

The environment defaults can be overridden before submission:

| Variable | Default |
| --- | --- |
| `NANO4_WORK_DIR` | `/work/asciibase64` |
| `NANO4_PYTHON_MODULE` | `miniconda3/26.1.1` |
| `GPT2_DATA_DIR` | `$NANO4_WORK_DIR/gpt2-data/c4-1024` |
| `GPT2_OUTPUT_ROOT` | `$NANO4_WORK_DIR/gpt2-pretrain` |
| `HF_HOME` | `$NANO4_WORK_DIR/huggingface` |
| `HF_TOKEN_PATH` | `$HOME/.cache/huggingface/token` |
| `HF_HUB_REPO` | `asciibase64/gpt2-c4` |
| `WANDB_MODE` | `online` |

## Checkpoints and interruptions

Outputs live in `$GPT2_OUTPUT_ROOT/gpt2-<training-job-id>/`. Checkpoints are saved
after the first completed step, every 250 steps, whenever 5 minutes have elapsed
since the last save, and at normal or requested shutdown. Wall-clock checks occur
at optimizer step boundaries, so a long step may extend the save interval.

Slurm sends the batch shell `USR1` about 3 minutes before the allocation ends
(it can arrive up to 60 seconds earlier). The shell writes `stop-requested`;
all ranks synchronize the stop decision at a step boundary, cancel pending
evaluation, save, and exit. A second check uses the Slurm end timestamp, including
setup/compilation time. A warning received during evaluation is handled when
evaluation completes. No checkpoint saving or distributed communication occurs
inside Python signal handlers.

Every checkpoint contains model weights/config, tokenizer, optimizer, scheduler,
Trainer state, training arguments, and each rank's RNG state. After every rank
finishes saving, rank zero atomically writes `checkpoint-complete.json`, then
retains the newest two complete checkpoints. Incomplete directories never
replace or delete the last usable checkpoint.

The upload job starts after training ends, including failure, timeout, or
cancellation. It uploads only the newest validated checkpoint from that run to
the Hub repository root; it does not upload the whole run directory. Temporary
network failures get up to three attempts, with 10/30-second retry delays.
Successful commits are recorded in `upload-result.json` and printed in the log.
Authentication errors, missing checkpoints, and exhausted retries exit nonzero.

A graceful stop saves the last completed optimizer step. `SIGKILL`, node failure,
or an interrupted save can only fall back to the last complete checkpoint. If
the first checkpoint has not completed yet, the uploader reports that none is
available. Local data is retained if the Hub is unavailable. Cancelling both
jobs also cancels the uploader; to request a graceful stop, signal only the
training batch shell:

```bash
scancel --batch --signal=USR1 TRAINING_JOB_ID
```

To retry an upload from an allowed node with Hub access:

```bash
python push_to_hub.py --run-dir "$GPT2_OUTPUT_ROOT/gpt2-TRAINING_JOB_ID"
# Or register another upload allocation:
sbatch upload.sbatch "$GPT2_OUTPUT_ROOT/gpt2-TRAINING_JOB_ID"
```

The complete checkpoint can be passed to Trainer's `resume_from_checkpoint`.
`train.py` starts a new run by default; automatic resubmission/resume is not enabled.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
bash -n job_env.sh submit.sh prepare.sbatch train.sbatch upload.sbatch
```

Tests use local synthetic data/model/tokenizer, two CPU ranks, mocked Slurm, and
mocked Hub calls. They do not download C4, submit real jobs, or upload to the Hub.
On the cluster, additionally check normal completion, an early batch `USR1`, and
a forced termination after a complete save: each should release its registered
upload job, with the expected checkpoint step visible in the upload log/Hub commit.
