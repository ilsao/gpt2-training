import os
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import PartialState
from accelerate.utils import broadcast_object_list

from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import default_data_collator, set_seed

from checkpoints import COMPLETE_MARKER, complete_checkpoints, mark_checkpoint_complete
from prepare_dataset import validate_prepared_dataset


def configure_model(config, tokenizer):
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    config.resid_pdrop = 0.0
    config.embd_pdrop = 0.0
    config.attn_pdrop = 0.0
    config.activation_function = "gelu_pytorch_tanh"
    return config


class ShutdownCallback(TrainerCallback):
    """Coordinate shutdown/save decisions only at safe, collective boundaries."""

    def __init__(self, stop_path, deadline=None, reserve_seconds=180, save_seconds=300):
        self.stop_path = Path(stop_path)
        self.deadline = deadline
        self.reserve_seconds = reserve_seconds
        self.save_seconds = save_seconds
        self.last_save = time.monotonic()
        self.stop_reason = None

    def request_stop(self, signum, frame):
        # Signal handlers never save, raise, or enter a distributed collective.
        self.stop_reason = signal.Signals(signum).name

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_save = time.monotonic()

    def _update_control(self, args, state, control):
        if self.deadline is not None and time.time() >= self.deadline - self.reserve_seconds:
            self.stop_reason = self.stop_reason or "time_limit"
        if self.stop_path.exists():
            self.stop_reason = self.stop_reason or "stop_requested"
        due = (state.global_step == 1 or time.monotonic() - self.last_save >= self.save_seconds)
        flags = torch.tensor(
            [bool(self.stop_reason), due or control.should_save],
            dtype=torch.int32, device=args.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        stop, save = flags.tolist()
        if stop:
            self.stop_reason = self.stop_reason or "stop_requested"
            control.should_training_stop = True
            control.should_evaluate = False
        control.should_save = bool(save or stop or control.should_training_stop)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        return self._update_control(args, state, control)

    def on_evaluate(self, args, state, control, **kwargs):
        # A warning received during evaluation must not start another training step.
        return self._update_control(args, state, control)

    def on_save(self, args, state, control, **kwargs):
        self.last_save = time.monotonic()


class CheckpointTrainer(Trainer):
    """Commit complete checkpoints before rotating previously committed ones."""

    def __init__(self, *args, shutdown_callback, **kwargs):
        self.shutdown_callback = shutdown_callback
        super().__init__(*args, **kwargs)
        if self.args.push_to_hub or self.args.save_only_model or self.args.save_on_each_node:
            raise ValueError("CheckpointTrainer requires local, complete, rank-zero checkpoint saves")
        self.add_callback(shutdown_callback)

    def _checkpoint_barrier(self):
        if dist.is_available() and dist.is_initialized():
            # CPU/Gloo must not receive accelerator device IDs (e.g. MPS on macOS).
            if self.args.device.type == "cuda":
                dist.barrier(device_ids=[self.args.device.index])
            else:
                dist.barrier()

    def _save_checkpoint(self, model, trial):
        checkpoint_dir = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
        if self.is_world_process_zero():
            # An overwritten directory must not retain an earlier completion marker.
            (checkpoint_dir / COMPLETE_MARKER).unlink(missing_ok=True)
        self._checkpoint_barrier()
        limit = self.args.save_total_limit
        try:
            # Trainer's built-in rotation counts incomplete directories as checkpoints.
            self.args.save_total_limit = None
            super()._save_checkpoint(model, trial)
        finally:
            self.args.save_total_limit = limit
        self._checkpoint_barrier()
        if self.is_world_process_zero():
            reason = self.shutdown_callback.stop_reason
            if reason is None:
                reason = "max_steps" if self.state.global_step >= self.state.max_steps else "periodic"
            mark_checkpoint_complete(checkpoint_dir, self.state.global_step, self.args.world_size, reason)
            if limit is not None and limit > 0:
                for path, _ in complete_checkpoints(checkpoint_dir.parent)[:-limit]:
                    shutil.rmtree(path)
            print(f"Complete checkpoint: {checkpoint_dir} ({reason})", flush=True)
        self._checkpoint_barrier()


def main():
    os.environ["WANDB_PROJECT"] = "gpt2-training"
    # Initialize distributed execution so both GPUs use the same output directory.
    state = PartialState()
    run_names = [
        (f"gpt2-{os.environ['SLURM_JOB_ID']}" if "SLURM_JOB_ID" in os.environ
         else f"gpt2-{datetime.now():%Y%m%d-%H%M%S-%f}")
        if state.is_main_process else None
    ]
    broadcast_object_list(run_names)
    run_name = run_names[0]

    set_seed(42)
    dataset, tokenizer, config = validate_prepared_dataset(os.environ["GPT2_DATA_DIR"])
    model = AutoModelForCausalLM.from_config(configure_model(config, tokenizer))
    output_dir = Path(os.environ["GPT2_OUTPUT_ROOT"]) / run_name
    deadline = os.environ.get("GPT2_TRAIN_DEADLINE", os.environ.get("SLURM_JOB_END_TIME"))
    shutdown = ShutdownCallback(output_dir / "stop-requested", float(deadline) if deadline else None)
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, shutdown.request_stop)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=1,
        max_steps=5000,
        learning_rate=1.25e-3,
        lr_scheduler_type="linear",
        warmup_steps=100,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_torch_fused",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        report_to="wandb",
        logging_steps=10,
        logging_first_step=True,
        include_num_input_tokens_seen=True,
        eval_strategy="steps",
        eval_steps=1000,
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        push_to_hub=False,
        seed=42,
        torch_compile=True,
    )

    trainer = CheckpointTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=default_data_collator,
        processing_class=tokenizer,
        shutdown_callback=shutdown,
    )

    trainer.train()


if __name__ == "__main__":
    main()
