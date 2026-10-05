"""Executable fixture for real two-rank save/shutdown/resume tests."""

import argparse
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, GPT2Config, TrainerCallback, TrainingArguments, set_seed
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from checkpoints import complete_checkpoints, latest_complete_checkpoint
from train import CheckpointTrainer, ShutdownCallback, configure_model


def tiny_tokenizer():
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "word": 2, "other": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]",
        bos_token="[EOS]", pad_token="[EOS]",
    )


class InterruptSaveTrainer(CheckpointTrainer):
    def _save_optimizer_and_scheduler(self, output_dir):
        if self.state.global_step == 5 and self.is_world_process_zero():
            # Model files already exist; there must not yet be a completion marker.
            os.kill(os.getpid(), signal.SIGKILL)
        return super()._save_optimizer_and_scheduler(output_dir)


class StopOneRank(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if args.process_index == 1 and state.global_step == 3:
            os.kill(os.getpid(), signal.SIGUSR1)


def make_trainer(run_dir, steps, stop=False, interrupt_save=False):
    set_seed(42)
    tokenizer = tiny_tokenizer()
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=32, n_embd=8, n_layer=1, n_head=2,
                        bos_token_id=1, eos_token_id=1)
    model = AutoModelForCausalLM.from_config(configure_model(config, tokenizer))
    dataset = Dataset.from_dict({
        "input_ids": [[2, 3, 2, 1, 3, 2, 2, 1]] * 32,
        "labels": [[2, 3, 2, 1, 3, 2, 2, 1]] * 32,
    })
    shutdown = ShutdownCallback(Path(run_dir) / "stop-requested")
    signal.signal(signal.SIGUSR1, shutdown.request_stop)
    args = TrainingArguments(
        output_dir=str(run_dir), max_steps=steps, use_cpu=True,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        learning_rate=1e-3, lr_scheduler_type="constant", optim="adamw_torch",
        save_strategy="steps", save_steps=250, save_total_limit=2,
        eval_strategy="steps", eval_steps=3, report_to="none", disable_tqdm=True,
        ddp_find_unused_parameters=False, dataloader_num_workers=0,
    )
    cls = InterruptSaveTrainer if interrupt_save else CheckpointTrainer
    return cls(
        model=model, args=args, train_dataset=dataset, eval_dataset=dataset,
        processing_class=tokenizer, shutdown_callback=shutdown,
        callbacks=[StopOneRank()] if stop else [],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--interrupt-save", action="store_true")
    args = parser.parse_args()
    if args.interrupt_save:
        trainer = make_trainer(args.run_dir, 5, interrupt_save=True)
        trainer.train(resume_from_checkpoint=str(args.run_dir / "checkpoint-4"))
        raise AssertionError("Expected rank-zero SIGKILL during checkpoint-5 save")

    trainer = make_trainer(args.run_dir, 8, stop=True)
    trainer.train()
    assert trainer.state.global_step == 3
    assert not any("eval_loss" in row for row in trainer.state.log_history)
    path, manifest = latest_complete_checkpoint(args.run_dir)
    assert path.name == "checkpoint-3" and manifest["world_size"] == 2
    torch.distributed.barrier()

    trainer = make_trainer(args.run_dir, 4)
    trainer.train(resume_from_checkpoint=str(path))
    assert trainer.state.global_step == 4
    assert [item[1]["global_step"] for item in complete_checkpoints(args.run_dir)] == [3, 4]
    final, manifest = latest_complete_checkpoint(args.run_dir)
    assert manifest["reason"] == "max_steps"
    assert (final / "rng_state_0.pth").is_file() and (final / "rng_state_1.pth").is_file()
    optimizer = torch.load(final / "optimizer.pt", map_location="cpu", weights_only=True)
    assert all(state["step"].item() == 4 for state in optimizer["state"].values())
    AutoModelForCausalLM.from_pretrained(final, local_files_only=True)
    torch.distributed.barrier()
    if trainer.is_world_process_zero():
        print("DISTRIBUTED_SAVE_RESUME_OK", flush=True)


if __name__ == "__main__":
    main()
