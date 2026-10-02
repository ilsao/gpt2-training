import os
from datetime import datetime

from accelerate import PartialState
from accelerate.utils import broadcast_object_list
from datasets import Dataset, load_dataset

from transformers import Trainer, TrainingArguments
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import default_data_collator, set_seed

BLOCK_SIZE = 1024


def pack_texts(examples, tokenizer):
    """Pack documents with EOS boundaries; drop only the last partial batch block."""
    documents = tokenizer(
        examples["text"], add_special_tokens=False, truncation=False,
        return_attention_mask=False, verbose=False,
    )["input_ids"]
    tokens = []
    for document in documents:
        tokens.extend(document)
        tokens.append(tokenizer.eos_token_id)
    length = len(tokens) // BLOCK_SIZE * BLOCK_SIZE
    blocks = [tokens[i:i + BLOCK_SIZE] for i in range(0, length, BLOCK_SIZE)]
    # Fixed-size blocks need no padding. Preserve EOS labels; GPT-2 shifts internally.
    return {"input_ids": blocks, "labels": [block.copy() for block in blocks]}


def main():
    os.environ["WANDB_PROJECT"] = "gpt2-training"
    # Initialize distributed execution so both GPUs use the same output directory.
    state = PartialState()
    run_names = [f"gpt2-{datetime.now():%Y%m%d-%H%M%S-%f}" if state.is_main_process else None]
    broadcast_object_list(run_names)
    run_name = run_names[0]

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained("gpt2")
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_config(config)

    train_dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True
    )

    eval_dataset = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True
    )

    train_dataset = train_dataset.shuffle(seed=42, buffer_size=10000)
    train_dataset = train_dataset.map(
        pack_texts,
        batched=True,
        batch_size=1000,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["text", "timestamp", "url"],
    )
    # Materialize a small fixed set of packed blocks for repeatable, bounded evaluation.
    eval_dataset = eval_dataset.map(
        pack_texts,
        batched=True,
        batch_size=1000,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["text", "timestamp", "url"],
    )
    with state.main_process_first():
        eval_dataset = Dataset.from_list(list(eval_dataset.take(1024)))

    training_args = TrainingArguments(
        output_dir=os.path.join(os.environ["GPT2_OUTPUT_ROOT"], run_name),
        run_name=run_name,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8,
        max_steps=1000,
        learning_rate=2.5e-4,
        lr_scheduler_type="cosine",
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
        eval_steps=250,
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        seed=42,
        torch_compile=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(f"{training_args.output_dir}/final")
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(f"{training_args.output_dir}/final")


if __name__ == "__main__":
    main()
