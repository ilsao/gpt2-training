"""Prepare a bounded, reusable C4 dataset before allocating training GPUs."""

import argparse
import json
import os
import shutil
import tempfile
from itertools import islice
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset, load_from_disk
from transformers import AutoConfig, AutoTokenizer

from checkpoints import atomic_write_json

BLOCK_SIZE = 1024
SEED = 42
SHUFFLE_BUFFER = 10000
DOCUMENT_BATCH_SIZE = 1000
DEFAULT_TRAIN_BLOCKS = 1_280_000
DEFAULT_EVAL_BLOCKS = 1024
METADATA = {
    "format_version": 1, "dataset": "allenai/c4", "dataset_config": "en",
    "tokenizer": "gpt2", "block_size": BLOCK_SIZE, "seed": SEED,
    "shuffle_buffer": SHUFFLE_BUFFER, "document_batch_size": DOCUMENT_BATCH_SIZE,
}


def pack_texts(examples, tokenizer):
    """Preserve EOS labels; discard only the partial block of each document batch."""
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
    return {"input_ids": blocks, "labels": [block.copy() for block in blocks]}


def generate_blocks(split, num_blocks, tokenizer_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
    if split == "train":
        dataset = dataset.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)
    documents = iter(dataset)
    count = 0
    while count < num_blocks:
        batch = list(islice(documents, DOCUMENT_BATCH_SIZE))
        if not batch:
            raise ValueError(f"C4 {split} exhausted after {count} blocks; requested {num_blocks}")
        packed = pack_texts({"text": [row["text"] for row in batch]}, tokenizer)
        for inputs, labels in zip(packed["input_ids"], packed["labels"]):
            yield {"input_ids": inputs, "labels": labels}
            count += 1
            if count == num_blocks:
                return


def prepare_dataset(output_dir, train_blocks=DEFAULT_TRAIN_BLOCKS, eval_blocks=DEFAULT_EVAL_BLOCKS):
    if train_blocks <= 0 or eval_blocks <= 0:
        raise ValueError("Train/evaluation block counts must be positive")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}; use a new directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(staging / "tokenizer")
        AutoConfig.from_pretrained("gpt2").save_pretrained(staging / "config")
        features = Features({
            name: Sequence(Value("int32"), length=BLOCK_SIZE)
            for name in ("input_ids", "labels")
        })
        splits = {}
        for split, count in (("train", train_blocks), ("validation", eval_blocks)):
            splits[split] = Dataset.from_generator(
                generate_blocks, features=features, keep_in_memory=False,
                cache_dir=str(staging / "cache"), split=split,
                gen_kwargs={
                    "split": split, "num_blocks": count,
                    "tokenizer_dir": str(staging / "tokenizer"),
                },
            )
        DatasetDict(splits).save_to_disk(str(staging / "dataset"))
        # The saved Arrow shards are independent of the generator cache.
        del splits
        shutil.rmtree(staging / "cache")
        atomic_write_json(staging / "metadata.json", {
            **METADATA, "train_blocks": train_blocks, "eval_blocks": eval_blocks,
            "eos_token_id": tokenizer.eos_token_id, "vocab_size": len(tokenizer),
        })
        validate_prepared_dataset(staging)
        os.rename(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Prepared {train_blocks} train / {eval_blocks} evaluation blocks at {output_dir}")


def validate_prepared_dataset(data_dir):
    """Only read fully published data, with the tokenizer/config used to pack it."""
    data_dir = Path(data_dir)
    with (data_dir / "metadata.json").open() as stream:
        metadata = json.load(stream)
    for name, expected in METADATA.items():
        if metadata.get(name) != expected:
            raise ValueError(f"Prepared dataset metadata mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(data_dir / "tokenizer", local_files_only=True)
    config = AutoConfig.from_pretrained(data_dir / "config", local_files_only=True)
    if (tokenizer.eos_token_id != metadata["eos_token_id"]
            or tokenizer.pad_token_id != tokenizer.eos_token_id
            or len(tokenizer) != metadata["vocab_size"]
            or config.model_type != "gpt2" or config.vocab_size != len(tokenizer)
            or config.n_positions < BLOCK_SIZE):
        raise ValueError("Prepared dataset tokenizer/config mismatch")
    dataset = load_from_disk(str(data_dir / "dataset"), keep_in_memory=False)
    expected_features = Features({
        name: Sequence(Value("int32"), length=BLOCK_SIZE)
        for name in ("input_ids", "labels")
    })
    if set(dataset) != {"train", "validation"}:
        raise ValueError("Prepared dataset must contain train and validation splits")
    for split, key in (("train", "train_blocks"), ("validation", "eval_blocks")):
        count = metadata[key]
        if type(count) is not int or count <= 0 or len(dataset[split]) != count:
            raise ValueError(f"Prepared dataset block count mismatch: {split}")
        if dataset[split].features != expected_features:
            raise ValueError(f"Prepared dataset block format mismatch: {split}")
        for index in {0, count - 1}:
            row = dataset[split][index]
            if row["input_ids"] != row["labels"]:
                raise ValueError(f"Prepared dataset labels mismatch: {split}")
    return dataset, tokenizer, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-blocks", type=int, default=DEFAULT_TRAIN_BLOCKS)
    parser.add_argument("--eval-blocks", type=int, default=DEFAULT_EVAL_BLOCKS)
    args = parser.parse_args()
    prepare_dataset(args.output_dir, args.train_blocks, args.eval_blocks)


if __name__ == "__main__":
    main()
