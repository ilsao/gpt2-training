"""Shared checkpoint completion protocol; no training or Hub dependencies."""

import json
import os
import tempfile
from pathlib import Path

COMPLETE_MARKER = "checkpoint-complete.json"


def atomic_write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def required_files(checkpoint_dir, world_size):
    checkpoint_dir = Path(checkpoint_dir)
    files = {
        "config.json", "tokenizer_config.json", "trainer_state.json",
        "training_args.bin", "optimizer.pt", "scheduler.pt",
    }
    if (checkpoint_dir / "tokenizer.json").is_file():
        files.add("tokenizer.json")
    else:
        files.update(("vocab.json", "merges.txt"))

    for weights in ("model.safetensors", "pytorch_model.bin"):
        if (checkpoint_dir / weights).is_file():
            files.add(weights)
            break
    else:
        for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if (checkpoint_dir / index).is_file():
                files.add(index)
                with (checkpoint_dir / index).open() as stream:
                    files.update(json.load(stream)["weight_map"].values())
                break
        else:
            raise ValueError(f"No model weights in {checkpoint_dir}")

    if world_size == 1:
        files.add("rng_state.pth")
    else:
        files.update(f"rng_state_{rank}.pth" for rank in range(world_size))
    return files


def mark_checkpoint_complete(checkpoint_dir, global_step, world_size, reason):
    checkpoint_dir = Path(checkpoint_dir)
    required = required_files(checkpoint_dir, world_size)
    # Include every serialized artifact, including optional generation/scaler state.
    files = {
        str(path.relative_to(checkpoint_dir)): path.stat().st_size
        for path in checkpoint_dir.rglob("*")
        if path.is_file() and path.name != COMPLETE_MARKER
    }
    manifest = {
        "format_version": 1, "global_step": global_step,
        "world_size": world_size, "reason": reason, "files": files,
    }
    _validate_manifest(checkpoint_dir, manifest, required)
    atomic_write_json(checkpoint_dir / COMPLETE_MARKER, manifest)
    return manifest


def _validate_manifest(checkpoint_dir, manifest, required=None):
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint manifest")
    step, world_size = manifest["global_step"], manifest["world_size"]
    if type(step) is not int or step < 0 or type(world_size) is not int or world_size < 1:
        raise ValueError("Invalid checkpoint step/world size")
    if checkpoint_dir.name != f"checkpoint-{step}":
        raise ValueError("Checkpoint directory and global_step disagree")
    files = manifest["files"]
    required = required if required is not None else required_files(checkpoint_dir, world_size)
    if not required.issubset(files):
        raise ValueError(f"Missing checkpoint artifacts: {sorted(required - files.keys())}")
    for name, size in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid checkpoint artifact path: {name}")
        artifact = checkpoint_dir / relative
        if type(size) is not int or size <= 0 or not artifact.is_file() or artifact.stat().st_size != size:
            raise ValueError(f"Missing or incomplete checkpoint artifact: {artifact}")
    with (checkpoint_dir / "trainer_state.json").open() as stream:
        if json.load(stream)["global_step"] != step:
            raise ValueError("Trainer state and checkpoint manifest disagree")
    return manifest


def read_complete_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / COMPLETE_MARKER).open() as stream:
        return _validate_manifest(checkpoint_dir, json.load(stream))


def complete_checkpoints(run_dir):
    """Return only validated, committed checkpoints in ascending step order."""
    checkpoints = []
    for path in Path(run_dir).glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            manifest = read_complete_checkpoint(path)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        checkpoints.append((path, manifest))
    return sorted(checkpoints, key=lambda item: item[1]["global_step"])


def latest_complete_checkpoint(run_dir):
    checkpoints = complete_checkpoints(run_dir)
    if not checkpoints:
        raise ValueError(f"No complete checkpoint available in {run_dir}")
    return checkpoints[-1]
