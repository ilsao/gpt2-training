import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from datasets import IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Config, TrainerControl, TrainerState
from transformers.activations import ACT2FN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distributed_smoke import tiny_tokenizer

import prepare_dataset
from checkpoints import atomic_write_json, latest_complete_checkpoint, mark_checkpoint_complete
from prepare_dataset import BLOCK_SIZE, pack_texts, validate_prepared_dataset
from push_to_hub import upload_checkpoint
from train import ShutdownCallback, configure_model

ROOT = Path(__file__).resolve().parents[1]


def documents():
    for _ in range(8):
        yield {"text": "word " * 700}


def make_checkpoint(run_dir, step, world_size=1):
    path = Path(run_dir) / f"checkpoint-{step}"
    path.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors",
                 "training_args.bin", "optimizer.pt", "scheduler.pt"):
        (path / name).write_bytes(b"artifact")
    rng_names = ["rng_state.pth"] if world_size == 1 else [f"rng_state_{rank}.pth" for rank in range(world_size)]
    for name in rng_names:
        (path / name).write_bytes(b"rng")
    atomic_write_json(path / "trainer_state.json", {"global_step": step})
    mark_checkpoint_complete(path, step, world_size, "periodic")
    return path


class DataAndModelTests(unittest.TestCase):
    def test_eos_packing_and_copied_labels(self):
        tokenizer = tiny_tokenizer()
        packed = pack_texts({"text": ["word " * 700, "other " * 700]}, tokenizer)
        self.assertEqual(len(packed["input_ids"]), 1)
        block = packed["input_ids"][0]
        self.assertEqual(len(block), BLOCK_SIZE)
        self.assertEqual(block[700], tokenizer.eos_token_id)
        self.assertEqual(block, packed["labels"][0])
        self.assertIsNot(block, packed["labels"][0])

    def test_preparation_disk_roundtrip_and_metadata_rejection(self):
        tokenizer = tiny_tokenizer()
        config = GPT2Config(vocab_size=len(tokenizer), n_positions=1024, n_embd=8, n_layer=1, n_head=2,
                            bos_token_id=1, eos_token_id=1)
        real_tokenizer_loader = AutoTokenizer.from_pretrained
        real_config_loader = AutoConfig.from_pretrained
        def tokenizer_loader(name, *args, **kwargs):
            return tokenizer if name == "gpt2" else real_tokenizer_loader(name, *args, **kwargs)
        def config_loader(name, *args, **kwargs):
            return config if name == "gpt2" else real_config_loader(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared"
            with patch.object(prepare_dataset.AutoTokenizer, "from_pretrained", side_effect=tokenizer_loader), \
                 patch.object(prepare_dataset.AutoConfig, "from_pretrained", side_effect=config_loader), \
                 patch("datasets.config.HF_DATASETS_CACHE", Path(directory) / "hf-cache"), \
                 patch.object(prepare_dataset, "load_dataset", side_effect=lambda *a, **k: IterableDataset.from_generator(documents)):
                prepare_dataset.prepare_dataset(path, train_blocks=3, eval_blocks=2)
            dataset, loaded_tokenizer, _ = validate_prepared_dataset(path)
            self.assertEqual(len(dataset["train"]), 3)
            self.assertEqual(len(dataset["validation"]), 2)
            self.assertEqual(dataset["train"][0]["input_ids"], dataset["train"][0]["labels"])
            self.assertEqual(loaded_tokenizer.pad_token_id, loaded_tokenizer.eos_token_id)
            self.assertFalse((path / "cache").exists())
            with self.assertRaises(FileExistsError):
                prepare_dataset.prepare_dataset(path, 3, 2)
            metadata = json.loads((path / "metadata.json").read_text())
            for key, bad_value in (("block_size", 512), ("train_blocks", 4), ("eos_token_id", 3)):
                atomic_write_json(path / "metadata.json", {**metadata, key: bad_value})
                with self.assertRaises(ValueError):
                    validate_prepared_dataset(path)
            (path / "metadata.json").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_prepared_dataset(path)

    def test_preparation_failure_never_publishes_directory(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(prepare_dataset.AutoTokenizer, "from_pretrained", side_effect=RuntimeError("download failed")):
            path = Path(directory) / "prepared"
            with self.assertRaises(RuntimeError):
                prepare_dataset.prepare_dataset(path, 3, 2)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_model_uses_zero_dropout_and_native_tanh_gelu(self):
        tokenizer = tiny_tokenizer()
        config = configure_model(GPT2Config(vocab_size=4, n_layer=1, n_head=2, n_embd=8,
                                           bos_token_id=1, eos_token_id=1), tokenizer)
        model = AutoModelForCausalLM.from_config(config)
        self.assertTrue(all(module.p == 0 for module in model.modules() if isinstance(module, torch.nn.Dropout)))
        self.assertEqual((config.resid_pdrop, config.embd_pdrop, config.attn_pdrop), (0, 0, 0))
        activation = model.transformer.h[0].mlp.act
        self.assertIsInstance(activation, type(ACT2FN["gelu_pytorch_tanh"]))
        values = torch.linspace(-4, 4, 100)
        torch.testing.assert_close(activation(values), torch.nn.functional.gelu(values, approximate="tanh"))


class CheckpointAndUploadTests(unittest.TestCase):
    def test_incomplete_or_damaged_checkpoint_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            first = make_checkpoint(directory, 1, world_size=2)
            latest = make_checkpoint(directory, 3, world_size=2)
            (latest / "rng_state_1.pth").unlink()
            incomplete = Path(directory) / "checkpoint-4"
            incomplete.mkdir()
            (incomplete / "model.safetensors").write_bytes(b"partial")
            self.assertEqual(latest_complete_checkpoint(directory)[0], first)
            shutil.rmtree(first)
            with self.assertRaisesRegex(ValueError, "No complete checkpoint"):
                latest_complete_checkpoint(directory)

    def test_hub_upload_retries_and_records_success(self):
        with tempfile.TemporaryDirectory() as directory, patch("push_to_hub.time.sleep") as sleep:
            path = make_checkpoint(directory, 3)
            (Path(directory) / "checkpoint-4").mkdir()
            api = Mock()
            commit = SimpleNamespace(oid="commit", commit_url="https://huggingface.co/test/model/commit/commit")
            api.upload_folder.side_effect = [TimeoutError("temporary"), commit]
            self.assertIs(upload_checkpoint(directory, "test/model", api), commit)
            self.assertEqual(api.upload_folder.call_count, 2)
            sleep.assert_called_once_with(10)
            options = api.upload_folder.call_args.kwargs
            self.assertEqual(options["folder_path"], str(path))
            self.assertEqual(options["path_in_repo"], "")
            self.assertFalse(options["run_as_future"])
            self.assertIn("optimizer.pt", options["allow_patterns"])
            self.assertEqual(json.loads((Path(directory) / "upload-result.json").read_text())["global_step"], 3)

    def test_hub_permanent_and_exhausted_failures_remain_failures(self):
        with tempfile.TemporaryDirectory() as directory, patch("push_to_hub.time.sleep") as sleep:
            path = make_checkpoint(directory, 1)
            api = Mock()
            error = RuntimeError("authentication denied")
            error.response = SimpleNamespace(status_code=403)
            api.upload_folder.side_effect = error
            with self.assertRaises(RuntimeError):
                upload_checkpoint(directory, "test/model", api)
            self.assertEqual(api.upload_folder.call_count, 1)
            api.reset_mock()
            api.upload_folder.side_effect = TimeoutError("offline")
            with self.assertRaises(TimeoutError):
                upload_checkpoint(directory, "test/model", api)
            self.assertEqual(api.upload_folder.call_count, 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [10, 30])
            self.assertFalse((Path(directory) / "upload-result.json").exists())
            self.assertTrue(path.exists())

    def test_deadline_and_elapsed_save_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(device=torch.device("cpu"))
            state = TrainerState(global_step=2)
            shutdown = ShutdownCallback(Path(directory) / "stop", deadline=1000)
            with patch("train.time.time", return_value=821):
                control = shutdown.on_step_end(args, state, TrainerControl(should_evaluate=True))
            self.assertTrue(control.should_training_stop)
            self.assertTrue(control.should_save)
            self.assertFalse(control.should_evaluate)
            shutdown = ShutdownCallback(Path(directory) / "stop")
            shutdown.last_save = 0
            with patch("train.time.monotonic", return_value=301):
                control = shutdown.on_step_end(args, state, TrainerControl())
            self.assertTrue(control.should_save)
            self.assertFalse(control.should_training_stop)

    def test_real_two_rank_shutdown_resume_and_killed_save(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "1",
                   "TOKENIZERS_PARALLELISM": "false"}
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
                       "--master_addr=127.0.0.1", f"--master_port={port}",
                       str(ROOT / "tests/distributed_smoke.py"), "--run-dir", directory]
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("DISTRIBUTED_SAVE_RESUME_OK", result.stdout)
            result = subprocess.run(command + ["--interrupt-save"], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=90)
            self.assertNotEqual(result.returncode, 0, "Expected an interrupted save")
            self.assertTrue((Path(directory) / "checkpoint-5/model.safetensors").exists(), result.stdout + result.stderr)
            self.assertFalse((Path(directory) / "checkpoint-5/checkpoint-complete.json").exists())
            self.assertEqual(latest_complete_checkpoint(directory)[1]["global_step"], 4)


MOCK_COMMAND = '''import json, os, sys, time
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ["MOCK_LOG"], "a") as stream:
    stream.write(json.dumps([name, *sys.argv[1:]]) + "\\n")
if name == "python" and os.environ.get("MOCK_PREFLIGHT_FAIL"):
    sys.exit(1)
if name == "sbatch":
    if "train.sbatch" in sys.argv:
        print("123;cluster")
    elif os.environ.get("MOCK_UPLOAD_FAIL"):
        sys.exit(1)
    else:
        marker = Path(sys.argv[-1]) / "upload-registered"
        if marker.exists():
            sys.exit(2)
        if os.environ.get("MOCK_MARKER_FAIL"):
            marker.mkdir(parents=True)
        print("456")
if name == "torchrun":
    Path(os.environ["MOCK_READY"]).touch()
    stop = Path(os.environ["GPT2_OUTPUT_ROOT"]) / "gpt2-123/stop-requested"
    while not stop.exists():
        time.sleep(0.01)
    time.sleep(0.05)
    sys.exit(7)
'''


class SlurmTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("submit.sh", "job_env.sh", "train.sbatch", "upload.sbatch"):
            shutil.copy(ROOT / name, self.root / name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name in ("python", "module", "sbatch", "scontrol", "scancel", "torchrun", "nvidia-smi"):
            path = bin_dir / name
            path.write_text(f"#!{sys.executable}\n" + MOCK_COMMAND)
            path.chmod(0o755)
        (self.root / ".venv/bin").mkdir(parents=True)
        (self.root / ".venv/bin/activate").write_text(":\n")
        self.log = self.root / "commands.jsonl"
        self.env = {
            **os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "MOCK_LOG": str(self.log),
            "NANO4_WORK_DIR": str(self.root / "work"), "GPT2_DATA_DIR": str(self.root / "data"),
            "GPT2_OUTPUT_ROOT": str(self.root / "outputs"), "HF_HOME": str(self.root / "hf"),
            "SLURM_SUBMIT_DIR": str(self.root), "SLURM_JOB_ID": "123",
            "MOCK_READY": str(self.root / "ready"),
        }

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def submit(self, **extra):
        return subprocess.run(["bash", "submit.sh"], cwd=self.root, env={**self.env, **extra},
                              capture_output=True, text=True, timeout=10)

    def test_dependency_registered_before_ready_marker(self):
        result = self.submit()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [row for row in self.calls() if row[0] in ("sbatch", "scontrol", "scancel")]
        self.assertEqual(calls, [
            ["sbatch", "--parsable", "train.sbatch", "--wait-for-upload"],
            ["sbatch", "--parsable", "--dependency=afterany:123", "upload.sbatch", str(self.root / "outputs/gpt2-123")],
        ])
        self.assertEqual((self.root / "outputs/gpt2-123/upload-registered").read_text(), "456\n")

    def test_upload_registration_failure_cancels_waiting_training(self):
        result = self.submit(MOCK_UPLOAD_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(["scancel", "123"], self.calls())
        self.assertFalse(any(row[0] == "scontrol" for row in self.calls()))
        self.assertFalse((self.root / "outputs/gpt2-123/upload-registered").exists())

    def test_marker_publication_failure_cancels_both_jobs(self):
        result = self.submit(MOCK_MARKER_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls()[-2:], [["scancel", "456"], ["scancel", "123"]])

    def test_invalid_data_never_submits_jobs(self):
        result = self.submit(MOCK_PREFLIGHT_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(row[0] == "sbatch" for row in self.calls()))

    def test_batch_usr1_waits_for_real_training_exit_status(self):
        run_dir = self.root / "outputs/gpt2-123"
        run_dir.mkdir(parents=True)
        (run_dir / "upload-registered").write_text("456\n")
        process = subprocess.Popen(["bash", "train.sbatch", "--wait-for-upload"], cwd=self.root, env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            end = time.monotonic() + 5
            while not (self.root / "ready").exists() and process.poll() is None and time.monotonic() < end:
                time.sleep(0.01)
            self.assertTrue((self.root / "ready").exists())
            os.kill(process.pid, signal.SIGUSR1)
            output, errors = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 7, output + errors)
            self.assertTrue((self.root / "outputs/gpt2-123/stop-requested").exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


if __name__ == "__main__":
    unittest.main()
