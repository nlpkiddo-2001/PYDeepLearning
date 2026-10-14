"""
Distributed training tests. All run on CPU with the gloo backend (no GPUs needed):
torchrun starts 2 processes, each with `--device cpu`.

1. FSDP grads == single-process grads over the same rows (tests/fsdp_grad_check.py)
2. train() runs every step it was asked for
3. the CLI works end to end: plain python (1 process) and torchrun (2 processes, sharded)

The CLI tests use a tiny random Qwen3 saved to disk, so load_model() runs for real
(from_pretrained -> to(device) -> shard_model) without downloading 16 GB.
"""
import os
import subprocess
import sys
from collections import Counter

import pytest

from sft.train.train import TrainConfig, train
from tests.conftest import make_tiny_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests/fixtures/sample.jsonl")
CPU_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "CHECK_DEVICE": "cpu"}


def run(cmd, timeout=600):
    """run a command from the repo root; print its output on failure so pytest shows it"""
    result = subprocess.run(cmd, cwd=ROOT, env=CPU_ENV, capture_output=True, text=True, timeout=timeout)
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"exit code {result.returncode}\n{output[-4000:]}"
    return output


def torchrun(nproc, *args):
    # --standalone: torchrun picks a free port itself, so tests can't collide on 29500
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", *args]


@pytest.fixture(scope="module")
def tiny_model_dir(tok, tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_qwen3")
    make_tiny_model(len(tok)).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def test_fsdp_grads_match_single_process():
    out = run(torchrun(2, "-m", "tests.fsdp_grad_check"))
    assert "grads_match=True" in out


def test_train_runs_all_steps(tok):
    cfg = TrainConfig(data_path=FIXTURE, device="cpu", total_steps=3, warmup_steps=1, log_every=1000)
    losses = train(cfg, model=make_tiny_model(len(tok)), tokenizer=tok)
    assert len(losses) == cfg.total_steps, f"train() stopped after {len(losses)} of {cfg.total_steps} steps"


def cli_args(model_dir):
    return [
        "-m", "sft.train.train",
        "--model_name", model_dir, "--data_path", FIXTURE, "--device", "cpu",
        "--total_steps", "3", "--warmup_steps", "1", "--grad_accum_steps", "2",
    ]


def test_cli_single_process(tiny_model_dir):
    out = run([sys.executable, *cli_args(tiny_model_dir)])
    assert "step    2" in out, out[-2000:]
    assert "rows 2 " in out          # 1 process x grad_accum 2


def test_cli_torchrun_two_processes(tiny_model_dir):
    out = run(torchrun(2, *cli_args(tiny_model_dir)))
    assert "step    2" in out, out[-2000:]
    assert "rows 4 " in out          # 2 ranks x grad_accum 2
    assert out.count("step    0") == 1, "only rank 0 should log"
