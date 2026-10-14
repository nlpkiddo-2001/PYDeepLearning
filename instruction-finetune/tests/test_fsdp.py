"""
Distributed training tests. All run on CPU with the gloo backend (no GPUs needed):
torchrun starts 2 processes, each with `--device cpu`.

1. FSDP grads == single-process grads over the same rows (tests/fsdp_grad_check.py)
2. train() runs every step it was asked for
3. the CLI works end to end: plain python (1 process) and torchrun (2 processes, sharded)

The CLI tests use a tiny random Qwen3 saved to disk, so load_model() runs for real
(from_pretrained -> to(device) -> shard_model) without downloading 16 GB.
"""
import math
import os
import subprocess
import sys
from collections import Counter

import pytest

from sft.data.chat_tokenizer import ChatTokenizer, TokenizerConfig
from sft.train.parallel import DistInfo
from sft.train.train import TrainConfig, plan_steps, steps_per_epoch, train
from tests.conftest import make_tiny_model

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    cfg = TrainConfig(data_path=FIXTURE, device="cpu", total_steps=3, warmup_ratio=0.34, log_every=1000)
    losses = train(cfg, model=make_tiny_model(len(tok)), tokenizer=tok)
    assert len(losses) == cfg.total_steps, f"train() stopped after {len(losses)} of {cfg.total_steps} steps"


def test_total_steps_from_epochs(tok):
    cfg = TrainConfig(data_path=FIXTURE, device="cpu", epochs=2.5, warmup_ratio=0.1, log_every=1000)
    chat_tok = ChatTokenizer(tok, TokenizerConfig(max_seq_len=cfg.max_seq_len))
    info = DistInfo(device="cpu")
    per_epoch = steps_per_epoch(cfg, chat_tok, info)
    total, warmup = plan_steps(cfg, chat_tok, info)
    assert total == math.ceil(2.5 * per_epoch)
    assert warmup == round(0.1 * total)

    losses = train(cfg, model=make_tiny_model(len(tok)), tokenizer=tok)
    assert len(losses) == total


def cli_args(model_dir):
    return [
        "-m", "sft.train.train",
        "--model_name", model_dir, "--data_path", FIXTURE, "--device", "cpu",
        "--total_steps", "3", "--warmup_ratio", "0.34", "--grad_accum_steps", "2",
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


def load_exported(path):
    model = AutoModelForCausalLM.from_pretrained(path)
    AutoTokenizer.from_pretrained(path)               # tokenizer was saved too
    return model


def check_file(save_dir):
    """the file itself: bf16, tied weight stored once, config says bf16 (what vLLM / dtype='auto' will use)"""
    import json
    from safetensors import safe_open

    with safe_open(os.path.join(save_dir, "model.safetensors"), "pt") as f:
        keys = set(f.keys())
        dtypes = {f.get_slice(k).get_dtype() for k in keys}
    assert dtypes == {"BF16"}, dtypes
    assert "lm_head.weight" not in keys, "tied weight stored twice"
    with open(os.path.join(save_dir, "config.json")) as f:
        config = json.load(f)
    assert config["dtype"] == "bfloat16"
    assert config["architectures"] == ["Qwen3ForCausalLM"], "FSDP's wrapper class leaked into config.json"


def check_export(save_dir, start_dir):
    trained = load_exported(save_dir)
    start = AutoModelForCausalLM.from_pretrained(start_dir)

    assert trained.lm_head.weight is trained.model.embed_tokens.weight, "tie lost in export"
    assert trained.config.tie_word_embeddings

    start_sd, trained_sd = start.state_dict(), trained.state_dict()
    assert start_sd.keys() == trained_sd.keys()
    for k in start_sd:
        assert start_sd[k].shape == trained_sd[k].shape, k

    # training ran (lr 1e-5, 3 steps): weights moved, but not by much
    w0 = start.model.layers[0].mlp.down_proj.weight.float()
    w1 = trained.model.layers[0].mlp.down_proj.weight.float()
    assert not torch.equal(w0, w1), "exported weights are the starting weights: export didn't get the trained model"
    assert (w1 - w0).norm() / w0.norm() < 0.1


def test_export_single_process(tiny_model_dir, tmp_path):
    run([sys.executable, *cli_args(tiny_model_dir), "--save_dir", str(tmp_path / "out")])
    check_export(tmp_path / "out", tiny_model_dir)


def test_export_torchrun_two_processes(tiny_model_dir, tmp_path):
    out = run(torchrun(2, *cli_args(tiny_model_dir), "--save_dir", str(tmp_path / "out")))
    assert out.count("saved HF checkpoint") == 1, "only rank 0 should write/log"
    check_file(tmp_path / "out")
    check_export(tmp_path / "out", tiny_model_dir)
