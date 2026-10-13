"""
End-to-end: the real train() loop must memorize 32 tiny samples.
If this fails, something in data -> loss -> optimizer -> scheduler is broken.

Slow on CPU (~2 min), so it's opt-in:  RUN_SLOW=1 python -m pytest tests/test_overfit.py
"""
import json
import os
import random

import pytest

from sft.train.train import TrainConfig, train
from tests.conftest import make_tiny_model

WORDS = ["apple", "river", "copper", "violet", "tiger", "marble",
         "cactus", "harbor", "lemon", "falcon", "cobalt", "maple"]


def write_records(path, n=32, seed=0):
    rng = random.Random(seed)
    with open(path, "w") as f:
        for i in range(n):
            answer = " ".join(rng.choice(WORDS) for _ in range(6))  # random -> must be memorized, can't be inferred
            f.write(json.dumps({
                "id": f"overfit-{i}",
                "metadata": {"source": "test", "task_type": "test"},
                "messages": [
                    {"role": "user", "content": f"Secret {i}?"},
                    {"role": "assistant", "content": answer},
                ],
            }) + "\n")


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")
def test_overfit_32_samples(tok, tmp_path):
    data = tmp_path / "overfit.jsonl"
    write_records(data)

    model = make_tiny_model(len(tok), hidden=128)
    model.train()
    cfg = TrainConfig(
        data_path=str(data), device="cpu", max_seq_len=2048,   # all 32 samples fit in one row
        lr=3e-3, warmup_steps=5, total_steps=100, weight_decay=0.0, log_every=1000,
    )
    losses = train(cfg, model=model, tokenizer=tok)

    assert losses[0] > 10          # random init: ~ln(151936) = 11.9
    assert losses[-1] < 0.1
