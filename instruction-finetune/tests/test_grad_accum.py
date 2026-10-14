"""
Gradient accumulation must give the same gradient as one big batch.

Correct:  sum_i loss_sum_i / N_total          (every token weighted 1 / N_total)
Wrong:    mean_i (loss_sum_i / N_i)           (tokens in short rows weighted more)
"""
from collections import Counter

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from sft.data.chat_tokenizer import ChatTokenizer, TokenizerConfig
from sft.train.train import collate, compute_loss_sum, row_stream

FIXTURE = "tests/fixtures/sample.jsonl"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")


def tiny_model(vocab_size):
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
    )
    cfg._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(cfg).float()
    model.config.use_cache = False
    return model


def two_rows_with_different_counts(tok):
    chat_tok = ChatTokenizer(tok, TokenizerConfig(max_seq_len=512))
    rows = row_stream(FIXTURE, chat_tok, 512, Counter())
    a = next(rows)
    for b in rows:
        if b.num_supervised != a.num_supervised:
            return [a, b]


def grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def test_accum_matches_single_backward(tok):
    model = tiny_model(len(tok))
    rows = two_rows_with_different_counts(tok)
    n_total = sum(r.num_supervised for r in rows)

    # reference: one backward over the whole step
    total = sum(compute_loss_sum(model, collate(r, "cpu"), "cpu") for r in rows)
    (total / n_total).backward()
    g_ref = grads(model).clone()
    model.zero_grad(set_to_none=True)

    # grad accumulation: one backward per micro-batch, same global denominator
    for r in rows:
        (compute_loss_sum(model, collate(r, "cpu"), "cpu") / n_total).backward()
    g_acc = grads(model).clone()
    model.zero_grad(set_to_none=True)

    # the wrong way: per-micro-batch mean, then average
    for r in rows:
        (compute_loss_sum(model, collate(r, "cpu"), "cpu") / r.num_supervised / len(rows)).backward()
    g_wrong = grads(model)

    assert torch.allclose(g_acc, g_ref, rtol=1e-4, atol=1e-6)
    assert not torch.allclose(g_wrong, g_ref, rtol=1e-3, atol=1e-6)
