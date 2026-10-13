"""
Packing must be invisible to the model.

1. packed loss (and grads) == sum of the losses of each sample run alone
2. changing sample A's tokens doesn't change sample B's logits in the same row
3. negative control: with use_cache=True or an attention_mask, B DOES see A.
   This proves test 2 can catch a leak, and pins the handoff gotcha.
   If a future transformers version stops leaking here, update this test.
"""
import json

import pytest
import torch

from sft.data.chat_tokenizer import ChatTokenizer, TokenizerConfig
from sft.data.packing import make_row
from sft.train.train import collate, compute_loss_sum
from tests.conftest import make_tiny_model

FIXTURE = "tests/fixtures/sample.jsonl"
BACKENDS = ["sdpa", "eager", "flex_attention"]


@pytest.fixture(scope="module")
def samples(tok):
    chat_tok = ChatTokenizer(tok, TokenizerConfig(max_seq_len=4096))
    out = []
    with open(FIXTURE) as f:
        for line in f:
            out.extend(chat_tok(json.loads(line)).samples)
    assert len(out) >= 2
    return out


def grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


@pytest.mark.parametrize("attn", BACKENDS)
def test_packed_loss_and_grads_match_unpacked(tok, samples, attn):
    # fp32 on purpose (autocast=False): this tests the packing logic, not bf16 rounding.
    # under bf16 the grads differ by ~2e-3 relative, which would hide a real leak of that size.
    model = make_tiny_model(len(tok), attn)
    check_grads = attn != "flex_attention"     # flex has no backward on CPU
    torch.set_grad_enabled(check_grads)

    try:
        packed = compute_loss_sum(model, collate(make_row(samples), "cpu"), "cpu", autocast=False)
        if check_grads:
            packed.backward()
            g_packed = grads(model).clone()
            model.zero_grad(set_to_none=True)

        unpacked = sum(
            compute_loss_sum(model, collate(make_row([s]), "cpu"), "cpu", autocast=False) for s in samples
        )
        torch.testing.assert_close(packed, unpacked, rtol=1e-5, atol=1e-4)

        if check_grads:
            unpacked.backward()
            torch.testing.assert_close(g_packed, grads(model), rtol=1e-4, atol=1e-5)
    finally:
        torch.set_grad_enabled(True)


def _b_logits_diff(model, a, b, **forward_kwargs):
    """max change in B's logits when every token of A is replaced"""
    row = make_row([a, b])
    n_a = len(a.input_ids)
    ids = torch.tensor([row.input_ids])
    pos = torch.tensor([row.position_ids])
    ids_changed = ids.clone()
    ids_changed[0, :n_a] = (ids_changed[0, :n_a] + 7) % 1000

    with torch.no_grad():
        l1 = model(input_ids=ids, position_ids=pos, **forward_kwargs).logits[0, n_a:]
        l2 = model(input_ids=ids_changed, position_ids=pos, **forward_kwargs).logits[0, n_a:]
    return (l1 - l2).abs().max().item()


@pytest.mark.parametrize("attn", BACKENDS)
def test_no_cross_sample_attention(tok, samples, attn):
    model = make_tiny_model(len(tok), attn).eval()
    assert _b_logits_diff(model, samples[0], samples[1], use_cache=False) < 1e-5


@pytest.mark.parametrize("attn", BACKENDS)
@pytest.mark.parametrize(
    "forward_kwargs",
    [{"use_cache": True}, {"use_cache": False, "attention_mask": "ones"}],
    ids=["use_cache", "attention_mask"],
)
def test_leak_is_detected(tok, samples, attn, forward_kwargs):
    model = make_tiny_model(len(tok), attn).eval()
    kwargs = dict(forward_kwargs)
    if kwargs.get("attention_mask") == "ones":
        n = len(samples[0].input_ids) + len(samples[1].input_ids)
        kwargs["attention_mask"] = torch.ones(1, n, dtype=torch.long)
    assert _b_logits_diff(model, samples[0], samples[1], **kwargs) > 1e-3
