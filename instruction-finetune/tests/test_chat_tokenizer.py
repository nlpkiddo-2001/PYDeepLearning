"""Tests for sft.data.chat_tokenizer.

Run:  pytest tests/test_chat_tokenizer.py -v
Debug a failure:  pytest tests/test_chat_tokenizer.py -k <name> -s   (prints render_labels output)
"""

import copy

import pytest
from transformers import AutoTokenizer

from sft.data.chat_tokenizer import (
    IGNORE_INDEX,
    PROBE_RECORD,
    ChatTokenizer,
    RejectReason,
    TokenizerConfig,
    probe_template,
    render_labels,
)

QWEN3 = "Qwen/Qwen3-4B"
LLAMA = "unsloth/Llama-3.2-3B-Instruct"


@pytest.fixture(scope="module")
def qwen_tok():
    return AutoTokenizer.from_pretrained(QWEN3)


@pytest.fixture(scope="module")
def llama_tok():
    return AutoTokenizer.from_pretrained(LLAMA)


def make(tok, **cfg):
    return ChatTokenizer(tok, TokenizerConfig(check_roundtrip=True, **cfg))


def supervised_runs(sample, tok) -> list[str]:
    """Decoded text of each contiguous supervised region."""
    runs, cur = [], []
    for tid, lab in zip(sample.input_ids, sample.labels):
        if lab != IGNORE_INDEX:
            cur.append(tid)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [tok.decode(r, skip_special_tokens=False) for r in runs]


def assert_invariants(sample, tok):
    """Checks that must hold for every sample, for every model."""
    assert len(sample.input_ids) == len(sample.labels)
    for tid, lab in zip(sample.input_ids, sample.labels):
        assert lab in (IGNORE_INDEX, tid), "labels must be unshifted copies of input_ids"
    assert sample.num_supervised == sum(l != IGNORE_INDEX for l in sample.labels) > 0
    assert sample.meta["num_tokens"] == len(sample.input_ids)
    assert sample.meta["straddling_tokens"] == 0
    # Every supervised turn ends with the end-of-turn token, so the model learns to stop.
    for run in supervised_runs(sample, tok):
        assert run.endswith(tok.eos_token), f"supervised run does not end with eos: {run[-40:]!r}"
    # No user / tool / system text is ever supervised.
    for run in supervised_runs(sample, tok):
        for leaked in ("Weather in Paris and Rome?", "18C, sunny", "You are a helpful assistant"):
            assert leaked not in run, f"non-assistant text supervised: {leaked!r}"


# --------------------------------------------------------------------------- #
# Qwen3
# --------------------------------------------------------------------------- #


def test_probe_passes(qwen_tok):
    ct = make(qwen_tok)
    assert ct.adapter.name == "qwen3"
    assert probe_template(ct) == []


def test_single_mode_supervises_exactly_assistant_turns(qwen_tok):
    r = make(qwen_tok)(PROBE_RECORD)
    assert r.mode == "single" and not r.rejections and len(r.samples) == 1
    s = r.samples[0]
    print(render_labels(s, qwen_tok))
    assert_invariants(s, qwen_tok)
    runs = supervised_runs(s, qwen_tok)
    assert len(runs) == 2
    assert runs[0].startswith("<think>\nI need to call the tool")
    assert runs[0].count("<tool_call>") == 2
    assert runs[1] == (
        "<think>\nBoth results are in.\n</think>\n\n"
        "Paris is 18C and sunny; Rome is 24C and cloudy.<|im_end|>"
    )


def test_single_equals_per_turn(qwen_tok):
    single = make(qwen_tok)(PROBE_RECORD).samples[0]
    per_turn = make(qwen_tok, mode="per_turn")(PROBE_RECORD).samples
    sup = lambda s: [t for t, l in zip(s.input_ids, s.labels) if l != IGNORE_INDEX]
    assert sup(single) == [t for s in per_turn for t in sup(s)]


def test_prompt_matches_inference(qwen_tok):
    ct = make(qwen_tok, mode="per_turn")
    msgs, tools = ct.adapter.prepare(PROBE_RECORD["messages"], PROBE_RECORD["tools"])
    for s in ct(PROBE_RECORD).samples:
        assert_invariants(s, qwen_tok)
        ref = qwen_tok.apply_chat_template(
            msgs[: s.meta["turn_idx"]], tools=tools, add_generation_prompt=True,
            tokenize=True, return_dict=False, **ct.adapter.template_kwargs(msgs),
        )
        first_sup = s.labels.index(next(l for l in s.labels if l != IGNORE_INDEX))
        assert s.input_ids[:first_sup] == list(ref)


def test_no_reasoning_falls_back_and_keeps_empty_think_in_prompt(qwen_tok):
    rec = copy.deepcopy(PROBE_RECORD)
    for m in rec["messages"]:
        m.pop("reasoning_content", None)
    r = make(qwen_tok)(rec)
    assert r.mode == "per_turn" and r.fallback_reason and len(r.samples) == 2
    for s in r.samples:
        print(render_labels(s, qwen_tok))
        assert_invariants(s, qwen_tok)
        assert "<think>" not in "".join(supervised_runs(s, qwen_tok))


def test_second_user_query_falls_back(qwen_tok):
    rec = copy.deepcopy(PROBE_RECORD)
    rec["messages"] += [
        {"role": "user", "content": "And Berlin?"},
        {"role": "assistant", "reasoning_content": "No tool needed.", "content": "Probably cooler."},
    ]
    r = make(qwen_tok)(rec)
    assert r.mode == "per_turn" and len(r.samples) == 3
    last = r.samples[-1]
    assert_invariants(last, qwen_tok)
    # Earlier reasoning is stripped from the context, exactly as at inference.
    context = qwen_tok.decode(last.input_ids)
    assert "I need to call the tool" not in context


def test_train_false_turn_not_supervised(qwen_tok):
    rec = copy.deepcopy(PROBE_RECORD)
    rec["messages"][2]["train"] = False
    s = make(qwen_tok)(rec).samples[0]
    assert_invariants(s, qwen_tok)
    runs = supervised_runs(s, qwen_tok)
    assert len(runs) == 1 and "Both results are in" in runs[0]


@pytest.mark.parametrize(
    "max_seq_len, n_samples", [(10_000, 1), (260, 1), (50, 0)]
)
def test_overlong(qwen_tok, max_seq_len, n_samples):
    r = make(qwen_tok, max_seq_len=max_seq_len)(PROBE_RECORD)
    assert len(r.samples) == n_samples
    for s in r.samples:
        assert len(s.input_ids) <= max_seq_len
    if n_samples == 0:
        assert {x.reason for x in r.rejections} == {RejectReason.OVERLONG}


def test_overlong_single_mode_rejects(qwen_tok):
    r = make(qwen_tok, max_seq_len=50, mode="single")(PROBE_RECORD)
    assert not r.samples and r.rejections[0].reason == RejectReason.OVERLONG


def test_truncate_keeps_partial_supervision(qwen_tok):
    r = make(qwen_tok, max_seq_len=230, mode="single", overlong="truncate")(PROBE_RECORD)
    assert len(r.samples) == 1 and len(r.samples[0].input_ids) == 230


def test_no_train_turns_rejected(qwen_tok):
    rec = copy.deepcopy(PROBE_RECORD)
    for m in rec["messages"]:
        m["train"] = False
    r = make(qwen_tok)(rec)
    assert not r.samples and r.rejections[0].reason == RejectReason.NO_TRAIN_TURNS


def test_bad_record_never_raises(qwen_tok):
    for rec in (
        {"id": "a", "messages": [{"role": "user"}, {"role": "assistant", "content": "hi"}]},
        {"id": "b", "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": 5}]},
        {"id": "c", "messages": [{"role": "assistant", "content": "hi", "tool_calls": "oops"}]},
    ):
        r = make(qwen_tok)(rec)
        assert r.samples or r.rejections


def test_input_record_not_mutated(qwen_tok):
    rec = copy.deepcopy(PROBE_RECORD)
    make(qwen_tok)(rec)
    assert rec == PROBE_RECORD


# --------------------------------------------------------------------------- #
# Generic adapter (Llama-3.2): BOS handling, template limits
# --------------------------------------------------------------------------- #


def single_call_record():
    rec = copy.deepcopy(PROBE_RECORD)
    rec["messages"][2]["tool_calls"] = rec["messages"][2]["tool_calls"][:1]
    del rec["messages"][4]
    return rec


def test_llama_single_bos(llama_tok):
    r = make(llama_tok)(single_call_record())
    s = r.samples[0]
    assert s.input_ids[0] == llama_tok.bos_token_id
    assert s.input_ids.count(llama_tok.bos_token_id) == 1


def test_llama_probe_reports_parallel_calls_unsupported(llama_tok):
    problems = probe_template(make(llama_tok))
    assert problems and "single tool-calls" in problems[0]
