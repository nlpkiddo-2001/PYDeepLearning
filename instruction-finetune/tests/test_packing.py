"""Tests for sft.data.packing.

Run:  pytest tests/test_packing.py -v
"""

import random

import pytest

from sft.data.chat_tokenizer import IGNORE_INDEX, TokenizedSample
from sft.data.packing import make_row, pack_greedy


def fake(i: int, n: int) -> TokenizedSample:
    """n-token sample whose first label is -100 and the rest supervised."""
    return TokenizedSample(list(range(n)), [IGNORE_INDEX] + list(range(1, n)), n - 1, {"id": f"s{i}"})


def assert_row_invariants(row, pack_len):
    assert len(row.input_ids) == len(row.labels) == len(row.position_ids) == sum(row.seq_lens)
    assert len(row.input_ids) <= pack_len
    assert row.num_supervised == sum(l != IGNORE_INDEX for l in row.labels)
    start = 0
    for n in row.seq_lens:
        assert row.position_ids[start:start + n] == list(range(n)), "position_ids must restart at 0"
        assert row.labels[start] == IGNORE_INDEX, "first label of every sample must be -100"
        start += n


# --------------------------------------------------------------------------- #
# make_row
# --------------------------------------------------------------------------- #


def test_make_row_concatenates_and_masks_boundaries():
    a = TokenizedSample([10, 11, 12], [-100, 11, 12], 2, {"id": "a"})
    b = TokenizedSample([20, 21], [20, 21], 2, {"id": "b"})  # first label supervised on purpose
    row = make_row([a, b])
    assert row.input_ids == [10, 11, 12, 20, 21]
    assert row.labels == [-100, 11, 12, -100, 21]
    assert row.position_ids == [0, 1, 2, 0, 1]
    assert row.seq_lens == [3, 2]
    assert row.num_supervised == 3  # recounted: b lost its first supervised label
    assert row.sample_ids == ["a", "b"]


def test_make_row_does_not_mutate_samples():
    b = TokenizedSample([20, 21], [20, 21], 2, {"id": "b"})
    make_row([b])
    assert b.labels == [20, 21]


# --------------------------------------------------------------------------- #
# pack_greedy
# --------------------------------------------------------------------------- #


def test_greedy_small_exact():
    rows = list(pack_greedy([fake(0, 4), fake(1, 3), fake(2, 5), fake(3, 2)], pack_len=8))
    assert [r.seq_lens for r in rows] == [[4, 3], [5, 2]]


def test_greedy_exact_fit_fills_row():
    rows = list(pack_greedy([fake(0, 5), fake(1, 3), fake(2, 8)], pack_len=8))
    assert [r.seq_lens for r in rows] == [[5, 3], [8]]


def test_greedy_realistic_lengths():
    random.seed(0)
    pack_len = 32768
    lens = [min(pack_len, max(64, int(random.lognormvariate(7.5, 1.2)))) for _ in range(5000)]
    rows = list(pack_greedy((fake(i, n) for i, n in enumerate(lens)), pack_len=pack_len))
    for row in rows:
        assert_row_invariants(row, pack_len)
    # every sample kept exactly once, in arrival order
    assert [sid for r in rows for sid in r.sample_ids] == [f"s{i}" for i in range(len(lens))]
    fill = sum(lens) / (len(rows) * pack_len)
    assert fill > 0.75, f"greedy fill unexpectedly low: {fill:.1%}"


def test_greedy_is_lazy_on_endless_stream():
    def endless():
        i = 0
        while True:
            yield fake(i, 3)
            i += 1

    rows = pack_greedy(endless(), pack_len=8)
    first = next(rows)  # must not try to read the whole stream
    assert first.seq_lens == [3, 3]


def test_greedy_empty_stream():
    assert list(pack_greedy([], pack_len=8)) == []


def test_greedy_oversize_raises():
    with pytest.raises(ValueError, match="pack_len"):
        list(pack_greedy([fake(0, 10)], pack_len=8))
