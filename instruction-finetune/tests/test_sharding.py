"""
Data sharding for multi-GPU: each rank must see different records, together they must see all of them,
and a rank with nothing to train on must crash instead of hanging the other ranks.

No torchrun needed: we call row_stream once per rank and pretend to be that rank.
"""
import json
from collections import Counter

import pytest

from sft.data.chat_tokenizer import ChatTokenizer, TokenizerConfig
from sft.train.train import row_stream

FIXTURE = "tests/fixtures/sample.jsonl"
WORLD_SIZE = 8


def write_copies(path, n):
    """n lines cycling through the fixture records, each with a unique id"""
    records = [json.loads(line) for line in open(FIXTURE)]
    ids = []
    with open(path, "w") as f:
        for i in range(n):
            rec = dict(records[i % len(records)])
            rec["id"] = f"{rec['id']}-{i}"
            f.write(json.dumps(rec) + "\n")
            ids.append(rec["id"])
    return ids


def first_row_ids(path, tok, rank, world_size):
    chat_tok = ChatTokenizer(tok, TokenizerConfig(max_seq_len=4096))
    stream = row_stream(path, chat_tok, 4096, Counter(), rank=rank, world_size=world_size)
    return set(next(stream).sample_ids)


def test_shards_are_disjoint_and_cover_all_records(tok, tmp_path):
    path = tmp_path / "data.jsonl"
    all_ids = write_copies(path, 16)                          # 2 lines per rank
    trainable = {i for i in all_ids if not i.startswith("all-frozen")}  # all-frozen has no train turns

    per_rank = [first_row_ids(path, tok, r, WORLD_SIZE) for r in range(WORLD_SIZE)]

    for a in range(WORLD_SIZE):
        for b in range(a + 1, WORLD_SIZE):
            assert not per_rank[a] & per_rank[b], f"ranks {a} and {b} share records"
    assert set().union(*per_rank) == trainable


@pytest.mark.parametrize(
    "rank",
    [5, 2],
    ids=["owns_no_lines", "owns_only_a_rejected_line"],   # 3 lines on 8 ranks: rank 5 gets none, rank 2 gets all-frozen
)
def test_rank_without_samples_raises_instead_of_hanging(tok, rank):
    with pytest.raises(RuntimeError, match=f"rank {rank}/{WORLD_SIZE}"):
        first_row_ids(FIXTURE, tok, rank, WORLD_SIZE)
