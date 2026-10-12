"""
Module for Packing

Takes the "TokenizedSample" from tokenizer and turns them into fixed size training rows.

Packed row format: several samples concatenated into one row.

    input_ids:     [a0 a1 | c0 c1 c2 | d0 d1]
    position_ids:  [ 0  1 |  0  1  2 |  0  1]    restart at 0 for every sample
    labels:        [-100 . | -100 . . | -100 .]   first label of every sample forced to -100

- position_ids resets are how transformers finds sample boundaries and builds a block-diagonal
  causal mask (sdpa/eager/flex) or cu_seqlens (flash-attention), so samples never attend to each
  other. This only works if the model gets NO attention_mask and use_cache=False.
- The model shifts labels internally, so the last token of one sample would predict the first
  token of the next one. Forcing each sample's first label to -100 removes that cross-sample target.
- A sample is never split across rows: the tokenizer already dropped samples longer than
  max_seq_len, and the pack length equals max_seq_len, so every sample fits in an empty row.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from sft.data.chat_tokenizer import IGNORE_INDEX, TokenizedSample

@dataclass
class PackedRow:
    input_ids : list[int]
    labels : list[int]
    position_ids : list[int]
    seq_lens: list[int]
    num_supervised : int # loss norm needs the total supervised tokens across all micro batches and gpus. fix from the grad_acc blog
    sample_ids : list[str]


def make_row(samples: list[TokenizedSample]):
    """make a packed row from the tokenized samples"""
    input_ids, labels, position_ids, seq_lens, sample_ids = [], [], [], [], []
    for sample in samples:
        n = len(sample.input_ids)
        input_ids.extend(sample.input_ids)

        labels.append(IGNORE_INDEX)
        labels.extend(sample.labels[1:])
        position_ids.extend(range(n))
        seq_lens.append(n)
        sample_ids.append(sample.meta["id"])

    num_supervised = sum(l != IGNORE_INDEX for l in labels)
    return PackedRow(input_ids, labels, position_ids, seq_lens, num_supervised, sample_ids)


def pack_greedy(samples: Iterable[TokenizedSample], pack_len: int) -> Iterable[PackedRow]:
    """
    It reads one sample, outputs rows as they fill, and never holds more than one row in memory.
    """
    current : list[TokenizedSample] = []
    current_len = 0

    for sample in samples:
        n = len(sample.input_ids)

        if n > pack_len:
            raise ValueError(
                f"sample {sample.meta['id']} has {n} tokens "
                f"> pack_len={pack_len}"
            )

        if current_len + n > pack_len:
            yield(make_row(current))
            current = []
            current_len = 0
        current.append(sample)
        current_len += n

    if current:
        yield(make_row(current))

