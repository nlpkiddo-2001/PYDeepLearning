"""Windowed best-fit sequence packing with per-document boundaries.

A window of up to `window` documents is kept sorted by length; each pack repeatedly takes the largest document that
still fits (best-fit decreasing), then the window is refilled from the sampler. Documents are never split across
sequences. A document longer than seq_len is cut at the last *complete assistant turn* that fits (see turns.py; so a tool call is
never cut in half). The cut length is decided once, in the catalog, so the sampler's token accounting is exact.
"""
from __future__ import annotations

from bisect import bisect_right, insort
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from dataset.sft.loader.catalog import SourceView
from dataset.sft.loader.metrics import Metrics
from dataset.sft.loader.turns import truncate_at_turn  # noqa: F401  (re-exported)

IGNORE = -100


@dataclass
class Pack:
    input_ids: np.ndarray       # int32 [L]
    labels: np.ndarray          # int32 [L]  (-100 = not trained)
    position_ids: np.ndarray    # int32 [L]  restart at 0 for every document
    doc_ids: np.ndarray         # int32 [L]  0..k-1 per document, -1 = padding
    n_real: int
    n_trainable: int
    doc_src: list[int] = field(default_factory=list)
    doc_len: list[int] = field(default_factory=list)
    doc_idx: list[int] = field(default_factory=list)   # store index of each document (traceability / tests)


class Packer:
    def __init__(self, sampler, views: Sequence[SourceView], seq_len: int, window: int, over_length: str,
                 min_truncated_trainable: int, pad_id: int, metrics: Metrics):
        self.sampler, self.views, self.L = sampler, list(views), seq_len
        self.window, self.over_length = window, over_length
        self.min_trunc, self.pad_id, self.metrics = min_truncated_trainable, pad_id, metrics
        self._keys: list[tuple[int, int]] = []          # (cut, counter), sorted
        self._items: dict[int, tuple[int, int, bool]] = {}   # counter -> (src, store idx, truncated)
        self._counter = 0
        self._sampler_done = False

    # ------------------------------------------------------------------ window
    def _admit(self, src: int, idx: int, cut: int, truncated: bool) -> None:
        key = (cut, self._counter)
        insort(self._keys, key)
        self._items[self._counter] = (src, idx, truncated)
        self._counter += 1

    def _cut_for(self, ref) -> Optional[tuple[int, bool]]:
        view = self.views[ref.src]
        n, cut = int(view.store.n_tokens[ref.idx]), int(view.cut[ref.idx])   # cut: exact kept length, set by the catalog
        if cut <= 0 or cut > self.L:
            return None
        return cut, cut < n

    def _fill(self) -> None:
        while len(self._keys) < self.window and not self._sampler_done:
            ref = self.sampler.next()
            if ref is None:
                self._sampler_done = True
                break
            got = self._cut_for(ref)
            if got is None:
                self.metrics.dropped_overlength[ref.src] += 1
                continue
            self._admit(ref.src, ref.idx, got[0], got[1])

    # -------------------------------------------------------------------- pack
    def next_pack(self) -> Optional[Pack]:
        self._fill()
        if not self._keys:
            return None
        L = self.L
        chosen: list[tuple[int, int]] = []
        cap = L
        while cap > 0:
            i = bisect_right(self._keys, (cap, float("inf"))) - 1
            if i < 0:
                break
            key = self._keys.pop(i)
            chosen.append(key)
            cap -= key[0]
        ids = np.full(L, self.pad_id, np.int32)
        labels = np.full(L, IGNORE, np.int32)
        pos = np.zeros(L, np.int32)
        docs = np.full(L, -1, np.int32)
        o, trainable = 0, 0
        doc_src, doc_len, doc_idx = [], [], []
        for d, (cut, counter) in enumerate(chosen):
            src, idx, truncated = self._items.pop(counter)
            a, m = self.views[src].store.get(idx)
            a, m = a[:cut], m[:cut]
            ids[o:o + cut] = a
            lab = np.where(m == 1, a, IGNORE)
            lab[0] = IGNORE            # nothing predicts a document's first token
            labels[o:o + cut] = lab
            pos[o:o + cut] = np.arange(cut, dtype=np.int32)
            docs[o:o + cut] = d
            t = int((lab != IGNORE).sum())
            trainable += t
            self.metrics.doc(src, cut, t, truncated)
            doc_src.append(src)
            doc_len.append(cut)
            doc_idx.append(idx)
            o += cut
        self.metrics.pack(o, L - o)
        return Pack(ids, labels, pos, docs, o, trainable, doc_src, doc_len, doc_idx)

    # ------------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {"entries": [[k[0], k[1], *self._items[k[1]][:2], int(self._items[k[1]][2])] for k in self._keys],
                "counter": self._counter, "sampler_done": self._sampler_done}

    def load_state_dict(self, st: dict) -> None:
        self._keys, self._items = [], {}
        for cut, counter, src, idx, trunc in st["entries"]:
            self._keys.append((cut, counter))
            self._items[counter] = (src, idx, bool(trunc))
        self._keys.sort()
        self._counter, self._sampler_done = st["counter"], st["sampler_done"]
