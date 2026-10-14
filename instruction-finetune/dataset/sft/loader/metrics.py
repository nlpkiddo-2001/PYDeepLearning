"""Counters describing what the loader actually produced (per rank; all-reduce them in the trainer if needed)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


class Metrics:
    def __init__(self, names: Sequence[str], domains: Sequence[str]):
        self.names, self.domains = list(names), list(domains)
        S = len(self.names)
        self.docs = np.zeros(S, dtype=np.int64)
        self.tokens = np.zeros(S, dtype=np.int64)
        self.trainable = np.zeros(S, dtype=np.int64)
        self.truncated = np.zeros(S, dtype=np.int64)
        self.dropped_overlength = np.zeros(S, dtype=np.int64)
        self.packs = 0
        self.real_tokens = 0
        self.pad_tokens = 0

    def doc(self, s: int, tokens: int, trainable: int, truncated: bool) -> None:
        self.docs[s] += 1
        self.tokens[s] += tokens
        self.trainable[s] += trainable
        self.truncated[s] += int(truncated)

    def pack(self, real: int, pad: int) -> None:
        self.packs += 1
        self.real_tokens += real
        self.pad_tokens += pad

    def snapshot(self) -> dict:
        tot = max(int(self.tokens.sum()), 1)
        doms: dict[str, float] = {}
        for d, t in zip(self.domains, self.tokens):
            doms[d] = doms.get(d, 0.0) + float(t) / tot
        seen = self.real_tokens + self.pad_tokens
        return {
            "packs": self.packs,
            "fill_rate": self.real_tokens / seen if seen else 0.0,
            "trainable_fraction": float(self.trainable.sum()) / tot,
            "tokens": int(self.tokens.sum()),
            "domain_token_share": doms,
            "source_token_share": {n: float(t) / tot for n, t in zip(self.names, self.tokens)},
            "truncated_docs": {n: int(t) for n, t in zip(self.names, self.truncated) if t},
            "dropped_overlength": {n: int(t) for n, t in zip(self.names, self.dropped_overlength) if t},
            "docs_per_pack": float(self.docs.sum()) / self.packs if self.packs else 0.0,
        }

    def state_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("docs", "tokens", "trainable", "truncated", "dropped_overlength")} | {
            "packs": self.packs, "real_tokens": self.real_tokens, "pad_tokens": self.pad_tokens}

    def load_state_dict(self, st: dict) -> None:
        for k in ("docs", "tokens", "trainable", "truncated", "dropped_overlength"):
            setattr(self, k, np.array(st[k], dtype=np.int64))
        self.packs, self.real_tokens, self.pad_tokens = st["packs"], st["real_tokens"], st["pad_tokens"]
