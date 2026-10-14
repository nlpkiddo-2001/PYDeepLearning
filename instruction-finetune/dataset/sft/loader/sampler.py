"""Deterministic token-budget mixture sampler.

Each source is visited in a seeded per-epoch permutation; *which* source supplies the next document is chosen by
largest token deficit against the stage's target shares (deficit round-robin). Unlike random sampling this makes
the realized token mix track the target to within one document, for any run length, with no RNG state to restore.

Distribution: epoch permutations are identical on all ranks and rank r takes `perm[r::world]`, so ranks consume
disjoint documents with no communication. State is a handful of integers -> exact resume.
"""
from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from dataset.sft.loader.catalog import SourceView
from dataset.sft.loader.config import MixConfig

log = logging.getLogger("sft_data")


@dataclass(frozen=True)
class DocRef:
    src: int        # index into the catalog's view list
    idx: int        # index into that source's store
    length: int     # accounting length (min(n_tokens, seq_len))
    stage: int = 0


class MixtureSampler:
    def __init__(self, cfg: MixConfig, views: Sequence[SourceView], rank: int = 0, world_size: int = 1,
                 seed: Optional[int] = None):
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.cfg, self.views, self.rank, self.world = cfg, list(views), rank, world_size
        self.seed = cfg.seed if seed is None else seed
        S = len(self.views)
        avail = {v.name: float(v.train_tokens) for v in self.views}
        n_stages = max(1, len(cfg.stages))
        names = [v.name for v in self.views]
        self.stage_probs = []
        for k in range(n_stages):
            pr = cfg.stage_weights(k, avail)
            self.stage_probs.append(np.array([pr.get(n, 0.0) for n in names]))
        self.boundaries = [(st.until_tokens / world_size) if st.until_tokens is not None else float("inf")
                           for st in (cfg.stages or [None])] if cfg.stages else [float("inf")]
        self.n_rank = np.array([max(0, -(-(len(v.train_idx) - rank) // world_size)) for v in self.views], dtype=np.int64)
        self.allowed = np.array([int(np.floor(cfg.max_epochs_for(v.name) * n)) for v, n in zip(self.views, self.n_rank)],
                                dtype=np.int64)
        self._salt = [zlib.crc32(n.encode()) for n in names]
        self._slice_cache: dict[int, tuple[int, np.ndarray]] = {}
        self._fallback = np.array([a ** cfg.alpha if a > 0 else 0.0 for a in avail.values()])
        self.reset()
        for v, n in zip(self.views, self.n_rank):
            if n == 0:
                log.warning("[%s] has no documents for rank %d/%d", v.name, rank, world_size)

    # -------------------------------------------------------------------- state
    def reset(self) -> None:
        S = len(self.views)
        self.served = np.zeros(S, dtype=np.int64)          # docs served per source (epoch = served // n_rank)
        self.stage = 0
        self.stage_served = np.zeros(S, dtype=np.float64)  # tokens served per source within the current stage
        self.stage_total = 0.0
        self.total_tokens = 0
        self.prev_remaining: Optional[list[bool]] = None   # active-source set at the previous draw
        self._warned_fallback = False

    def state_dict(self) -> dict:
        return {"served": self.served.tolist(), "stage": self.stage, "stage_served": self.stage_served.tolist(),
                "stage_total": self.stage_total, "total_tokens": self.total_tokens,
                "prev_remaining": self.prev_remaining}

    def load_state_dict(self, st: dict) -> None:
        if len(st["served"]) != len(self.views):
            raise ValueError("sampler state does not match the configured sources")
        self.served = np.array(st["served"], dtype=np.int64)
        self.stage = int(st["stage"])
        self.stage_served = np.array(st["stage_served"], dtype=np.float64)
        self.stage_total = float(st["stage_total"])
        self.total_tokens = int(st["total_tokens"])
        self.prev_remaining = st.get("prev_remaining")

    # ----------------------------------------------------------------- sampling
    def _epoch_slice(self, s: int, epoch: int) -> np.ndarray:
        hit = self._slice_cache.get(s)
        if hit is not None and hit[0] == epoch:
            return hit[1]
        n = len(self.views[s].train_idx)
        perm = np.random.default_rng([self.seed, self._salt[s], epoch]).permutation(n)
        sl = perm[self.rank::self.world]
        self._slice_cache[s] = (epoch, sl)
        return sl

    @property
    def exhausted(self) -> bool:
        return bool((self.served >= self.allowed).all())

    def epochs(self) -> dict[str, float]:
        return {v.name: float(self.served[i]) / max(int(self.n_rank[i]), 1) for i, v in enumerate(self.views)}

    def next(self) -> Optional[DocRef]:
        while self.stage < len(self.stage_probs) - 1 and self.total_tokens >= self.boundaries[self.stage]:
            self.stage += 1
            self.stage_served[:] = 0.0
            self.stage_total = 0.0
        remaining = self.served < self.allowed
        now = remaining.tolist()
        if self.prev_remaining is not None and now != self.prev_remaining:
            # a source hit its epoch cap: targets renormalize over the rest FROM NOW ON. Keeping the old
            # deficits would make the sources that were behind catch up in one long burst.
            self.stage_served[:] = 0.0
            self.stage_total = 0.0
        self.prev_remaining = now
        p = np.where(remaining, self.stage_probs[self.stage], 0.0)
        if p.sum() <= 0:
            p = np.where(remaining, self._fallback, 0.0)
            if p.sum() <= 0:
                return None
            if not self._warned_fallback:
                log.warning("stage %d: all weighted sources are exhausted; falling back to remaining sources", self.stage)
                self._warned_fallback = True
        t = p / p.sum()
        deficit = np.where(p > 0, t * (self.stage_total + 1.0) - self.stage_served, -np.inf)
        s = int(np.argmax(deficit))
        n = int(self.n_rank[s])
        e, pos = divmod(int(self.served[s]), n)
        doc = int(self._epoch_slice(s, e)[pos])
        view = self.views[s]
        length = int(view.eff_len[doc])
        self.served[s] += 1
        self.stage_served[s] += length
        self.stage_total += length
        self.total_tokens += length
        return DocRef(s, int(view.train_idx[doc]), length, self.stage)


class SequentialSampler:
    """Fixed, ordered list of documents (evaluation)."""

    def __init__(self, refs: Sequence[DocRef]):
        self.refs, self.pos = list(refs), 0

    def next(self) -> Optional[DocRef]:
        if self.pos >= len(self.refs):
            return None
        r = self.refs[self.pos]
        self.pos += 1
        return r

    def state_dict(self) -> dict:
        return {"pos": self.pos}

    def load_state_dict(self, st: dict) -> None:
        self.pos = int(st["pos"])


def simulate(cfg: MixConfig, views: Sequence[SourceView], budget_tokens: float, world_size: int = 1) -> dict:
    """Run the sampler alone (lengths only, no token reads) to see realized shares/epochs for a token budget (rank 0)."""
    s = MixtureSampler(cfg, views, 0, world_size)
    per_rank = budget_tokens / world_size
    tok = np.zeros(len(views))
    docs = np.zeros(len(views), dtype=np.int64)
    by_stage: dict[int, np.ndarray] = {}
    got = 0.0
    while got < per_rank:
        r = s.next()
        if r is None:
            break
        tok[r.src] += r.length
        docs[r.src] += 1
        by_stage.setdefault(r.stage, np.zeros(len(views)))[r.src] += r.length
        got += r.length
    return {"tokens_per_rank": tok, "docs_per_rank": docs, "epochs": s.epochs(), "reached": got >= per_rank,
            "achieved_tokens": got * world_size, "stage_tokens": by_stage, "exhausted": s.exhausted}
