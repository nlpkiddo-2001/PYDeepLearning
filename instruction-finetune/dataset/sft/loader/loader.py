"""PackedStream (framework-agnostic, numpy) and PackedDataset (torch IterableDataset adapter).

Batch dict (all arrays are [micro_batch_size, seq_len] unless noted):
  input_ids, labels (-100 = not trained), position_ids (restart per document), doc_ids (-1 = padding),
  cu_seqlens int32 [n+1] and max_seqlen  -> flash-attn varlen over the flattened batch (padding forms its own segment),
  n_trainable  int   -> trained tokens in this batch: normalise the loss by the SUM of this over all ranks/accumulation steps,
  n_real       int   -> non-padding tokens.

Exact resume: `state_dict()` / `load_state_dict()` restore sampler, packer window and counters; the topology
(world size, rank) and the data-config hash are verified. The stream runs in-process (no DataLoader workers) because the
state must be observable; use `prefetch` for overlap with the GPU.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Iterator, Optional

import numpy as np

from dataset.sft.loader.catalog import Catalog, SourceView, build_catalog
from dataset.sft.loader.compile import make_cache_spec
from dataset.sft.loader.config import MixConfig
from dataset.sft.loader.metrics import Metrics
from dataset.sft.loader.packer import IGNORE, Pack, Packer
from dataset.sft.loader.sampler import DocRef, MixtureSampler, SequentialSampler

log = logging.getLogger("sft_data")
STATE_VERSION = 1


class StateMismatchError(RuntimeError):
    pass


def collate(packs: list[Pack]) -> dict[str, Any]:
    ids = np.stack([p.input_ids for p in packs])
    labels = np.stack([p.labels for p in packs])
    pos = np.stack([p.position_ids for p in packs])
    docs = np.stack([p.doc_ids for p in packs])
    B, L = docs.shape
    flat = docs.reshape(-1)
    row = np.repeat(np.arange(B), L)
    change = np.flatnonzero((flat[1:] != flat[:-1]) | (row[1:] != row[:-1])) + 1
    cu = np.concatenate([[0], change, [B * L]]).astype(np.int32)
    return {"input_ids": ids, "labels": labels, "position_ids": pos, "doc_ids": docs,
            "cu_seqlens": cu, "max_seqlen": int(np.diff(cu).max()),
            "n_trainable": int((labels != IGNORE).sum()), "n_real": int(sum(p.n_real for p in packs))}


def to_torch(batch: dict[str, Any]) -> dict[str, Any]:
    import torch
    out = dict(batch)
    for k in ("input_ids", "labels"):
        out[k] = torch.from_numpy(batch[k].astype(np.int64))
    for k in ("position_ids", "doc_ids", "cu_seqlens"):
        out[k] = torch.from_numpy(batch[k].astype(np.int32))
    return out


def _pad_id(tok: Any) -> int:
    for v in (tok.pad_token_id, tok.eos_token_id, 0):
        if v is not None:
            return int(v)
    return 0


class PackedStream:
    def __init__(self, cfg: MixConfig, catalog: Catalog, pad_id: int, rank: int = 0, world_size: int = 1,
                 template_text: Optional[str] = None, seed: Optional[int] = None, drop_last: bool = True):
        self.cfg, self.catalog, self.rank, self.world = cfg, catalog, rank, world_size
        self.views: list[SourceView] = catalog.views
        self.pad_id, self.drop_last = pad_id, drop_last
        self.config_hash = cfg.data_hash(template_text)
        self.metrics = Metrics([v.name for v in self.views], [v.domain for v in self.views])
        self.sampler = MixtureSampler(cfg, self.views, rank, world_size, seed)
        self.packer = Packer(self.sampler, self.views, cfg.seq_len, cfg.pack_window, cfg.over_length,
                             cfg.min_truncated_trainable, pad_id, self.metrics)
        self.batches = 0
        self.tokenizer: Any = None

    @classmethod
    def from_config(cls, cfg: MixConfig, rank: int = 0, world_size: int = 1, seed: Optional[int] = None,
                    drop_last: bool = True) -> "PackedStream":
        spec, tok, template = make_cache_spec(cfg.tokenizer, cfg.chat_template, cfg.cache_dir)
        catalog = build_catalog(cfg, spec)
        stream = cls(cfg, catalog, _pad_id(tok), rank, world_size, template, seed, drop_last)
        stream.tokenizer = tok
        return stream

    # ---------------------------------------------------------------- iteration
    def next_batch(self) -> Optional[dict[str, Any]]:
        packs: list[Pack] = []
        for _ in range(self.cfg.micro_batch_size):
            p = self.packer.next_pack()
            if p is None:
                break
            packs.append(p)
        if not packs or (self.drop_last and len(packs) < self.cfg.micro_batch_size):
            return None
        while len(packs) < self.cfg.micro_batch_size:       # drop_last=False: pad the final batch with empty packs
            packs.append(_empty_pack(self.cfg.seq_len, self.pad_id))
        self.batches += 1
        return collate(packs)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while (b := self.next_batch()) is not None:
            yield b

    # ------------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {"version": STATE_VERSION, "config_hash": self.config_hash, "rank": self.rank, "world_size": self.world,
                "sources": [v.name for v in self.views], "batches": self.batches,
                "sampler": self.sampler.state_dict(), "packer": self.packer.state_dict(),
                "metrics": self.metrics.state_dict()}

    def load_state_dict(self, st: dict) -> None:
        if st.get("version") != STATE_VERSION:
            raise StateMismatchError(f"loader state version {st.get('version')} != {STATE_VERSION}")
        for key, mine in (("config_hash", self.config_hash), ("world_size", self.world), ("rank", self.rank),
                          ("sources", [v.name for v in self.views])):
            if st[key] != mine:
                raise StateMismatchError(f"cannot resume: {key} differs (checkpoint={st[key]!r}, now={mine!r}). "
                                         "Exact resume needs the same data config, sources and world size.")
        self.sampler.load_state_dict(st["sampler"])
        self.packer.load_state_dict(st["packer"])
        self.metrics.load_state_dict(st["metrics"])
        self.batches = int(st["batches"])

    # --------------------------------------------------------------------- eval
    def eval_batches(self, source: Optional[str] = None, rank: Optional[int] = None,
                     world_size: Optional[int] = None) -> list[dict[str, Any]]:
        """Packed held-out batches for one source (or all). Identical packing on every rank, then packs are dealt
        round-robin and padded with empty packs so every rank gets the same number of batches (safe for collectives)."""
        rank = self.rank if rank is None else rank
        world = self.world if world_size is None else world_size
        refs: list[DocRef] = []
        for si, v in enumerate(self.views):
            if source and v.name != source:
                continue
            eff = v.cut[v.val_idx]
            refs += [DocRef(si, int(i), int(e)) for i, e in zip(v.val_idx, eff)]
        m = Metrics([v.name for v in self.views], [v.domain for v in self.views])
        packer = Packer(SequentialSampler(refs), self.views, self.cfg.seq_len, self.cfg.pack_window,
                        self.cfg.over_length, self.cfg.min_truncated_trainable, self.pad_id, m)
        packs = []
        while (p := packer.next_pack()) is not None:
            packs.append(p)
        B = self.cfg.micro_batch_size
        per_rank = -(-len(packs) // world) if packs else 0
        mine = packs[rank::world]
        mine += [_empty_pack(self.cfg.seq_len, self.pad_id)] * (per_rank - len(mine))
        mine += [_empty_pack(self.cfg.seq_len, self.pad_id)] * (-len(mine) % B)
        return [collate(mine[i:i + B]) for i in range(0, len(mine), B)]


def _empty_pack(L: int, pad_id: int) -> Pack:
    return Pack(np.full(L, pad_id, np.int32), np.full(L, IGNORE, np.int32), np.zeros(L, np.int32),
                np.full(L, -1, np.int32), 0, 0)


try:  # torch is optional: the numpy stream works without it
    from torch.utils.data import IterableDataset as _IterableBase, get_worker_info as _worker_info
except ImportError:  # pragma: no cover
    _IterableBase, _worker_info = object, lambda: None


class PackedDataset(_IterableBase):
    """torch `IterableDataset` over a PackedStream. Use `DataLoader(ds, batch_size=None, num_workers=0)`.

    prefetch > 0 builds batches on a background thread (numpy memcpy releases the GIL); `state_dict()` then returns the
    state *after the last batch handed to the trainer*, so resume is exact regardless of prefetch depth.
    """

    def __init__(self, stream: PackedStream, prefetch: int = 0, as_torch: bool = True):
        self.stream, self.prefetch, self.as_torch = stream, prefetch, as_torch
        self._last_state: Optional[dict] = None

    def state_dict(self) -> dict:
        return self._last_state if (self.prefetch and self._last_state is not None) else self.stream.state_dict()

    def load_state_dict(self, st: dict) -> None:
        self.stream.load_state_dict(st)
        self._last_state = st

    def _convert(self, b):
        return to_torch(b) if self.as_torch else b

    def __iter__(self):
        info = _worker_info()
        if info is not None and info.num_workers > 0:
            raise RuntimeError("PackedDataset keeps resumable state in-process: use DataLoader(num_workers=0) "
                               "and `prefetch=N` for overlap")
        if not self.prefetch:
            for b in self.stream:
                yield self._convert(b)
            return
        q: queue.Queue = queue.Queue(self.prefetch)
        stop = threading.Event()
        end = object()

        def work():
            try:
                while not stop.is_set():
                    b = self.stream.next_batch()
                    item = end if b is None else (b, self.stream.state_dict())
                    while not stop.is_set():
                        try:
                            q.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if b is None:
                        return
            except BaseException as e:  # noqa - surface worker errors in the consumer
                q.put(e)

        self._last_state = self.stream.state_dict()
        t = threading.Thread(target=work, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is end:
                    return
                if isinstance(item, BaseException):
                    raise item
                batch, state = item
                self._last_state = state
                yield self._convert(batch)
        finally:
            stop.set()
            t.join()
            # batches the thread built but the trainer never received must not be lost
            self.stream.load_state_dict(self._last_state)
