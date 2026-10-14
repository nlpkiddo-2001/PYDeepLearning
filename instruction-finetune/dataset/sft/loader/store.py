"""Read side of the token cache: O(1) random access to (ids, loss_mask) by sample index, via memory maps."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


class CacheError(RuntimeError):
    pass


class SourceStore:
    def __init__(self, directory: Path, max_open_parts: int = 48):
        self.dir = Path(directory)
        done = sorted(self.dir.glob("part-*.done.json"))
        if not done:
            raise CacheError(f"{self.dir}: no compiled parts (run `python -m sft_data.cli compile`)")
        self.stems = [p.name.split(".")[0] for p in done]
        self.row_start = [json.loads(p.read_text())["row_start"] for p in done]
        metas = [pd.read_parquet(self.dir / f"{s}.meta.parquet") for s in self.stems]
        sizes = np.array([len(m) for m in metas], dtype=np.int64)
        self.part_starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        self.meta = pd.concat(metas, ignore_index=True)
        self.n = len(self.meta)
        self.n_tokens = self.meta.n_tokens.to_numpy()
        self._open: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self._max_open = max_open_parts

    def _part(self, p: int):
        hit = self._open.get(p)
        if hit is not None:
            self._open.move_to_end(p)
            return hit
        stem = self.stems[p]
        arrs = (np.load(self.dir / f"{stem}.ids.npy", mmap_mode="r"),
                np.load(self.dir / f"{stem}.mask.npy", mmap_mode="r"),
                np.load(self.dir / f"{stem}.off.npy"))
        self._open[p] = arrs
        while len(self._open) > self._max_open:
            self._open.popitem(last=False)
        return arrs

    def get(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        p = int(np.searchsorted(self.part_starts, idx, side="right")) - 1
        ids, mask, off = self._part(p)
        local = idx - int(self.part_starts[p])
        a, b = int(off[local]), int(off[local + 1])
        return ids[a:b], mask[a:b]

    def locate(self, idx: int) -> tuple[str, int]:
        """(shard stem such as 'part-00003', row within that shard's JSONL) - the sample's source coordinates."""
        p = int(np.searchsorted(self.part_starts, idx, side="right")) - 1
        return self.stems[p].rsplit("-c", 1)[0], self.row_start[p] + idx - int(self.part_starts[p])

    def info(self) -> dict:
        f = self.dir / "_source.json"
        return json.loads(f.read_text()) if f.exists() else {}
