"""One-time tokenization of `out/<group>/<source>/part-*.jsonl` into a versioned, memory-mappable token cache.

Layout (per tokenizer fingerprint, per source, one set of files per chunk of rows of an input shard so compiles are parallel + resumable):
  <cache_dir>/<fingerprint12>/<source>/part-00000-c000.ids.npy   int32  concatenated token ids
                                       part-00000-c000.mask.npy  uint8  1 = trained token
                                       part-00000-c000.off.npy   int64  [n_samples + 1] offsets
                                       part-00000-c000.meta.parquet     one row per input line (aligned with the JSONL row)
                                       part-00000-c000.done.json        input shard size + counts (written last, atomically)
                                       _source.json                source-level summary
Samples that cannot be rendered keep their row (status != 0, zero tokens) so row i of the cache == row i of the shard.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from dataset.sft.loader.render import OK, STATUS_NAMES, Renderer, load_tokenizer, tokenizer_fingerprint

log = logging.getLogger("sft_data")
META_COLUMNS = ["id", "content_hash", "n_tokens", "n_trainable", "status", "tool_valid", "lang",
                "n_tool_calls", "n_messages", "n_misaligned"]


@dataclass
class CacheSpec:
    tokenizer: str
    template_path: Optional[str]
    cache_root: Path
    fingerprint: str

    @property
    def root(self) -> Path:
        return self.cache_root / self.fingerprint[:12]

    def source_dir(self, source: str) -> Path:
        return self.root / source


def make_cache_spec(tokenizer: str, template_path: Optional[str], cache_dir: str) -> tuple[CacheSpec, Any, Optional[str]]:
    tok, template = load_tokenizer(tokenizer, template_path)
    return CacheSpec(tokenizer, template_path, Path(cache_dir), tokenizer_fingerprint(tok, template)), tok, template


def list_shards(src_dir: Path) -> list[Path]:
    return sorted(p for p in src_dir.glob("part-*.jsonl*") if p.suffix in (".jsonl", ".gz"))


def open_shard(path: Path):
    import gzip
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _part_done(out_dir: Path, stem: str, shard: Path, row_start: int, row_end: int) -> bool:
    d = out_dir / f"{stem}.done.json"
    if not d.exists():
        return False
    try:
        info = json.loads(d.read_text())
    except json.JSONDecodeError:
        return False
    return (info.get("input_size") == shard.stat().st_size and info.get("row_start") == row_start
            and info.get("row_end") == row_end
            and all((out_dir / f"{stem}.{k}").exists() for k in ("ids.npy", "mask.npy", "off.npy", "meta.parquet")))


_WORKER: dict[str, Any] = {}


def _init_worker(tokenizer: str, template_path: Optional[str]) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tok, _ = load_tokenizer(tokenizer, template_path)
    _WORKER["renderer"] = Renderer(tok)


def compile_part(task: tuple) -> dict[str, Any]:
    """Render rows [row_start, row_end) of one JSONL shard -> part files. Runs in a worker process (or in-process)."""
    source, shard_path, out_dir_s, stem, row_start, row_end = task
    shard, out_dir = Path(shard_path), Path(out_dir_s)
    renderer: Renderer = _WORKER["renderer"]
    ids_parts, mask_parts, offsets, rows = [], [], [0], []
    stats: Counter = Counter()
    errors: list[str] = []
    t0 = time.perf_counter()
    with open_shard(shard) as f:
        for k, line in enumerate(f):
            if k < row_start:
                continue
            if k >= row_end:
                break
            if not line.strip():
                raise ValueError(f"{shard}: blank line at row {k} would misalign cache rows")
            rec = json.loads(line)
            meta = rec.get("meta", {})
            r = renderer.render(rec)
            stats[STATUS_NAMES[r.status]] += 1
            if r.status != OK and len(errors) < 3:
                errors.append(f"{rec.get('id')}: {STATUS_NAMES[r.status]} {r.detail}")
            ids_parts.append(r.input_ids)
            mask_parts.append(r.loss_mask)
            offsets.append(offsets[-1] + len(r.input_ids))
            rows.append({
                "id": rec["id"], "content_hash": meta.get("content_hash", ""),
                "n_tokens": len(r.input_ids), "n_trainable": int(r.loss_mask.sum()), "status": r.status,
                "tool_valid": bool(meta.get("tool_valid", True)), "lang": str(meta.get("lang", "")),
                "n_tool_calls": int(meta.get("n_tool_calls", 0)), "n_messages": int(meta.get("n_messages", len(rec["messages"]))),
                "n_misaligned": r.n_misaligned})
            stats["tokens"] += len(r.input_ids)
            stats["misaligned"] += r.n_misaligned
    if len(rows) != row_end - row_start:
        raise ValueError(f"{shard}: expected rows [{row_start}, {row_end}) but read {len(rows)}")
    ids = np.concatenate(ids_parts) if ids_parts else np.zeros(0, np.int32)
    mask = np.concatenate(mask_parts) if mask_parts else np.zeros(0, np.uint8)
    _atomic_save(out_dir / f"{stem}.ids.npy", ids)
    _atomic_save(out_dir / f"{stem}.mask.npy", mask)
    _atomic_save(out_dir / f"{stem}.off.npy", np.asarray(offsets, np.int64))
    tmp = out_dir / f"{stem}.meta.parquet.tmp"
    pd.DataFrame(rows, columns=META_COLUMNS).to_parquet(tmp, index=False)
    os.replace(tmp, out_dir / f"{stem}.meta.parquet")
    done = {"input_size": shard.stat().st_size, "row_start": row_start, "row_end": row_end,
            "n_samples": len(rows), "n_tokens": int(len(ids)), "seconds": round(time.perf_counter() - t0, 2)}
    tmp = out_dir / f"{stem}.done.json.tmp"
    tmp.write_text(json.dumps(done))
    os.replace(tmp, out_dir / f"{stem}.done.json")
    return {"source": source, "stem": stem, "stats": dict(stats), "errors": errors}


@dataclass
class CompileJob:
    name: str
    path: Path
    license: Optional[str] = None
    shard_rows: Optional[dict[str, int]] = None   # shard file name -> rows (from the build manifest)


def _count_rows(shard: Path) -> int:
    with open_shard(shard) as f:
        return sum(1 for _ in f)


def plan_tasks(job: CompileJob, spec: CacheSpec, chunk_rows: int, force: bool) -> tuple[list[tuple], list[str]]:
    """(tasks still to run, all expected part stems). Also removes cache files of parts that are no longer expected."""
    out_dir = spec.source_dir(job.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks, expected = [], []
    for sh in list_shards(job.path):
        n = (job.shard_rows or {}).get(sh.name)
        if n is None:
            n = _count_rows(sh)
        base = sh.name.split(".")[0]
        for c, start in enumerate(range(0, max(n, 1), chunk_rows)):
            stem, end = f"{base}-c{c:03d}", min(start + chunk_rows, n)
            expected.append(stem)
            if force or not _part_done(out_dir, stem, sh, start, end):
                tasks.append((job.name, str(sh), str(out_dir), stem, start, end))
    keep = set(expected)
    for f in out_dir.glob("part-*"):
        if f.name.split(".")[0] not in keep:
            f.unlink()          # stale part from an earlier layout / shard count
    return tasks, expected


def compile_sources(jobs: list[CompileJob], spec: CacheSpec, workers: int = 1, force: bool = False,
                    chunk_rows: int = 2000) -> dict[str, dict[str, Any]]:
    """Compile many sources with ONE worker pool (small sources don't leave cores idle). Resumable and idempotent."""
    plans, all_tasks = {}, []
    for job in jobs:
        tasks, expected = plan_tasks(job, spec, chunk_rows, force)
        plans[job.name] = (len(tasks), len(expected))
        all_tasks += tasks
        log.info("[%s] %d/%d parts to compile (%s)", job.name, len(tasks), len(expected), spec.fingerprint[:12])
    totals = {j.name: Counter() for j in jobs}
    t0 = time.perf_counter()

    def absorb(res, i):
        totals[res["source"]].update(res["stats"])
        for e in res["errors"]:
            log.warning("[%s] %s", res["source"], e)
        if i % 10 == 0 or i == len(all_tasks):
            log.info("compiled %d/%d parts  %.1fM tokens  %.0fs", i, len(all_tasks),
                     sum(t["tokens"] for t in totals.values()) / 1e6, time.perf_counter() - t0)

    if all_tasks:
        if workers > 1:
            ctx = mp.get_context("spawn")  # fork is unsafe with the Rust tokenizers library
            with ctx.Pool(min(workers, len(all_tasks)), initializer=_init_worker,
                          initargs=(spec.tokenizer, spec.template_path)) as pool:
                for i, res in enumerate(pool.imap_unordered(compile_part, all_tasks), 1):
                    absorb(res, i)
        else:
            _init_worker(spec.tokenizer, spec.template_path)
            for i, t in enumerate(all_tasks, 1):
                absorb(compile_part(t), i)
    out = {}
    for job in jobs:
        info = {"source": job.name, "tokenizer": spec.tokenizer, "fingerprint": spec.fingerprint,
                "license": job.license, "n_parts": plans[job.name][1], "compiled_now": dict(totals[job.name]),
                "seconds": round(time.perf_counter() - t0, 1)}
        (spec.source_dir(job.name) / "_source.json").write_text(json.dumps(info, indent=1))
        out[job.name] = info
    return out
