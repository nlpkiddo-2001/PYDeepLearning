"""Source discovery, eligibility filters, deterministic train/val split."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

from dataset.sft.loader.compile import CacheSpec, list_shards
from dataset.sft.loader.config import ConfigError, MixConfig, SourceCfg
from dataset.sft.loader.store import CacheError, SourceStore
from dataset.sft.loader.turns import truncate_at_turn

log = logging.getLogger("sft_data")
_MOD = 1 << 48
CONTAMINATION_FILE = "_contamination.parquet"   # written by dataset/decontaminate.py


@dataclass
class SourceInfo:
    name: str
    group: str
    path: Path
    license: Optional[str]
    n_samples: int
    shard_rows: dict[str, int] = field(default_factory=dict)


def discover(data_root: str | Path) -> dict[str, SourceInfo]:
    """Sources produced by build_sft_dataset.py: <root>/<group>/<source>/{_manifest.json,_index.parquet,part-*.jsonl}."""
    out: dict[str, SourceInfo] = {}
    for man in sorted(Path(data_root).glob("*/*/_manifest.json")):
        d = man.parent
        info = json.loads(man.read_text())
        lic = None
        idx = d / "_index.parquet"
        if idx.exists():
            f = pq.ParquetFile(idx)
            if "license" in f.schema_arrow.names and f.metadata.num_row_groups:
                col = f.read_row_group(0, columns=["license"]).column(0)
                lic = col[0].as_py() if len(col) else None
        rows = {sh["file"]: int(sh["n_samples"]) for sh in info.get("shards", [])}
        out[d.name] = SourceInfo(d.name, d.parent.name, d, lic, int(info.get("n_samples", 0)), rows)
    return out


def contaminated_ids(info: SourceInfo, cfg: MixConfig) -> Optional[set[str]]:
    """Ids flagged by decontaminate.py, or None when no check applies (decontam: off / if_present + missing)."""
    if cfg.decontam == "off":
        return None
    path = info.path / CONTAMINATION_FILE
    if not path.exists():
        if cfg.decontam == "required":
            raise ConfigError(f"[{info.name}] no {CONTAMINATION_FILE}; run dataset/decontaminate.py "
                              f"(or set decontam: if_present / off)")
        log.warning("[%s] no benchmark overlap check found (%s); training on unchecked data", info.name,
                    CONTAMINATION_FILE)
        return None
    pf = pq.ParquetFile(path)
    meta = {k.decode(): v.decode() for k, v in (pf.schema_arrow.metadata or {}).items()}
    if meta.get("manifest_sha256") != hashlib.sha256((info.path / "_manifest.json").read_bytes()).hexdigest():
        raise ConfigError(f"[{info.name}] {CONTAMINATION_FILE} predates the current build; rerun dataset/decontaminate.py")
    t = pf.read(columns=["id", "max_item_coverage"]).to_pandas()
    return set(t.id[t.max_item_coverage >= cfg.decontam_min_coverage])


def split_scores(content_hashes, ids, split_seed: int) -> np.ndarray:
    """Uniform [0, 1) score per sample, a pure function of its content hash (stable across mixes/builds)."""
    salt = (split_seed * 0x9E3779B97F4A7C15) % _MOD
    vals = np.empty(len(ids), dtype=np.int64)
    for i, (h, sid) in enumerate(zip(content_hashes, ids)):
        key = h if len(h) >= 12 else __import__("hashlib").blake2b(str(sid).encode(), digest_size=8).hexdigest()
        vals[i] = (int(key[:12], 16) + salt) % _MOD
    return vals / _MOD


@dataclass
class SourceView:
    name: str
    domain: str
    cfg: SourceCfg
    store: SourceStore
    train_idx: np.ndarray               # int64 indices into the store (eligible, not validation)
    val_idx: np.ndarray
    eff_len: np.ndarray                 # kept length per train doc (after the over-length cut) == cut[train_idx]
    cut: Optional[np.ndarray] = None    # kept length per STORE index (n_tokens, or the turn-boundary cut if longer than seq_len)
    license: Optional[str] = None
    group: str = ""
    n_total: int = 0
    n_rejected: dict[str, int] = field(default_factory=dict)

    @property
    def train_tokens(self) -> int:
        return int(self.eff_len.sum())


@dataclass
class Catalog:
    views: list[SourceView]
    skipped: dict[str, str]

    def by_name(self) -> dict[str, SourceView]:
        return {v.name: v for v in self.views}

    def available_tokens(self) -> dict[str, float]:
        return {v.name: float(v.train_tokens) for v in self.views}


def verify_cache(info: SourceInfo, spec: CacheSpec) -> None:
    """The cache must cover every row of every shard, built from the shard's current bytes."""
    d = spec.source_dir(info.name)
    if not d.exists():
        raise CacheError(f"[{info.name}] not compiled for tokenizer fingerprint {spec.fingerprint[:12]} "
                         f"(run `python -m sft_data.cli compile`)")
    rows: dict[str, int] = {}
    sizes: dict[str, set] = {}
    for f in d.glob("part-*.done.json"):
        j = json.loads(f.read_text())
        base = f.name.split(".")[0].rsplit("-c", 1)[0]
        rows[base] = rows.get(base, 0) + j["n_samples"]
        sizes.setdefault(base, set()).add(j["input_size"])
    for sh in list_shards(info.path):
        base = sh.name.split(".")[0]
        want = info.shard_rows.get(sh.name)
        if base not in rows or sizes[base] != {sh.stat().st_size} or (want is not None and rows[base] != want):
            raise CacheError(f"[{info.name}] cache is stale or incomplete at {sh.name}; rerun compile")


def build_catalog(cfg: MixConfig, spec: CacheSpec, discovered: Optional[dict[str, SourceInfo]] = None) -> Catalog:
    discovered = discovered if discovered is not None else discover(cfg.data_root)
    deny = [s.lower() for s in cfg.license_deny]
    views: list[SourceView] = []
    skipped: dict[str, str] = {}
    for dom, scfg in cfg.enabled_sources():
        info = discovered.get(scfg.name)
        if info is None:
            raise ConfigError(f"source {scfg.name!r} not found under {cfg.data_root!r}; found: {sorted(discovered)}")
        if info.license and any(d in info.license.lower() for d in deny):
            skipped[scfg.name] = f"license {info.license!r} matches license_deny"
            log.warning("[%s] skipped: %s", scfg.name, skipped[scfg.name])
            continue
        verify_cache(info, spec)
        flagged = contaminated_ids(info, cfg)
        store = SourceStore(spec.source_dir(scfg.name))
        m = store.meta
        rejected: dict[str, int] = {}

        def reject(label: str, bad):
            n = int((bad & ok).sum())
            if n:
                rejected[label] = n
            return ok & ~bad

        ok = np.ones(len(m), dtype=bool)
        ok = reject("render_failed", (m.status != 0).to_numpy())
        ok = reject("too_few_trainable_tokens", (m.n_trainable < scfg.min_trainable_tokens).to_numpy())
        if flagged:
            ok = reject("benchmark_overlap", m.id.isin(flagged).to_numpy())
        if scfg.require_tool_valid:
            ok = reject("tool_invalid", (~m.tool_valid).to_numpy())
        if scfg.max_tokens:
            ok = reject("over_max_tokens", (m.n_tokens > scfg.max_tokens).to_numpy())
        if cfg.over_length == "drop":
            ok = reject("over_seq_len", (m.n_tokens > cfg.seq_len).to_numpy())
        if scfg.langs:
            ok = reject("lang_filtered", (~m.lang.isin(scfg.langs)).to_numpy())
        # exact over-length handling: cut at the last complete assistant turn that fits, or reject the sample here,
        # so the sampler's token accounting equals what the packer will really emit
        n_tok = m.n_tokens.to_numpy()
        cut = n_tok.copy()
        too_long = np.flatnonzero(ok & (n_tok > cfg.seq_len))
        no_fit = np.zeros(len(m), dtype=bool)
        for i in too_long:
            c = truncate_at_turn(np.asarray(store.get(int(i))[1]), cfg.seq_len, cfg.min_truncated_trainable)
            if c is None:
                no_fit[i] = True
            else:
                cut[i] = c
        ok = reject("no_assistant_turn_fits_seq_len", no_fit)
        # membership of the validation split depends on the content hash only (never on filters or the mix)
        scores = split_scores(m.content_hash.tolist(), m.id.tolist(), cfg.split_seed)
        in_val = scores < cfg.val_fraction
        if in_val.sum() > cfg.val_max_samples_per_source:
            cand = np.flatnonzero(in_val)
            in_val = np.zeros(len(m), dtype=bool)
            in_val[cand[np.argsort(scores[cand], kind="stable")[:cfg.val_max_samples_per_source]]] = True
        train_idx = np.flatnonzero(ok & ~in_val).astype(np.int64)
        val_idx = np.flatnonzero(ok & in_val).astype(np.int64)
        if len(train_idx) == 0:
            skipped[scfg.name] = f"no eligible training samples after filters ({rejected})"
            log.warning("[%s] skipped: %s", scfg.name, skipped[scfg.name])
            continue
        views.append(SourceView(scfg.name, dom.name, scfg, store, train_idx, val_idx, cut[train_idx].astype(np.int64),
                                cut, info.license, info.group, len(m), rejected))
    if not views:
        raise ConfigError("no usable sources after filters")
    return Catalog(views, skipped)
