#!/usr/bin/env python3
"""
Benchmark overlap (decontamination) check for the SFT corpus written by build_sft_dataset.py.

A training sample is flagged when any of its messages (content + tool-call arguments) shares an n-gram of
`ngram` lowercased words with the text of a benchmark item (benchmarks.yaml). Benchmark items shorter than
`ngram` words but with at least `min_words` are matched as a whole. N-grams shared by more than
`max_items_per_ngram` benchmark items are treated as template text (IFEval's "your entire response should be
in ..." instructions, BFCL boilerplate) and ignored, so they do not flag every instruction-following sample.

Outputs (the corpus itself is never modified; sft_data drops flagged ids at catalog time):
  out/<group>/<source>/_contamination.parquet   one row per flagged sample
        id, shard, row, n_hits, benchmarks, items, max_item_coverage
        schema metadata: manifest_sha256 (staleness check), benchmarks_sha256, ngram
  out/decontam_stats.json, out/decontam_report.md

Commands (run from dataset/):
  python decontaminate.py --out ../out --benchmarks benchmarks.yaml
  python decontaminate.py --out ../out --only smoltalk toucan_sft --enable livecodebench
  python -m unittest test_decontaminate -v

Re-run after every rebuild of a source: sft_data refuses a _contamination.parquet whose manifest hash no
longer matches. `max_item_coverage` (share of the matched item's n-grams found in the sample) separates a
copied benchmark problem (~1.0) from one shared sentence (small); review the report before trusting either.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

log = logging.getLogger("decontam")
WORD_RE = re.compile(r"\w+")
RESULT_FILE = "_contamination.parquet"
MAX_ITEMS_LISTED = 10


class BenchmarkError(RuntimeError):
    """A benchmark could not be loaded; the check would be incomplete."""


# ----------------------------------------------------------------------------
# Benchmark text
# ----------------------------------------------------------------------------
def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def strings_under(v: Any) -> Iterator[str]:
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from strings_under(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from strings_under(x)


def get_path(row: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(row, dict):
            return None
        row = row.get(part)
    return row


def iter_json_records(path: Path) -> Iterator[dict]:
    """JSON array or JSONL (BFCL ships JSONL under a .json name)."""
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        while head and head.isspace():
            head = f.read(1)
        f.seek(0)
        if head == "[":
            yield from json.load(f)
            return
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_benchmark_rows(b: dict) -> Iterator[dict]:
    loader = b.get("loader", "hf")
    if loader == "hf":
        from datasets import load_dataset
        split = b.get("split", "test")
        if split == "all":
            for ds in load_dataset(b["repo"], b.get("config")).values():
                yield from ds
        else:
            for s in ([split] if isinstance(split, str) else split):
                yield from load_dataset(b["repo"], b.get("config"), split=s)
    elif loader == "hf_files":
        from huggingface_hub import hf_hub_download
        for fn in b["files"]:
            yield from iter_json_records(Path(hf_hub_download(b["repo"], fn, repo_type="dataset",
                                                              revision=b.get("revision"))))
    else:
        raise ValueError(f"[{b['name']}] unknown loader {loader!r}")


def benchmark_items(b: dict, rows: Iterable[dict]) -> Iterator[tuple[str, list[str]]]:
    """(item_id, [text, ...]) per benchmark row; each field string is matched separately."""
    for i, row in enumerate(rows):
        texts = [s for f in b["fields"] for s in strings_under(get_path(row, f)) if s.strip()]
        if texts:
            yield str(get_path(row, b["id_field"]) if b.get("id_field") else i), texts


# ----------------------------------------------------------------------------
# N-gram index
# ----------------------------------------------------------------------------
@dataclass
class Index:
    ngram: int
    vocab: dict[str, int] = field(default_factory=dict)
    grams: dict[int, dict[int, int]] = field(default_factory=dict)   # length -> gram hash -> item gid
    items: list[tuple[str, str]] = field(default_factory=list)      # gid -> (benchmark, item id)
    item_ngrams: list[int] = field(default_factory=list)            # gid -> number of indexed n-grams
    dropped_template: int = 0

    def ids(self, ws: list[str], grow: bool = False) -> list[int]:
        if grow:
            return [self.vocab.setdefault(w, len(self.vocab)) for w in ws]
        return [self.vocab.get(w, -1) for w in ws]


def windows(ids: list[int], length: int) -> Iterator[int]:
    """hash() of every all-in-vocabulary window (int tuples hash deterministically across processes)."""
    run = 0
    for j, x in enumerate(ids):
        run = run + 1 if x >= 0 else 0
        if run >= length:
            yield hash(tuple(ids[j - length + 1:j + 1]))


def build_index(benchmarks: list[tuple[str, Iterable[tuple[str, list[str]]]]], ngram: int,
                min_words: int, max_items_per_ngram: int) -> Index:
    idx = Index(ngram)
    owners: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for bname, items in benchmarks:
        for item_id, texts in items:
            gid = len(idx.items)
            idx.items.append((bname, item_id))
            for text in texts:
                ws = words(text)
                if len(ws) >= ngram:
                    for h in windows(idx.ids(ws, grow=True), ngram):
                        owners[ngram][h].add(gid)
                elif len(ws) >= min_words:
                    owners[len(ws)][hash(tuple(idx.ids(ws, grow=True)))].add(gid)
    idx.item_ngrams = [0] * len(idx.items)
    for length, table in owners.items():
        keep: dict[int, int] = {}
        for h, gids in table.items():
            if len(gids) > max_items_per_ngram:
                idx.dropped_template += 1
                continue
            g = min(gids)
            keep[h] = g
            for x in gids:
                idx.item_ngrams[x] += 1
        idx.grams[length] = keep
    return idx


def sample_texts(rec: dict) -> Iterator[str]:
    for m in rec.get("messages") or []:
        if isinstance(m.get("content"), str):
            yield m["content"]
        for c in m.get("tool_calls") or []:
            args = (c.get("function") or {}).get("arguments")
            if isinstance(args, str):
                yield args


def match_sample(rec: dict, idx: Index) -> Optional[dict]:
    hits = 0
    per_item: dict[int, set[int]] = defaultdict(set)
    for text in sample_texts(rec):
        ids = idx.ids(words(text))
        for length, table in idx.grams.items():
            for h in windows(ids, length):
                g = table.get(h)
                if g is not None:
                    hits += 1
                    per_item[g].add(h)
    if not hits:
        return None
    cov = {g: len(hs) / max(idx.item_ngrams[g], 1) for g, hs in per_item.items()}
    ranked = sorted(cov, key=lambda g: (-cov[g], g))
    return {"n_hits": hits,
            "benchmarks": sorted({idx.items[g][0] for g in per_item}),
            "items": [f"{idx.items[g][0]}:{idx.items[g][1]}" for g in ranked[:MAX_ITEMS_LISTED]],
            "gids": ranked,
            "max_item_coverage": round(cov[ranked[0]], 4)}


# ----------------------------------------------------------------------------
# Corpus scan (fork workers share the index read-only)
# ----------------------------------------------------------------------------
_INDEX: Optional[Index] = None


def scan_shard(path: str) -> tuple[str, int, list[dict]]:
    assert _INDEX is not None
    flagged, n = [], 0
    with open(path, encoding="utf-8") as f:
        for row, line in enumerate(f):
            if not line.strip():
                continue
            n += 1
            rec = json.loads(line)
            got = match_sample(rec, _INDEX)
            if got:
                flagged.append({"id": rec["id"], "shard": Path(path).name, "row": row, **got})
    return path, n, flagged


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_result(src_dir: Path, flagged: list[dict], meta: dict) -> None:
    flagged = sorted(flagged, key=lambda r: (r["shard"], r["row"]))
    table = pa.table({
        "id": pa.array([r["id"] for r in flagged], pa.string()),
        "shard": pa.array([r["shard"] for r in flagged], pa.string()),
        "row": pa.array([r["row"] for r in flagged], pa.int64()),
        "n_hits": pa.array([r["n_hits"] for r in flagged], pa.int64()),
        "benchmarks": pa.array([r["benchmarks"] for r in flagged], pa.list_(pa.string())),
        "items": pa.array([r["items"] for r in flagged], pa.list_(pa.string())),
        "max_item_coverage": pa.array([r["max_item_coverage"] for r in flagged], pa.float64()),
    }).replace_schema_metadata({k: str(v) for k, v in meta.items()})
    tmp = src_dir / (RESULT_FILE + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(src_dir / RESULT_FILE)


def discover_sources(out: Path, only: Optional[list[str]]) -> list[Path]:
    dirs = sorted(m.parent for m in out.glob("*/*/_manifest.json"))
    if only:
        missing = set(only) - {d.name for d in dirs}
        if missing:
            raise SystemExit(f"unknown sources: {sorted(missing)}; found {[d.name for d in dirs]}")
        dirs = [d for d in dirs if d.name in only]
    return dirs


def load_registry(path: Path, enable: list[str]) -> tuple[dict, list[dict]]:
    cfg = yaml.safe_load(path.read_text()) or {}
    benches = cfg.get("benchmarks") or []
    names = {b["name"] for b in benches}
    if set(enable) - names:
        raise SystemExit(f"--enable: unknown benchmarks {sorted(set(enable) - names)}")
    active = [b for b in benches if b.get("enabled", True) is not False or b["name"] in enable]
    for b in active:
        if not b.get("repo") or not b.get("fields"):
            raise SystemExit(f"benchmark {b.get('name')!r}: `repo` and `fields` are required")
    return cfg, active


def render_report(stats: dict) -> str:
    lines = [f"# Benchmark overlap report ({stats['run_at']})", "",
             f"n-gram {stats['ngram']}, min_words {stats['min_words']}, "
             f"template n-grams ignored: {stats['template_ngrams_ignored']:,}", "",
             "## Benchmarks (items reached by at least one training sample)", "",
             "| benchmark | items | contaminated items | % |", "|---|---:|---:|---:|"]
    for b, s in stats["benchmarks"].items():
        pct = 100 * s["items_matched"] / s["items"] if s["items"] else 0
        lines.append(f"| {b} | {s['items']:,} | {s['items_matched']:,} | {pct:.1f} |")
    bnames = list(stats["benchmarks"])
    lines += ["", "## Flagged training samples per source", "",
              "| source | samples | flagged | coverage>=0.5 | " + " | ".join(bnames) + " |",
              "|---|---:|---:|---:|" + "---:|" * len(bnames)]
    for src, s in stats["sources"].items():
        cells = " | ".join(f"{s['per_benchmark'].get(b, 0):,}" for b in bnames)
        lines.append(f"| {src} | {s['samples']:,} | {s['flagged']:,} | {s['flagged_cov50']:,} | {cells} |")
    lines += ["", "## Highest-coverage examples (review these)", ""]
    for ex in stats["examples"]:
        lines.append(f"- `{ex['id']}` coverage {ex['max_item_coverage']:.2f}, hits {ex['n_hits']}: "
                     + ", ".join(ex["items"][:3]))
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../out", help="build_sft_dataset.py output root")
    ap.add_argument("--benchmarks", default=str(Path(__file__).with_name("benchmarks.yaml")))
    ap.add_argument("--only", nargs="*", help="source names (default: every source under --out)")
    ap.add_argument("--enable", nargs="*", default=[], help="also run benchmarks marked enabled: false")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu count - 1")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    for noisy in ("httpx", "huggingface_hub", "datasets", "fsspec", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out = Path(args.out)
    reg_path = Path(args.benchmarks)
    cfg, active = load_registry(reg_path, args.enable)
    ngram, min_words = int(cfg.get("ngram", 13)), int(cfg.get("min_words", 8))
    max_items = int(cfg.get("max_items_per_ngram", 5))
    sources = discover_sources(out, args.only)

    # an incomplete check must not look like a clean one: any benchmark failure aborts before scanning
    loaded: list[tuple[str, list[tuple[str, list[str]]]]] = []
    for b in active:
        t0 = time.time()
        try:
            items = list(benchmark_items(b, load_benchmark_rows(b)))
        except Exception as e:
            raise BenchmarkError(f"benchmark {b['name']} ({b['repo']}) failed to load: {e!r}; fix it, "
                                 f"or set enabled: false in {reg_path.name} and accept that it is unchecked") from e
        if not items:
            raise BenchmarkError(f"benchmark {b['name']}: no text found under fields {b['fields']}")
        log.info("benchmark %-18s %6d items  %.1fs", b["name"], len(items), time.time() - t0)
        loaded.append((b["name"], items))

    global _INDEX
    _INDEX = build_index(loaded, ngram, min_words, max_items)
    n_grams = sum(len(t) for t in _INDEX.grams.values())
    log.info("index: %d items, %d n-grams, %d template n-grams ignored, vocab %d",
             len(_INDEX.items), n_grams, _INDEX.dropped_template, len(_INDEX.vocab))

    bench_sha = hashlib.sha256(json.dumps({"cfg": {k: v for k, v in cfg.items() if k != "benchmarks"},
                                           "active": active}, sort_keys=True, default=str).encode()).hexdigest()
    workers = args.workers or max(1, (mp.cpu_count() or 2) - 1)
    stats: dict[str, Any] = {"run_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "ngram": ngram,
                             "min_words": min_words, "template_ngrams_ignored": _INDEX.dropped_template,
                             "benchmarks_sha256": bench_sha, "benchmarks": {}, "sources": {}, "examples": []}
    matched_gids: set[int] = set()
    examples: list[dict] = []
    ctx = mp.get_context("fork")   # workers inherit the (large, read-only) index without pickling it
    with ctx.Pool(workers) as pool:
        for src in sources:
            t0 = time.time()
            shards = sorted(str(p) for p in src.glob("part-*.jsonl"))
            if not shards:
                log.warning("[%s] no part-*.jsonl shards (gzip output is not supported); skipped", src.name)
                continue
            n, flagged = 0, []
            for _, k, got in pool.imap_unordered(scan_shard, shards):
                n += k
                flagged.extend(got)
            per_b = Counter(b for r in flagged for b in r["benchmarks"])
            for r in flagged:
                matched_gids.update(r["gids"])
            examples.extend({k: r[k] for k in ("id", "n_hits", "items", "max_item_coverage")} for r in flagged)
            write_result(src, flagged, {"manifest_sha256": sha256_file(src / "_manifest.json"),
                                        "benchmarks_sha256": bench_sha, "ngram": ngram})
            stats["sources"][src.name] = {"group": src.parent.name, "samples": n, "flagged": len(flagged),
                                          "flagged_cov50": sum(r["max_item_coverage"] >= 0.5 for r in flagged),
                                          "per_benchmark": dict(per_b)}
            log.info("[%s] %d samples, %d flagged (%s)  %.0fs", src.name, n, len(flagged),
                     ", ".join(f"{b}={c}" for b, c in per_b.most_common()) or "-", time.time() - t0)
    for bname, items in loaded:
        gids = {g for g, (b, _) in enumerate(_INDEX.items) if b == bname}
        stats["benchmarks"][bname] = {"items": len(items), "items_matched": len(gids & matched_gids)}
    stats["examples"] = sorted(examples, key=lambda r: (-r["max_item_coverage"], -r["n_hits"]))[:25]

    (out / "decontam_stats.json").write_text(json.dumps(stats, indent=2))
    (out / "decontam_report.md").write_text(render_report(stats))
    log.info("wrote %s and %s", out / "decontam_stats.json", out / "decontam_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
