"""python -m sft_data.cli {compile,stats,preview,validate} --config mix.yaml"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time

import numpy as np
import pandas as pd

from dataset.sft.loader.catalog import build_catalog, discover
from dataset.sft.loader.compile import CompileJob, compile_sources, list_shards, make_cache_spec, open_shard
from dataset.sft.loader.config import MixConfig
from dataset.sft.loader.loader import PackedStream
from dataset.sft.loader.render import Renderer
from dataset.sft.loader.sampler import simulate

log = logging.getLogger("sft_data")
COLOR = sys.stdout.isatty()


def c(txt, code):
    return f"\033[{code}m{txt}\033[0m" if COLOR else str(txt)


def fmt_tok(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def cmd_compile(args) -> int:
    cfg = MixConfig.from_yaml(args.config)
    spec, _, _ = make_cache_spec(cfg.tokenizer, cfg.chat_template, cfg.cache_dir)
    found = discover(cfg.data_root)
    known = {s.name for _, s in cfg.enabled_sources()}
    unknown = sorted(set(args.sources or []) - known)
    if unknown:
        print(c(f"--sources not in the config's enabled sources: {unknown}; config has {sorted(known)}", "31"))
        return 2
    wanted = [s.name for _, s in cfg.enabled_sources() if not args.sources or s.name in args.sources]
    for name in wanted:
        if name not in found:
            print(c(f"source {name!r} not found under {cfg.data_root!r}", "31"))
            return 2
    print(f"cache: {spec.root}")
    jobs = [CompileJob(n, found[n].path, found[n].license, found[n].shard_rows) for n in wanted]
    infos = compile_sources(jobs, spec, workers=args.workers, force=args.force, chunk_rows=args.chunk_rows)
    for name in wanted:
        done = infos[name]["compiled_now"]
        bad = {k: v for k, v in done.items() if k not in ("ok", "tokens", "misaligned")}
        print(f"{c(name, '1;36'):30s} compiled {fmt_tok(done.get('tokens', 0))} tokens"
              + (c(f"   not renderable: {bad}", "33") if bad else "") + (f"   misaligned spans: {done['misaligned']}" if done.get("misaligned") else ""))
    return 0


def _mix_table(cfg, catalog, probs) -> pd.DataFrame:
    rows = []
    for i, v in enumerate(catalog.views):
        m = v.store.meta
        tr = m.n_trainable.to_numpy()[v.train_idx].sum()
        full = m.n_tokens.to_numpy()[v.train_idx].sum()
        rows.append({"source": v.name, "domain": v.domain, "docs": len(v.train_idx), "val": len(v.val_idx),
                     "tokens": fmt_tok(v.train_tokens), f"kept@{cfg.seq_len}%": round(100 * v.train_tokens / max(full, 1)),
                     "trainable%": round(100 * tr / max(full, 1)),
                     "target%": round(100 * probs[i], 1),
                     "rejected": ", ".join(f"{k}={n}" for k, n in v.n_rejected.items())})
    return pd.DataFrame(rows)


def cmd_stats(args) -> int:
    cfg = MixConfig.from_yaml(args.config)
    stream = PackedStream.from_config(cfg, 0, args.world_size)
    cat, views = stream.catalog, stream.views
    probs = stream.sampler.stage_probs[0]
    print(c("== sources (after filters) ==", "1"))
    print(_mix_table(cfg, cat, probs).to_string(index=False))
    for k, v in cat.skipped.items():
        print(c(f"skipped {k}: {v}", "33"))
    print(c("\n== how much of each source survives at a given sequence length (est.) ==", "1"))
    lens = [2048, 4096, 8192, 16384, 32768, 65536]
    rows = []
    for v in views:
        n = v.store.n_tokens[v.train_idx]
        row = {"source": v.name}
        for L in lens:
            row[f"{L // 1024}K docs/tok"] = f"{100 * (n <= L).mean():.0f}/{100 * np.minimum(n, L).sum() / n.sum():.0f}%"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"(current seq_len={cfg.seq_len}, over_length={cfg.over_length}: 'docs' = share of whole documents, 'tok' = share of tokens kept)")
    if len(cfg.stages) > 1:
        print(c("\n== stages ==", "1"))
        for i, st in enumerate(cfg.stages):
            top = sorted(zip([v.name for v in views], stream.sampler.stage_probs[i]), key=lambda x: -x[1])[:6]
            print(f"  {st.name}: until={st.until_tokens} " + ", ".join(f"{n}={100 * p:.1f}%" for n, p in top if p > 0))
    if args.budget_tokens:
        sim = simulate(cfg, views, args.budget_tokens, args.world_size)
        tot = sim["tokens_per_rank"].sum()
        rows = [{"source": v.name, "tokens": fmt_tok(sim["tokens_per_rank"][i] * args.world_size),
                 "share%": round(100 * sim["tokens_per_rank"][i] / max(tot, 1), 1),
                 "target%": round(100 * probs[i], 1), "epochs": round(sim["epochs"][v.name], 2)}
                for i, v in enumerate(views)]
        print(c(f"\n== simulated budget {fmt_tok(args.budget_tokens)} tokens, world_size={args.world_size} ==", "1"))
        print(pd.DataFrame(rows).to_string(index=False))
        dom: dict[str, float] = {}
        for i, v in enumerate(views):
            dom[v.domain] = dom.get(v.domain, 0) + sim["tokens_per_rank"][i] / max(tot, 1)
        print("domains: " + "  ".join(f"{k}={100 * x:.1f}%" for k, x in dom.items()))
        if not sim["reached"]:
            print(c(f"!! budget NOT reachable: sources run out after {fmt_tok(sim['achieved_tokens'])} tokens "
                    f"(raise max_epochs or add data)", "31"))
    if args.probe_tokens:
        t0 = time.perf_counter()
        n = 0
        for _ in range(10 ** 9):
            b = stream.next_batch()
            if b is None:
                break
            n += b["n_real"]
            if n >= args.probe_tokens:
                break
        dt = time.perf_counter() - t0
        snap = stream.metrics.snapshot()
        print(c(f"\n== probe: {fmt_tok(n)} real tokens in {dt:.1f}s ({n / dt / 1e3:.0f}K tok/s, single process) ==", "1"))
        print(f"fill_rate={snap['fill_rate']:.3f}  docs/pack={snap['docs_per_pack']:.1f}  trainable={snap['trainable_fraction']:.3f}")
        print("domain shares: " + "  ".join(f"{k}={100 * x:.1f}%" for k, x in snap["domain_token_share"].items()))
        if snap["truncated_docs"] or snap["dropped_overlength"]:
            print(f"truncated docs: {snap['truncated_docs']}  dropped (too long): {snap['dropped_overlength']}")
    return 0


def cmd_preview(args) -> int:
    cfg = MixConfig.from_yaml(args.config)
    stream = PackedStream.from_config(cfg, 0, 1)
    r = Renderer(stream.tokenizer)
    for _ in range(args.n):
        pack = stream.packer.next_pack()
        print(c(f"=== pack: {pack.n_real}/{cfg.seq_len} tokens, {len(pack.doc_len)} docs, "
                f"{pack.n_trainable} trained ===", "1;36"))
        o = 0
        for d, (s, n) in enumerate(zip(pack.doc_src, pack.doc_len)):
            ids, lab = pack.input_ids[o:o + n], pack.labels[o:o + n]
            txt = r.decode_with_mask(ids, (lab != -100).astype(np.uint8)) if COLOR else r.tok.decode(ids)
            print(c(f"--- doc {d}: {stream.views[s].name}, {n} tokens (green = trained) ---", "33"))
            print(txt[:args.chars] + (c(f"\n[... {len(txt) - args.chars} more chars]", "2") if len(txt) > args.chars else ""))
            o += n
    return 0


def cmd_validate(args) -> int:
    cfg = MixConfig.from_yaml(args.config)
    spec, tok, template = make_cache_spec(cfg.tokenizer, cfg.chat_template, cfg.cache_dir)
    found = discover(cfg.data_root)
    catalog = build_catalog(cfg, spec, found)
    renderer, rng, failures = Renderer(tok), random.Random(0), 0
    vocab = len(tok)
    for v in catalog.views:
        info = found[v.name]
        picks = rng.sample(range(v.store.n), min(args.samples, v.store.n))
        bad = 0
        for i in picks:
            ids, mask = v.store.get(i)
            stem, row = v.store.locate(i)
            with open_shard(info.path / next(p.name for p in list_shards(info.path) if p.name.split('.')[0] == stem)) as f:
                for k, line in enumerate(f):
                    if k == row:
                        rec = json.loads(line)
                        break
            fresh = renderer.render(rec)
            same = (fresh.status == v.store.meta.status.iat[i] and np.array_equal(fresh.input_ids, ids)
                    and np.array_equal(fresh.loss_mask, mask) and int(mask.sum()) == v.store.meta.n_trainable.iat[i]
                    and (len(ids) == 0 or int(ids.max()) < vocab))
            bad += not same
        print(f"{'PASS' if not bad else c('FAIL', '31')}  {v.name:24s} {len(picks)} cached samples == fresh render" + (f" ({bad} differ)" if bad else ""))
        failures += bad
    stream = PackedStream.from_config(cfg, 0, 1)
    ok = True
    for k, b in enumerate(stream):
        for row in range(b["input_ids"].shape[0]):
            ids, lab, pos, doc = (b[x][row] for x in ("input_ids", "labels", "position_ids", "doc_ids"))
            real = doc >= 0
            starts = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]] & real)
            ok &= bool((pos[starts] == 0).all()) and bool((lab[starts] == -100).all())
            ok &= bool(((lab == -100) | (lab == ids)).all()) and bool((lab[~real] == -100).all())
            ok &= bool((doc[real] == np.sort(doc[real])).all())
        if k >= 20:
            break
    print(f"{'PASS' if ok else c('FAIL', '31')}  packed-batch invariants (position reset, boundary masking, padding)")
    return 0 if (failures == 0 and ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sft_data.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("compile", "stats", "preview", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "compile":
            p.add_argument("--sources", nargs="*")
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--force", action="store_true")
            p.add_argument("--chunk-rows", type=int, default=2000, help="rows per compile task (parallelism unit)")
        if name == "stats":
            p.add_argument("--budget-tokens", type=float, default=0)
            p.add_argument("--world-size", type=int, default=1)
            p.add_argument("--probe-tokens", type=float, default=5e6)
        if name == "preview":
            p.add_argument("--n", type=int, default=1)
            p.add_argument("--chars", type=int, default=2500)
        if name == "validate":
            p.add_argument("--samples", type=int, default=200)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return {"compile": cmd_compile, "stats": cmd_stats, "preview": cmd_preview, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
