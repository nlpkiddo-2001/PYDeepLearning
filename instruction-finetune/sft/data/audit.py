"""
Audit training JSONL files on CPU before spending GPU time on them.

Runs every record through the same ChatTokenizer the trainer uses (no model, no GPU) and reports:
  - schema problems: non-canonical layout, duplicate ids/content, string tool-call arguments,
    tool calls with no tools defined, tool calls written as text, observations sent as user turns,
    non-JSON-Schema parameter types, unknown roles
  - rejections by reason, per-turn fallbacks (and why), and the token amplification they cause
  - sample lengths and supervised tokens: overall, per file, per task_type, per source
  - for each candidate max_seq_len: what is kept, greedy-pack fill, and optimizer steps per epoch

Usage (files, directories and globs all work; directories are searched for *.jsonl):
    CUDA_VISIBLE_DEVICES= python -m sft.data.audit data/
    CUDA_VISIBLE_DEVICES= python -m sft.data.audit data/part-*.jsonl --grad_accum_steps 4 --json runs/audit.json

--max_seq_len (default 131072) is the tokenizer limit for THIS audit. The default is large on purpose,
to measure true lengths. The per-length table is then a lower bound on what is kept: at a real limit,
an overlong record falls back to per-turn samples, and its earlier (shorter) turns survive.
Pass the exact training value to see exactly what training will get.
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

from sft.data.chat_tokenizer import ChatTokenizer, TokenizerConfig

KNOWN_ROLES = {"system", "user", "assistant", "tool"}
JSON_SCHEMA_TYPES = {"object", "string", "integer", "number", "boolean", "array", "null"}
CANDIDATE_LENS = [4096, 8192, 16384, 32768, 65536]

_chat_tok: ChatTokenizer | None = None   # one per worker process


def _init_worker(model_name: str, max_seq_len: int):
    global _chat_tok
    from transformers import AutoTokenizer
    _chat_tok = ChatTokenizer(AutoTokenizer.from_pretrained(model_name), TokenizerConfig(max_seq_len=max_seq_len))


def record_meta(record: dict) -> tuple[str, str]:
    """(task_type, source): canonical metadata.{task_type,source}, else the top-level group/source layout"""
    metadata = record.get("metadata") or {}
    task_type = metadata.get("task_type") or record.get("group") or "<none>"
    source = metadata.get("source") or record.get("source") or "<none>"
    return str(task_type), str(source)


def _param_types(schema, out: set):
    """collect every `type` value in a tool's parameter schema (recursively)"""
    if isinstance(schema, dict):
        t = schema.get("type")
        if isinstance(t, str):
            out.add(t)
        for v in schema.values():
            _param_types(v, out)
    elif isinstance(schema, list):
        for v in schema:
            _param_types(v, out)


def _schema_issues(record: dict) -> list[str]:
    issues = []
    if not record.get("id"):
        issues.append("missing_id")
    if "metadata" not in record:
        issues.append("non_canonical_layout")          # top-level source/group instead of metadata.{...}
    if record_meta(record)[0] == "<none>":
        issues.append("missing_task_type")

    messages = record.get("messages") or []
    if not messages:
        issues.append("no_messages")

    has_calls = text_calls = obs_as_user = string_args = False
    for m in messages:
        role = m.get("role")
        if role not in KNOWN_ROLES:
            issues.append(f"unknown_role:{role}")
        content = m.get("content") if isinstance(m.get("content"), str) else ""
        if role == "assistant" and "<function=" in content:
            text_calls = True
        if role == "user" and content.startswith("OBSERVATION:"):
            obs_as_user = True
        for call in m.get("tool_calls") or []:
            has_calls = True
            if isinstance((call.get("function") or {}).get("arguments"), str):
                string_args = True

    if string_args:
        issues.append("tool_args_are_string")           # the agreed format is a JSON object
    if has_calls and not record.get("tools"):
        issues.append("tool_calls_but_no_tools")       # the model never sees the schemas it calls
    if text_calls:
        issues.append("tool_calls_written_as_text")    # <function=...> inside assistant content
    if obs_as_user:
        issues.append("observation_as_user_turn")      # tool output sent as role=user

    types = set()
    for tool in record.get("tools") or []:
        _param_types((tool.get("function") or {}).get("parameters"), types)
    if types - JSON_SCHEMA_TYPES:
        issues.append("non_json_schema_param_types")   # e.g. "dict"/"int" (Python names) instead of "object"/"integer"
    return issues


def _bad_param_types(record: dict) -> list[str]:
    types = set()
    for tool in record.get("tools") or []:
        _param_types((tool.get("function") or {}).get("parameters"), types)
    return sorted(types - JSON_SCHEMA_TYPES)


def _audit_line(item: tuple[str, int, str]) -> dict:
    """Runs in a worker. Returns only small numbers, never token ids."""
    file, line_no, line = item
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        return {"file": file, "line_no": line_no, "parse_error": str(exc)[:200]}

    task_type, source = record_meta(record)
    res = _chat_tok(record)
    return {
        "file": file,
        "line_no": line_no,
        "id": record.get("id"),
        "content_hash": (record.get("meta") or {}).get("content_hash"),
        "task_type": task_type,
        "source": source,
        "issues": _schema_issues(record),
        "bad_param_types": _bad_param_types(record),
        "n_user": sum(m.get("role") == "user" for m in record.get("messages") or []),
        "n_assistant": sum(m.get("role") == "assistant" for m in record.get("messages") or []),
        "mode": res.mode,
        "fallback_reason": res.fallback_reason,
        "rejections": [r.reason.value for r in res.rejections],
        "samples": [(len(s.input_ids), s.num_supervised) for s in res.samples],
    }


def expand_paths(paths: list[str]) -> list[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
        elif any(ch in p for ch in "*?["):
            files += sorted(glob.glob(p, recursive=True))
        else:
            files.append(p)
    seen = set()
    return [f for f in files if not (f in seen or seen.add(f))]


def _items(files: list[str]):
    for file in files:
        with open(file) as f:
            for i, line in enumerate(f):
                if line.strip():
                    yield file, i, line


def audit(files: list[str], model_name: str, max_seq_len: int, workers: int) -> list[dict]:
    total = sum(1 for _ in _items(files))
    results = []
    with Pool(workers, initializer=_init_worker, initargs=(model_name, max_seq_len)) as pool:
        # imap keeps file order (the packing estimate depends on it); chunks keep the pipes busy
        for r in pool.imap(_audit_line, _items(files), chunksize=16):
            results.append(r)
            if len(results) % 1000 == 0 or len(results) == total:
                print(f"\r  tokenized {len(results)}/{total} records", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return results


def _pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def _greedy_rows(lengths: list[int], pack_len: int) -> tuple[int, float]:
    """same rule as pack_greedy: start a new row when the next sample doesn't fit. Returns (rows, fill)."""
    rows, current, used = 0, 0, 0
    for n in lengths:
        if current + n > pack_len:
            rows += 1
            current = 0
        current += n
        used += n
    if current:
        rows += 1
    return rows, (used / (rows * pack_len) if rows else 0.0)


def _group_table(title: str, results: list[dict], key: str):
    table = defaultdict(Counter)
    for r in results:
        t = table[r[key]]
        t["records"] += 1
        t["samples"] += len(r["samples"])
        t["tokens"] += sum(n for n, _ in r["samples"])
        t["supervised"] += sum(s for _, s in r["samples"])
        t["no_samples"] += not r["samples"]
        t["per_turn"] += r["mode"] == "per_turn"
    total_sup = max(1, sum(t["supervised"] for t in table.values()))
    print(f"\n=== by {title}")
    print(f"  {'':28s} {'records':>8s} {'per-turn':>8s} {'samples':>8s} {'tokens':>14s} {'supervised':>13s} {'share':>6s} {'no sample':>9s}")
    for k, t in sorted(table.items(), key=lambda kv: -kv[1]["supervised"]):
        print(f"  {str(k)[-28:]:28s} {t['records']:8d} {t['per_turn']:8d} {t['samples']:8d} {t['tokens']:14,d} "
              f"{t['supervised']:13,d} {t['supervised'] / total_sup:6.1%} {t['no_samples']:9d}")
    return {k: dict(v) for k, v in table.items()}


def report(results: list[dict], world_size: int, grad_accum_steps: int) -> dict:
    parsed = [r for r in results if "parse_error" not in r]
    parse_errors = [r for r in results if "parse_error" in r]

    issues = Counter(i for r in parsed for i in r["issues"])
    ids = Counter(r["id"] for r in parsed if r["id"])
    duplicate_ids = {k: v for k, v in ids.items() if v > 1}
    hashes = Counter(r["content_hash"] for r in parsed if r["content_hash"])
    duplicate_content = sum(v - 1 for v in hashes.values() if v > 1)
    rejections = Counter(x for r in parsed for x in r["rejections"])
    modes = Counter(r["mode"] or "rejected" for r in parsed)
    fallbacks = Counter(r["fallback_reason"] for r in parsed if r["mode"] == "per_turn" and r["fallback_reason"])
    no_sample_records = sum(1 for r in parsed if not r["samples"])

    lengths = [n for r in parsed for n, _ in r["samples"]]
    supervised = [s for r in parsed for _, s in r["samples"]]

    # per-turn repeats the context once per assistant turn. Amplification = tokens trained on /
    # tokens of the conversation itself (its longest sample = the whole conversation up to the last turn)
    conv_tokens = sum(max(n for n, _ in r["samples"]) for r in parsed if r["samples"])
    amplification = sum(lengths) / max(1, conv_tokens)

    print(f"\n=== {len(results)} lines: {len(parsed)} parsed, {len(parse_errors)} JSON errors")
    for r in parse_errors[:5]:
        print(f"  {r['file']}:{r['line_no']}: {r['parse_error']}")

    print("\n=== schema  (records affected)")
    print(f"  duplicate ids: {len(duplicate_ids)}" + (f"  e.g. {list(duplicate_ids)[:5]}" if duplicate_ids else ""))
    if hashes:
        print(f"  duplicate content (meta.content_hash): {duplicate_content}")
    for k, v in issues.most_common():
        print(f"  {k}: {v} ({v / max(1, len(parsed)):.1%})")
    bad_types = Counter(t for r in parsed for t in r["bad_param_types"])
    if bad_types:
        print(f"    non-JSON-Schema types seen (records): {dict(bad_types)}")

    print("\n=== tokenizer")
    print(f"  records with no usable sample: {no_sample_records} ({no_sample_records / max(1, len(parsed)):.1%})")
    print(f"  modes: {dict(modes)}")
    if fallbacks:
        print("  per-turn fallback reasons:")
        for k, v in fallbacks.most_common(8):
            print(f"    {v:7d}  {k}")
    print(f"  token amplification from per-turn: {amplification:.2f}x  "
          f"({sum(lengths):,} tokens trained for {conv_tokens:,} tokens of conversation)")
    print(f"  rejections: {dict(rejections) or 'none'}")

    print(f"\n=== samples: {len(lengths)}  tokens: {sum(lengths):,}  supervised: {sum(supervised):,} "
          f"({sum(supervised) / max(1, sum(lengths)):.1%} of tokens)")
    for name, vals in (("length", lengths), ("supervised", supervised)):
        print(f"  {name:10s} p50 {_pct(vals, .5):7d}  p90 {_pct(vals, .9):7d}  p99 {_pct(vals, .99):7d}  max {max(vals, default=0):7d}")

    by_file = _group_table("file", parsed, "file")
    by_task = _group_table("task_type", parsed, "task_type")
    by_source = _group_table("source", parsed, "source")

    print(f"\n=== per max_seq_len  (steps/epoch for {world_size} GPUs x grad_accum {grad_accum_steps}; "
          f"kept = lower bound, see --help)")
    print(f"  {'max_seq_len':>11s} {'samples kept':>13s} {'supervised kept':>16s} {'rows':>9s} {'fill':>6s} {'steps/epoch':>12s}")
    per_len = {}
    for L in CANDIDATE_LENS:
        kept = [(n, s) for n, s in zip(lengths, supervised) if n <= L]
        rows, fill = _greedy_rows([n for n, _ in kept], L)
        steps = math.ceil(rows / (world_size * grad_accum_steps)) if rows else 0
        sup_kept = sum(s for _, s in kept) / max(1, sum(supervised))
        per_len[L] = {"samples_kept": len(kept), "supervised_kept": sup_kept, "rows": rows, "fill": fill, "steps_per_epoch": steps}
        print(f"  {L:11d} {len(kept) / max(1, len(lengths)):13.1%} {sup_kept:16.1%} {rows:9d} {fill:6.1%} {steps:12d}")

    return {
        "lines": len(results), "parse_errors": len(parse_errors), "duplicate_ids": len(duplicate_ids),
        "duplicate_content": duplicate_content, "schema_issues": dict(issues),
        "no_sample_records": no_sample_records, "modes": dict(modes), "fallback_reasons": dict(fallbacks),
        "amplification": amplification, "rejections": dict(rejections),
        "samples": len(lengths), "tokens": sum(lengths), "supervised": sum(supervised),
        "by_file": by_file, "by_task_type": by_task, "by_source": by_source, "per_max_seq_len": per_len,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="JSONL files, directories (searched for *.jsonl) or globs")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B", help="whose tokenizer and chat template")
    p.add_argument("--max_seq_len", type=int, default=131072)
    p.add_argument("--world_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    p.add_argument("--json", help="also write the summary here")
    args = p.parse_args()

    files = expand_paths(args.paths)
    if not files:
        sys.exit(f"no .jsonl files found in {args.paths}")
    print(f"auditing {len(files)} file(s) with {args.workers} workers: {files}", file=sys.stderr)

    results = audit(files, args.model_name, args.max_seq_len, args.workers)
    summary = report(results, args.world_size, args.grad_accum_steps)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nsummary written to {args.json}")


if __name__ == "__main__":
    main()
