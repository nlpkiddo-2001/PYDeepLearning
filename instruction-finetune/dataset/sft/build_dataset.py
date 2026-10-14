#!/usr/bin/env python3
"""
build_sft_dataset.py
====================
Download -> normalize -> filter -> tokenize -> language-tag -> dedup -> save.

Every source dataset is converted to ONE common schema (OpenAI-style chat + tools):

{
  "id": "toucan_sft:123",
  "source": "toucan_sft",
  "group": "tool_calling",
  "messages": [
     {"role": "system"|"user"|"assistant"|"tool",
      "content": "...",
      "tool_calls": [{"id": "call_1", "type": "function",
                      "function": {"name": "...", "arguments": "<JSON string>"}}],   # assistant only
      "tool_call_id": "call_1", "name": "..."}                                       # tool only
  ],
  "tools": [{"type": "function", "function": {"name", "description", "parameters"}}],
  "meta": { tokens_*, lang*, n_*_turns, content_hash, license, extra(json str), ... }
}

Output layout (weighting / token capping is the DATALOADER's job, not done here)
-------------------------------------------------------------------------------
  out/<group>/<dataset>/part-00000.jsonl      # 10,000 samples per shard (--shard-size)
  out/<group>/<dataset>/_index.parquet        # one row per sample: tokens, lang, shard, row, hashes...
  out/<group>/<dataset>/_manifest.json        # per-shard sample / token counts + dataset totals
  out/build_stats.json, out/report.md

Commands
--------
  python build_sft_dataset.py inspect --config datasets.yaml --only hermes_fc --n 3
  python build_sft_dataset.py wherecheck --config datasets.yaml --only toucan_sft --sample 2000
  python build_sft_dataset.py build   --config datasets.yaml --out out
  python build_sft_dataset.py report  --out out
  python -m unittest discover -s . -p 'test_build_sft_dataset.py' -v

Performance and large sources
-----------------------------
  --workers N              worker processes for normalize / tokenize / hash / MinHash / language
                           tagging / JSON serialization (default 0 = auto: min(cpu-1, 12);
                           1 = single process). The parent reads, applies `where`, runs the global
                           dedup and writes shards IN ORDER, so output is byte-identical for any N.
  --batch-size 512         rows per work chunk (tokenizer batch + unit of parallel work; default 256)
  --index-batch-size 10000 Parquet rows buffered in memory; no full index DataFrame
  --no-near-dedup          opt-in faster exact-only dedup; changes retained data
  --no-streaming           opt-in disk-backed HF loading for cached large sources
  --color auto|always|never  colored terminal output (auto = TTY only; NO_COLOR / FORCE_COLOR honored)

Workers are started once per build (spawn; each loads its own tokenizer) and at most
3 x workers chunks are in flight, so memory stays bounded on 5M-row sources. Row streaming
and global dedup remain single-threaded in the parent; if `wait_workers` in timings_seconds is
~0 while `load`/`dedup`/`write` dominate, more workers will not help (try --no-streaming).
Dedup state is transactional per dataset: a dataset that fails midway is rolled back, so it
cannot make later datasets drop duplicates of samples that were never written, and its
partial shards are removed.

Build statistics include timings_seconds: parent wall-clock (load, filter, wait_workers, dedup,
write) and summed worker CPU (cpu_normalize, cpu_tokenize, cpu_hash, cpu_language, cpu_serialize).
Token counts retain the existing per-field estimate
plus --msg-overhead; they are NOT exact rendered-chat-template lengths.
Use the real target tokenizer, not "approx", for production token accounting.
The index is bounded-memory, but global exact/near dedup state still grows with
the number of unique samples. Reports load only seven columns, one source at a time.

Optional registry fields
------------------------
  streaming: false        use HF's disk-backed Arrow cache instead of streaming;
                          costs an initial download/prepare pass, useful on reruns
  data_files: {train: ["path/to/train.jsonl"]}
                          select compatible training files in mixed-schema repos
  user_field / assistant_field
                          explicit text pair fields for the generic adapter
  revision: <commit SHA>  reproducible source version (also used for split discovery)
  adapter: function_tags  trajectories whose tools/calls/results are plain text (SWE-smith xml format):
                          "---- BEGIN FUNCTION #n: name ----" blocks in the system prompt -> `tools`
                          (the section and its call-format instructions are removed from the prompt);
                          <function=name><parameter=k>v</parameter></function> in assistant text ->
                          structured tool_calls (values typed by the declared schema; the newline pair
                          wrapping multi-line values is not part of the value); the user reply right after
                          a call -> role "tool". A call with an unknown/missing/mistyped parameter makes the
                          sample count as invalid_tools.
  default_tools: swe_agent  tool schemas for rows that ship none (SWE-smith native rows use the same three tools)
  drop_if_last_assistant_matches: <regex>
                          drop samples whose final assistant message (no tool call) matches, counted as
                          drop_exit_artifact (harness status lines such as "Exit due to ...")
  `tool_call_ids: [id]` on tool messages is honored for linking results to calls.

Examples of corrections for sources in the original registry:
  SWE-smith: splits: [tool] (the repo has tool/xml/ticks, not train)
  OpenCharacter Synthetic-Dialogue: user_field: question,
      assistant_field: character_answer, system_from: "{character}"
  RoleBench: inspect and select only the desired compatible training data_files;
      do not combine profiles, statistics, training and benchmark test files.
Missing local JSONL paths are explicit failures; disabled sources are skipped.
An adapter rejecting every row is an explicit schema error, not a successful build.

IDs now use source:<raw-row-index>, unique even when upstream IDs repeat across
configs/splits; the original ID is retained in meta.orig_id. Exact content hashes
include canonicalized tool schemas; near dedup compares only within the same tool
schema. Manifests identify the new hash version and ID method. Rebuild all selected
sources in YAML priority order for consistent dedup.
Separate --only invocations do not deduplicate against previously written output.
Do not start concurrent builds targeting the same output directory. A running
Python process will not pick up edits; these changes apply to the next invocation.

Dedup priority = order of datasets in the YAML (earlier source wins a duplicate).
"""
from __future__ import annotations

import argparse
import ast
import copy
import glob
import gzip
import hashlib
import json
import keyword
import logging
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

log = logging.getLogger("sftbuild")
HF_TOKEN = os.environ.get("HF_TOKEN")

# ----------------------------------------------------------------------------
# Terminal colors (ANSI; no dependency). Disabled automatically when output is not a TTY,
# when NO_COLOR is set, or with --color never. FORCE_COLOR / --color always force them on.
# ----------------------------------------------------------------------------
COLOR_ENABLED = False
_ANSI = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34",
         "magenta": "35", "cyan": "36", "gray": "90"}


def paint(text: Any, *styles: str) -> str:
    """Wrap text in ANSI styles when colors are enabled (otherwise return it unchanged)."""
    text = str(text)
    if not COLOR_ENABLED or not styles:
        return text
    return f"\033[{';'.join(_ANSI[s] for s in styles)}m{text}\033[0m"


def resolve_color(mode: str, stream: Any = None) -> bool:
    if mode == "always":
        return True
    if mode == "never" or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stderr
    return bool(getattr(stream, "isatty", lambda: False)()) and os.environ.get("TERM") != "dumb"


class ColorFormatter(logging.Formatter):
    LEVEL_STYLES = {"DEBUG": ("gray",), "INFO": ("green",), "WARNING": ("yellow", "bold"),
                    "ERROR": ("red", "bold"), "CRITICAL": ("red", "bold")}

    def format(self, record: logging.LogRecord) -> str:
        level = paint(f"{record.levelname:<7s}", *self.LEVEL_STYLES.get(record.levelname, ()))
        stamp = paint(self.formatTime(record, "%Y-%m-%d %H:%M:%S"), "gray")
        msg = record.getMessage()
        if record.levelno >= logging.WARNING:
            msg = paint(msg, *self.LEVEL_STYLES[record.levelname][:1])
        out = f"{stamp} {level} {msg}"
        if record.exc_info:
            out += "\n" + paint(self.formatException(record.exc_info), "red")
        return out


def setup_logging(color_mode: str = "auto") -> None:
    global COLOR_ENABLED
    COLOR_ENABLED = resolve_color(color_mode)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColorFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    # streaming a dataset logs one INFO line per HTTP request, burying the build progress
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "fsspec", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def fmt_int(n: Any) -> str:
    return f"{int(n):,}"


def fmt_tokens(n: Any) -> str:
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def fmt_duration(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------
ROLE_MAP = {
    "system": "system",
    "user": "user", "human": "user", "prompter": "user",
    "assistant": "assistant", "gpt": "assistant", "ai": "assistant",
    "bot": "assistant", "model": "assistant", "chatgpt": "assistant",
    "tool": "tool", "function": "tool", "observation": "tool",
    "function_response": "tool", "tool_response": "tool",
}

ADAPTERS: dict[str, Callable] = {}


def adapter(name: str):
    def deco(fn):
        ADAPTERS[name] = fn
        return fn
    return deco


def _jsonable(o: Any) -> Any:
    """Validate/convert a python literal so json.dumps cannot fail on it.

    ast.literal_eval also yields sets, bytes, complex numbers and non-string dict keys
    (e.g. an arguments string like "{city}" or "{'a', 'b'}"); those are not JSON data.
    """
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, dict):
        if not all(isinstance(k, str) for k in o):
            raise ValueError("non-string dict key")
        return {k: _jsonable(v) for k, v in o.items()}
    raise ValueError(f"not JSON-compatible: {type(o).__name__}")


def loads_lenient(s: Any) -> Any:
    """json.loads, falling back to python-literal (hermes uses single-quoted dicts).

    Always returns JSON-compatible data (or None when the text is not a JSON-compatible literal).
    """
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return _jsonable(ast.literal_eval(s))
    except Exception:
        return None


def maybe_json(v: Any) -> Any:
    if isinstance(v, str):
        t = v.strip()
        if t[:1] in "[{":
            try:
                return json.loads(t)
            except Exception:
                return v
    return v


def flatten_content(c: Any) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # multimodal-style parts
        out = []
        for p in c:
            if isinstance(p, dict):
                out.append(str(p.get("text", "")))
            else:
                out.append(str(p))
        return "".join(out)
    return json.dumps(c, ensure_ascii=False)


def make_call(name: Any, arguments: Any, call_id: Optional[str] = None) -> dict:
    if isinstance(arguments, str):
        s = arguments.strip()
        if not s:
            args = "{}"
        else:
            try:
                json.loads(s)
                args = s
            except Exception:
                parsed = loads_lenient(s)
                args = json.dumps(parsed, ensure_ascii=False) if parsed is not None else s
    elif arguments is None:
        args = "{}"
    else:
        args = json.dumps(arguments, ensure_ascii=False)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def parse_call_payload(content: Any) -> list[dict]:
    """sharegpt 'function_call' turns: JSON {"name":..,"arguments":..} (or a list of them)."""
    obj = loads_lenient(content) if isinstance(content, str) else content
    if isinstance(obj, dict):
        obj = [obj]
    calls = []
    if isinstance(obj, list):
        for o in obj:
            if isinstance(o, dict) and o.get("name"):
                calls.append(make_call(o["name"], o.get("arguments", o.get("parameters")), o.get("id")))
    return calls


# Python-style type names (xlam / When2Call / ToolACE) -> JSON-Schema types, so every source teaches one dialect
PY_TYPES = {"dict": "object", "object": "object", "map": "object", "mapping": "object",
            "str": "string", "string": "string", "text": "string",
            "int": "integer", "integer": "integer", "long": "integer",
            "float": "number", "number": "number", "double": "number", "decimal": "number",
            "bool": "boolean", "boolean": "boolean",
            "list": "array", "array": "array", "tuple": "array", "set": "array", "sequence": "array",
            "null": "null", "none": "null"}
GENERIC_TYPE_RE = re.compile(r"^(?P<outer>[A-Za-z_.]+)\s*\[(?P<inner>.*)\]$")


def _split_top(s: str) -> list[str]:
    """Split 'A, B[C, D]' on top-level commas."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
            continue
        depth += ch == "["
        depth -= ch == "]"
        cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse_py_type(s: str) -> tuple[Optional[dict], bool]:
    """'List[int], optional' -> ({"type": "array", "items": {"type": "integer"}}, optional=True).

    Returns (None, optional) when the type has no JSON-Schema equivalent (Any, Callable, ...): the
    property then keeps its description but no `type`.
    """
    s = s.strip()
    optional = False
    low = s.lower()
    for suffix in (", optional", ",optional", " optional", ", default none", "?"):
        if low.endswith(suffix):
            s, optional = s[: -len(suffix)].strip(), True
            low = s.lower()
    m = GENERIC_TYPE_RE.match(s)
    if m:
        outer, inner = m.group("outer").split(".")[-1].lower(), _split_top(m.group("inner"))
        if outer == "optional" and len(inner) == 1:
            got, _ = parse_py_type(inner[0])
            return got, True
        if outer == "union":
            subs = [parse_py_type(x)[0] for x in inner if x.strip().lower() not in ("none", "nonetype")]
            optional = optional or len(subs) < len(inner)
            types = [x["type"] for x in subs if x and isinstance(x.get("type"), str)]
            if len(types) == len(subs) and types:
                return ({"type": types[0]} if len(set(types)) == 1 else {"type": sorted(set(types))}), optional
            return None, optional
        t = PY_TYPES.get(outer)
        if t == "array":
            item = parse_py_type(inner[0])[0] if len(inner) == 1 else None
            return ({"type": "array", "items": item} if item else {"type": "array"}), optional
        return ({"type": t} if t else None), optional
    if "|" in s:
        return parse_py_type(f"Union[{', '.join(p.strip() for p in s.split('|'))}]")[0], optional
    t = PY_TYPES.get(low)
    return ({"type": t} if t else None), optional


def norm_schema(node: Any, required_hint: Any = None) -> Any:
    """Normalize a (possibly python-typed) parameter schema to JSON Schema, recursively.

    * "dict"/"str"/"int"/"List[str]"/"str, optional" -> object/string/integer/array+items, optional
      properties are left out of `required`
    * `required` given next to `parameters` (When2Call) is moved inside it; names that are not
      properties are dropped
    * empty-string defaults (xlam's "no default") are removed
    Already-valid JSON Schema passes through unchanged (keys are only rewritten when they need it).
    """
    if not isinstance(node, dict):
        return node
    out = dict(node)
    optional = False
    t = out.get("type")
    if isinstance(t, str) and t.strip() and t.strip().lower() not in ("object", "string", "integer", "number",
                                                                      "boolean", "array", "null"):
        got, optional = parse_py_type(t)
        out.pop("type")
        if got:
            items = got.pop("items", None)
            out.update(got)
            if items and "items" not in out:
                out["items"] = items
    elif isinstance(t, list):
        mapped = []
        for x in t:
            got = parse_py_type(x)[0] if isinstance(x, str) else None
            mapped.append(got["type"] if got and isinstance(got.get("type"), str) else x)
        out["type"] = list(dict.fromkeys(mapped))
    if isinstance(out.get("default"), str) and out.get("type") in ("integer", "number", "boolean"):
        # xlam stores every default as a string ("40.733", "" for none, sometimes junk like "fasta")
        d, kind = out["default"].strip(), out["type"]
        try:
            if kind == "boolean":
                out["default"] = {"true": True, "false": False}[d.lower()]
            else:
                f = float(d)
                out["default"] = int(f) if kind == "integer" and f.is_integer() else f
                if kind == "integer" and not f.is_integer():
                    out.pop("default")
        except (KeyError, ValueError):
            out.pop("default")
    if isinstance(out.get("properties"), dict):
        props, opt_names = {}, set()
        for k, v in out["properties"].items():
            nv = norm_schema(v)
            if isinstance(nv, dict) and nv.pop("x-optional", False):
                opt_names.add(k)
            props[k] = nv
        out["properties"] = props
        out.setdefault("type", "object")
        req = out.get("required")
        if not isinstance(req, list) and isinstance(required_hint, list):
            req = required_hint
        if isinstance(req, list):
            req = [r for r in req if isinstance(r, str) and r in props and r not in opt_names]
            out["required"] = req
        elif "required" in out:
            out.pop("required")
    for key in ("items", "additionalProperties"):
        if isinstance(out.get(key), dict):
            out[key] = norm_schema(out[key])
            out[key].pop("x-optional", None)
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(out.get(key), list):
            out[key] = [norm_schema(x) for x in out[key]]
            for x in out[key]:
                if isinstance(x, dict):
                    x.pop("x-optional", None)
    if optional:
        out["x-optional"] = True  # consumed by the parent's `required` handling above
    return out


def norm_tool(t: Any) -> Optional[dict]:
    t = maybe_json(t)
    if not isinstance(t, dict):
        return None
    if t.get("type") == "function" and isinstance(t.get("function"), dict):
        fn = dict(t["function"])
        if isinstance(fn.get("parameters"), dict):
            fn["parameters"] = norm_schema(fn["parameters"], fn.pop("required", None))
            fn["parameters"].pop("x-optional", None)
        return {**t, "function": fn}
    if "name" not in t:
        return None
    params = t.get("parameters", t.get("parameter"))
    if not isinstance(params, dict):
        params = {}
    # xlam-style flat params {"city": {"type": "str", ...}} -> JSON schema
    if "properties" not in params and params and params.get("type") not in ("object", "dict"):
        params = {"type": "object", "properties": params}
    elif not params:
        params = {"type": "object", "properties": {}}
    params = norm_schema(params, t.get("required"))
    params.pop("x-optional", None)
    return {"type": "function",
            "function": {"name": t["name"], "description": t.get("description", ""), "parameters": params}}


def parse_tools(v: Any) -> list[dict]:
    v = maybe_json(v)
    if isinstance(v, dict):
        v = [v]
    if not isinstance(v, list):
        return []
    out = []
    for t in v:
        nt = norm_tool(t)
        if nt:
            out.append(nt)
    return out


def extract_tools(text: str) -> list[dict]:
    """Pull tool JSON blobs out of a system prompt (glaive / ToolACE style)."""
    dec = json.JSONDecoder()
    tools, i = [], 0
    while i < len(text):
        if text[i] in "[{":
            try:
                obj, end = dec.raw_decode(text, i)
                items = obj if isinstance(obj, list) else [obj]
                got = [norm_tool(o) for o in items if isinstance(o, dict) and "name" in o]
                if got:
                    tools.extend(g for g in got if g)
                    i = end
                    continue
            except Exception:
                pass
        i += 1
    return tools


def norm_message(m: Any, content_key: Optional[str] = None) -> Optional[dict]:
    if not isinstance(m, dict):
        return None
    raw_role = str(m.get("role", m.get("from", ""))).lower()
    content = None
    for k in ([content_key] if content_key else []) + ["content", "value", "text"]:
        if k in m and m[k] is not None:
            content = m[k]
            break
    content = flatten_content(content)

    if raw_role in ("function_call", "tool_call"):
        calls = parse_call_payload(content)
        return {"role": "assistant", "content": "", "tool_calls": calls} if calls else None

    role = ROLE_MAP.get(raw_role)
    if role is None:
        return None
    out: dict[str, Any] = {"role": role, "content": content}
    calls = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", tc) if isinstance(tc, dict) else {}
        if fn.get("name"):
            calls.append(make_call(fn["name"], fn.get("arguments"), tc.get("id")))
    if isinstance(m.get("function_call"), dict) and m["function_call"].get("name"):
        fc = m["function_call"]
        calls.append(make_call(fc["name"], fc.get("arguments")))
    if calls and role == "assistant":
        out["tool_calls"] = calls
    if role == "tool":
        ids = m.get("tool_call_ids")
        single = ids[0] if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], str) else None
        out["tool_call_id"] = m.get("tool_call_id") or single
        out["name"] = m.get("name")
    return out


# ----------------------------------------------------------------------------
# Adapters  (row -> {"messages", "tools", "extra"} | None)
# ----------------------------------------------------------------------------
MSG_KEYS = ["messages", "conversations", "conversation", "chat", "trajectory", "dialogue"]
PAIR_KEYS = [("instruction", "response"), ("instruction", "output"), ("instruction", "answer"),
             ("problem", "solution"), ("input", "output"), ("prompt", "response"),
             ("prompt", "completion"), ("question", "answer"), ("query", "response")]


class _Safe(dict):
    def __missing__(self, k):
        return ""


@adapter("generic")
def adapt_generic(row: dict, cfg: dict) -> Optional[dict]:
    ck = cfg.get("content_field")
    keys = [cfg["messages_field"]] if cfg.get("messages_field") else MSG_KEYS
    raw = None
    for k in keys:
        if row.get(k):
            raw = maybe_json(row[k])
            break
    msgs = None
    if isinstance(raw, list):
        msgs = [norm_message(m, ck) for m in raw]
        if any(m is None for m in msgs):
            return None
    elif raw is None:
        pairs = PAIR_KEYS
        if cfg.get("user_field") or cfg.get("assistant_field"):
            if not cfg.get("user_field") or not cfg.get("assistant_field"):
                raise ValueError("user_field and assistant_field must both be configured")
            pairs = [(cfg["user_field"], cfg["assistant_field"])]
        for a, b in pairs:
            if row.get(a) and row.get(b) and isinstance(row[a], str) and isinstance(row[b], str):
                msgs = [{"role": "user", "content": row[a]}, {"role": "assistant", "content": row[b]}]
                break
    if not msgs:
        return None

    sys_txt = None
    if cfg.get("system_from"):
        sys_txt = str(cfg["system_from"]).format_map(_Safe({k: v for k, v in row.items() if isinstance(v, (str, int, float))}))
    elif isinstance(row.get("system"), str) and row["system"].strip():
        sys_txt = row["system"]
    if sys_txt and msgs[0]["role"] != "system":
        msgs.insert(0, {"role": "system", "content": sys_txt})

    tools = parse_tools(row.get(cfg.get("tools_field", "tools")))
    return {"messages": msgs, "tools": tools, "extra": {}}


# ----------------------------------------------------------------------------
# "Function tag" agent trajectories (SWE-smith xml format; OpenHands / Qwen3-coder style)
#   * tools are described as plain text in the system prompt   (---- BEGIN FUNCTION #1: bash ----)
#   * calls are written inside the assistant text              (<function=bash><parameter=command>..)
#   * results come back as ordinary `user` messages
# The adapter below converts all three into the common schema (tools / tool_calls / tool role).
# ----------------------------------------------------------------------------
FUNCTION_SECTION_MARK = "You have access to the following functions:"
FUNCTION_BLOCK_RE = re.compile(
    r"---- BEGIN FUNCTION #\d+: (?P<name>\S+) ----\n(?P<body>.*?)\n---- END FUNCTION #\d+ ----", re.S)
FUNCTION_PARAM_RE = re.compile(r"^\s*\(\d+\) (?P<name>\S+) \((?P<type>[^,)]+), (?P<req>required|optional)\): ?(?P<desc>.*)$")
ALLOWED_VALUES_RE = re.compile(r"^Allowed values: \[(?P<vals>.*)\]$")
FUNCTION_CALL_RE = re.compile(r"<function=(?P<name>[^>\s]+)>(?P<body>.*?)</function>", re.S)
PARAM_TAG_RE = re.compile(r"<parameter=(?P<name>[^>\s]+)>(?P<value>.*?)</parameter>", re.S)
NO_PARAMS_RE = re.compile(r"\n?No parameters are required for this function\.\s*$")


def parse_function_defs(text: str) -> list[dict]:
    """Text function definitions -> OpenAI-style tool schemas (types/enums/required exactly as declared)."""
    tools = []
    for blk in FUNCTION_BLOCK_RE.finditer(text):
        head, sep, params_txt = blk["body"].partition("\nParameters:\n")
        if not sep:
            head = NO_PARAMS_RE.sub("", blk["body"])
        desc = head.strip()
        if desc.startswith("Description:"):
            desc = desc[len("Description:"):].strip()
        props: dict[str, dict] = {}
        required: list[str] = []
        cur = None
        for line in params_txt.splitlines():
            m = FUNCTION_PARAM_RE.match(line)
            if m:
                cur = m["name"]
                props[cur] = {"type": m["type"].strip(), "description": m["desc"].strip()}
                if m["req"] == "required":
                    required.append(cur)
            elif cur is not None and line.strip():
                allowed = ALLOWED_VALUES_RE.match(line.strip())
                if allowed:
                    props[cur]["enum"] = re.findall(r"`([^`]*)`", allowed["vals"])
                else:
                    props[cur]["description"] += "\n" + line.strip()
        parameters: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            parameters["required"] = required
        tools.append({"type": "function",
                      "function": {"name": blk["name"], "description": desc, "parameters": parameters}})
    return tools


def strip_function_section(text: str) -> str:
    """Drop the textual tool list and the call-format instructions that follow it."""
    i = text.find(FUNCTION_SECTION_MARK)
    return text[:i].rstrip() if i >= 0 else text


def coerce_param(value: str, spec: dict) -> tuple[Any, bool]:
    """Convert a <parameter> text value to the declared JSON type; (value, ok)."""
    kind = spec.get("type", "string")
    if kind == "string":
        # multi-line values are written as "<parameter=x>\nTEXT\n</parameter>": one wrapper newline per side
        if value.startswith("\n"):
            value = value[1:]
            if value.endswith("\n"):
                value = value[:-1]
        return value, True
    v = value.strip()
    try:
        if kind == "integer":
            return int(v), True
        if kind == "number":
            return (int(v) if re.fullmatch(r"-?\d+", v) else float(v)), True
        if kind == "boolean" and v.lower() in ("true", "false"):
            return v.lower() == "true", True
        if kind in ("array", "object"):
            got = loads_lenient(v)
            if isinstance(got, list if kind == "array" else dict):
                return got, True
    except ValueError:
        pass
    return value, False


def parse_function_calls(content: str, tools: dict[str, dict]) -> tuple[str, list[dict], bool]:
    """Extract <function=..> blocks -> (remaining text, tool_calls, all_calls_valid).

    A call is invalid when its body has stray text, a parameter is unknown/duplicated/mistyped,
    or a required parameter is missing (checked only for functions with a known schema).
    """
    calls, ok = [], True
    for m in FUNCTION_CALL_RE.finditer(content):
        fn = tools.get(m["name"])
        props = fn["parameters"].get("properties", {}) if fn else None
        args: dict[str, Any] = {}
        if PARAM_TAG_RE.sub("", m["body"]).strip():
            ok = False
        for p in PARAM_TAG_RE.finditer(m["body"]):
            name = p["name"]
            if props is not None and name not in props:
                ok = False
            if name in args:
                ok = False
            args[name], good = coerce_param(p["value"], props.get(name, {}) if props else {})
            ok = ok and good
        if fn and set(fn["parameters"].get("required", [])) - set(args):
            ok = False
        calls.append(make_call(m["name"], args))
    return FUNCTION_CALL_RE.sub("", content).strip(), calls, ok


# Tool definitions shared by the SWE-smith trajectories, copied verbatim from the dataset's own XML-format
# system prompts. The native (`tool_calls`) rows use the same three tools but ship no definitions.
SWE_AGENT_FUNCTION_DEFS = r"""---- BEGIN FUNCTION #1: bash ----
Description: Execute a bash command in the terminal.

Parameters:
  (1) command (string, required): The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: submit ----
Description: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
No parameters are required for this function.
---- END FUNCTION #2 ----

---- BEGIN FUNCTION #3: str_replace_editor ----
Description: Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`

Parameters:
  (1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
Allowed values: [`view`, `create`, `str_replace`, `insert`, `undo_edit`]
  (2) path (string, required): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
  (3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.
  (4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
  (5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
  (6) insert_line (integer, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
  (7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
---- END FUNCTION #3 ----"""
DEFAULT_TOOLSETS: dict[str, list[dict]] = {"swe_agent": parse_function_defs(SWE_AGENT_FUNCTION_DEFS)}


@adapter("function_tags")
def adapt_function_tags(row: dict, cfg: dict) -> Optional[dict]:
    """generic adapter + textual function definitions/calls/observations -> structured tools/tool_calls/tool role.

    YAML: `default_tools: swe_agent` supplies tool schemas for rows that ship none.
    """
    s = adapt_generic(row, cfg)
    if s is None:
        return None
    msgs, tools, extra = s["messages"], s["tools"], s["extra"]
    if msgs[0]["role"] == "system" and FUNCTION_SECTION_MARK in msgs[0]["content"]:
        parsed = parse_function_defs(msgs[0]["content"])
        if parsed:
            tools = tools or parsed
            msgs[0]["content"] = strip_function_section(msgs[0]["content"])
    if not tools and cfg.get("default_tools"):
        if cfg["default_tools"] not in DEFAULT_TOOLSETS:
            raise ValueError(f"unknown default_tools {cfg['default_tools']!r}; known: {sorted(DEFAULT_TOOLSETS)}")
        tools = copy.deepcopy(DEFAULT_TOOLSETS[cfg["default_tools"]])
    by_name = {t["function"]["name"]: t["function"] for t in tools}

    out, called = [], False
    for m in msgs:
        if m["role"] == "assistant" and not m.get("tool_calls") and "<function=" in m["content"]:
            text, calls, ok = parse_function_calls(m["content"], by_name)
            if calls:
                m = {**m, "content": text, "tool_calls": calls}
            if not ok:
                extra["tool_parse_failed"] = True
        elif m["role"] == "user" and called:
            m = {**m, "role": "tool"}  # the reply right after a call is that call's result
        called = m["role"] == "assistant" and bool(m.get("tool_calls"))
        out.append(m)
    return {"messages": out, "tools": tools, "extra": extra}


@adapter("xlam")
def adapt_xlam(row: dict, cfg: dict) -> Optional[dict]:
    q = row.get("query")
    if not q:
        return None
    answers = maybe_json(row.get("answers")) or []
    tools = parse_tools(row.get("tools"))
    msgs = [{"role": "user", "content": q}]
    extra = {}
    if answers:
        calls = [make_call(a.get("name"), a.get("arguments")) for a in answers if isinstance(a, dict)]
        msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
    else:  # irrelevance set: target is "no suitable tool"
        msgs.append({"role": "assistant", "content": cfg.get(
            "no_call_reply", "None of the available tools can help with this request.")})
        extra["synthetic_reply"] = True
    return {"messages": msgs, "tools": tools, "extra": extra}


def parse_bracket_calls(v: str, tool_names: Optional[list[str]] = None) -> Optional[list[dict]]:
    """ToolACE: '[Func(a=1, b="x"), Other()]'"""
    aliases = {}
    name_pattern = ""
    reverse = {}
    if tool_names:
        aliases = {f"__sft_tool_{i}": name for i, name in enumerate(tool_names)}
        reverse = {name: alias for alias, name in aliases.items()}
        names = "|".join(re.escape(name) for name in sorted(reverse, key=len, reverse=True))
        name_pattern = r"|(?<![\w.])(?P<name>" + names + r")(?=\s*\()"
    prefix = "__sft_param_"
    while prefix in v:
        prefix = "_" + prefix
    parameters = {f"{prefix}{i}": name for i, name in enumerate(keyword.kwlist)}
    parameter_reverse = {name: alias for alias, name in parameters.items()}
    # Match quoted arguments first so tool-like text inside them is untouched.
    pattern = re.compile(
        r"""(?P<quoted>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')"""
        + name_pattern + r"|(?<![\w.])(?P<parameter>"
        + "|".join(keyword.kwlist) + r")(?=\s*=(?!=))", re.S
    )

    def replace(match):
        if tool_names and match["name"]:
            return reverse[match["name"]]
        if match["parameter"]:
            return parameter_reverse[match["parameter"]]
        return match[0]

    v = pattern.sub(replace, v)
    try:
        node = ast.parse(v, mode="eval").body
        if not isinstance(node, ast.List):
            return None
        calls = []
        for elt in node.elts:
            if not isinstance(elt, ast.Call):
                return None
            if elt.args or any(kw.arg is None for kw in elt.keywords):
                return None
            parsed_name = ast.unparse(elt.func)
            name = aliases.get(parsed_name, parsed_name)
            args = {parameters.get(kw.arg, kw.arg): ast.literal_eval(kw.value) for kw in elt.keywords}
            if len(args) != len(elt.keywords):
                return None
            calls.append(make_call(name, args))
        return calls
    except Exception:
        return None


@adapter("toolace")
def adapt_toolace(row: dict, cfg: dict) -> Optional[dict]:
    system = row.get("system") or ""
    tools = extract_tools(system)
    tool_names = [t["function"]["name"] for t in tools if isinstance(t["function"]["name"], str)]
    msgs = [{"role": "system", "content": system}] if system.strip() else []
    for t in row.get("conversations") or []:
        r, v = str(t.get("from", "")).lower(), flatten_content(t.get("value")).strip()
        role = ROLE_MAP.get(r)
        if role is None:
            return None
        if role == "assistant" and v.startswith("[") and v.endswith("]") and "(" in v:
            calls = parse_bracket_calls(v, tool_names)
            if calls is None:
                return None  # looks like a call but unparsable -> drop
            msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
        elif role == "tool":
            results = loads_lenient(v)
            if isinstance(results, list) and results and all(
                    isinstance(result, dict) and isinstance(result.get("name"), str) for result in results
            ):
                msgs.extend({"role": "tool", "name": result["name"],
                             "content": json.dumps(result, ensure_ascii=False)} for result in results)
            else:
                msgs.append({"role": role, "content": v})
        else:
            msgs.append({"role": role, "content": v})
    return {"messages": msgs, "tools": tools, "extra": {}}


GLAIVE_SPLIT = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")


def parse_glaive_call(body: str) -> Optional[dict]:
    body = body.replace("<functioncall>", "").strip()
    nm = re.search(r'"name":\s*"([^"]+)"', body)
    if not nm:
        return None
    am = re.search(r'"arguments":\s*\'(.*)\'\s*}\s*$', body, re.S) or \
         re.search(r'"arguments":\s*(\{.*\})\s*}\s*$', body, re.S)
    return make_call(nm.group(1), am.group(1) if am else "{}")


@adapter("glaive")
def adapt_glaive(row: dict, cfg: dict) -> Optional[dict]:
    system = re.sub(r"^\s*SYSTEM:\s*", "", row.get("system") or "")
    chat = row.get("chat") or ""
    tools = extract_tools(system)
    msgs = [{"role": "system", "content": system.strip()}] if system.strip() else []
    parts = GLAIVE_SPLIT.split(chat)
    i = 1
    while i < len(parts) - 1:
        tag, body = parts[i], parts[i + 1].replace("<|endoftext|>", "").strip()
        i += 2
        if tag == "USER:":
            msgs.append({"role": "user", "content": body})
        elif tag == "ASSISTANT:":
            if body.startswith("<functioncall>"):
                call = parse_glaive_call(body)
                if not call:
                    return None
                msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
            else:
                msgs.append({"role": "assistant", "content": body})
        else:
            msgs.append({"role": "tool", "content": body})
    return {"messages": msgs, "tools": tools, "extra": {}}


# ----------------------------------------------------------------------------
# Finalize / validate a normalized sample
# ----------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
TOOL_RESP_RE = re.compile(r"^\s*<tool_response>\s*(.*?)\s*</tool_response>\s*$", re.S)
BOILERPLATE_RE = re.compile(r"as an ai language model|as a large language model|i am an ai language model", re.I)


def finalize(sample: dict, cfg: dict, args) -> tuple[Optional[dict], Optional[str]]:
    msgs = sample["messages"]
    tools = sample.get("tools") or []
    extra = sample.get("extra", {})
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"].strip():
            return None, "invalid_tools"

    # inline <tool_call> blocks (hermes / qwen style) -> structured tool_calls
    for m in msgs:
        if m["role"] == "assistant" and "<tool_call>" in m["content"]:
            found = TOOL_CALL_RE.findall(m["content"])
            calls = []
            for payload in found:
                got = parse_call_payload(payload)
                if not got:
                    extra["tool_parse_failed"] = True
                calls.extend(got)
            if calls:
                m["tool_calls"] = (m.get("tool_calls") or []) + calls
                m["content"] = TOOL_CALL_RE.sub("", m["content"]).strip()
        elif m["role"] == "tool":
            mm = TOOL_RESP_RE.match(m["content"])
            if mm:
                m["content"] = mm.group(1)

    # drop empty messages (except assistant tool-call turns)
    msgs = [m for m in msgs if m["content"].strip() or m.get("tool_calls")]
    if not msgs:
        return None, "empty"
    artifact = cfg.get("drop_if_last_assistant_matches")
    if artifact and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls") \
            and re.search(artifact, msgs[-1]["content"]):
        return None, "exit_artifact"  # e.g. SWE-agent "Exit due to ..." status, not model output
    for m in msgs:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") if isinstance(c, dict) else None
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"].strip():
                return None, "invalid_tools"
            if c.get("id") and not isinstance(c["id"], str):
                return None, "invalid_tools"

    # link tool_call ids <-> tool results
    pending, n = [], 0
    links_valid = True
    used_ids = {c["id"] for m in msgs for c in m.get("tool_calls") or [] if c.get("id")}
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                if not c.get("id"):
                    n += 1
                    while f"call_{n}" in used_ids:
                        n += 1
                    c["id"] = f"call_{n}"
                    used_ids.add(c["id"])
                if any(p["id"] == c["id"] for p in pending):
                    links_valid = False
                pending.append(c)
        elif m["role"] == "tool":
            if m.get("tool_call_id"):
                if not any(c["id"] == m["tool_call_id"] for c in pending):
                    links_valid = False
                pending = [c for c in pending if c["id"] != m["tool_call_id"]]
            elif pending:
                pos = next((i for i, c in enumerate(pending)
                            if c["function"]["name"] == m.get("name")), None) if m.get("name") else 0
                if pos is None:
                    links_valid = False
                else:
                    c = pending.pop(pos)
                    m["tool_call_id"] = c["id"]
                    m["name"] = m.get("name") or c["function"]["name"]

    roles = [m["role"] for m in msgs]
    body = [r for r in roles if r != "system"]
    if "user" not in body or not any(m["role"] == "assistant" for m in msgs):
        return None, "invalid_structure"
    if body[0] != "user" and not cfg.get("allow_non_user_first", False):
        return None, "invalid_structure"
    if msgs[-1]["role"] != "assistant" and not cfg.get("allow_non_assistant_last", False):
        return None, "invalid_structure"

    if BOILERPLATE_RE.search(" ".join(m["content"] for m in msgs if m["role"] == "assistant")):
        return None, "boilerplate"

    # tool-call validity
    names = {t["function"]["name"] for t in tools}
    tool_valid, names_known = links_valid, True
    for m in msgs:
        for c in m.get("tool_calls") or []:
            try:
                a = json.loads(c["function"]["arguments"])
                if not isinstance(a, dict):
                    tool_valid = False
            except Exception:
                tool_valid = False
            if not c["function"]["name"] or (names and c["function"]["name"] not in names):
                names_known = False
    if extra.get("tool_parse_failed"):
        tool_valid = False
    extra["tool_valid"], extra["tool_names_known"] = tool_valid, names_known
    if args.drop_invalid_tools and not (tool_valid and names_known):
        return None, "invalid_tools"

    return {"messages": msgs, "tools": tools, "extra": extra}, None


# ----------------------------------------------------------------------------
# Language / code analysis  (Thanglish = lexicon-based; replace with a classifier later)
# ----------------------------------------------------------------------------
THANGLISH_LEX = set("""
enna ennanu ennada illa illai illana irukku irukkum iruku irukken irukkanga irundha irundhal
panna pannu pannunga pannitu panren panrathu pannalam seri sari romba nalla nalladhu vendam venum venam
theriyum theriyala theriyathu theriyuma puriyuthu puriyala purinjudhu sollu solla solren sollunga sollunga
paaru paaka paarunga vaanga vaa vara varum vandhu vanthu poga pogalam poi ponga anga inga enga
yaar yaaru epdi eppadi eppo eppodhu ippo ippadi appadi appo athu idhu ithu adhu naan neenga nee unga
enakku unakku ungalukku avan aval avanga konjam neraya thaan dhaan mattum aana aanaal ellam ellarum
vishayam velai vaarthai mudiyum mudiyathu mudiyala kudu kudunga podu pottu thambi machan kitta kooda pathi
paththi saapadu sapadu irukka pannanum pannum ennoda unnoda avanoda namma nammaku mela keela ulla veliya
""".split())
EN_STOP = set("""the a an and or of to in is are was were be been it that this for on with as at by from
have has had not but you your i we they he she will would can could should do does did if then so what which
who how when where why my our their there here""".split())
CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
URL_RE = re.compile(r"https?://\S+")
TAMIL_RE = re.compile(r"[\u0b80-\u0bff]")
LETTER_RE = re.compile(r"[^\W\d_]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[a-z']+")


def lang_stats(text: str) -> dict:
    text = text[:20000]
    code_chars = sum(len(b) for b in CODE_BLOCK_RE.findall(text))
    prose = URL_RE.sub(" ", INLINE_CODE_RE.sub(" ", CODE_BLOCK_RE.sub(" ", text)))
    n_letters = len(LETTER_RE.findall(prose))
    letters = n_letters or 1
    is_ascii = prose.isascii()
    tamil_ratio = 0.0 if is_ascii else len(TAMIL_RE.findall(prose)) / letters
    words = WORD_RE.findall(prose.lower())
    nw = len(words) or 1
    hits = sum(w in THANGLISH_LEX for w in words)
    score = hits / nw
    en = sum(w in EN_STOP for w in words) / nw
    ascii_letters = n_letters if is_ascii else len(ASCII_LETTER_RE.findall(prose))
    if tamil_ratio >= 0.3:
        lang = "ta"
    elif tamil_ratio >= 0.03:
        lang = "ta_mixed"
    elif hits >= 3 and score >= 0.12:
        lang = "thanglish"
    elif hits >= 2 and score >= 0.03:
        lang = "en_thanglish_mixed"
    elif len(words) < 5:
        lang = "und"
    elif en >= 0.2:
        lang = "en"
    elif ascii_letters / letters > 0.9:
        lang = "latin_other"
    else:
        lang = "other"
    return {"lang": lang, "thanglish_score": round(score, 4), "tamil_ratio": round(tamil_ratio, 4),
            "code_chars": code_chars}


# ----------------------------------------------------------------------------
# Tokenizer wrapper
# ----------------------------------------------------------------------------
class TokenCounter:
    def __init__(self, name: str):
        self.name = name
        self.tok = None
        self._fast_tok = None
        if name != "approx":
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            self.tok.model_max_length = 10 ** 9
            if self.tok is not None and hasattr(self.tok, "_tokenizer"):
                self._fast_tok = getattr(self.tok, "_tokenizer", None)
                if self._fast_tok is not None:
                    self._fast_tok.no_padding()
                    self._fast_tok.no_truncation()


    def count(self, texts: list[str]) -> list[int]:
        if self.tok is None:
            return [(len(t) + 3) // 4 for t in texts]  # rough: dry runs only
        nonempty = [t for t in texts if t]
        if not nonempty:
            return [0] * len(texts)
        if self._fast_tok is None:
            enc = self.tok(nonempty, add_special_tokens=False,
                           return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        else:
            enc = self._fast_tok.encode_batch(nonempty, add_special_tokens=False)
            enc = [e.ids for e in enc]
        lengths = iter(len(x) for x in enc)
        return [next(lengths) if t else 0 for t in texts]


PIECE_KEYS = ["system", "user", "assistant", "tool_result", "tool_calls", "tools_schema"]


def pieces(s: dict) -> dict:
    p = {k: [] for k in PIECE_KEYS}
    for m in s["messages"]:
        if m["role"] == "system":
            p["system"].append(m["content"])
        elif m["role"] == "user":
            p["user"].append(m["content"])
        elif m["role"] == "assistant":
            if m["content"]:
                p["assistant"].append(m["content"])
            for c in m.get("tool_calls") or []:
                p["tool_calls"].append(json.dumps(c["function"], ensure_ascii=False))
        else:
            p["tool_result"].append(m["content"])
    if s["tools"]:
        p["tools_schema"].append(json.dumps(s["tools"], ensure_ascii=False))
    return {k: "\n".join(v) for k, v in p.items()}


# ----------------------------------------------------------------------------
# Dedup
# ----------------------------------------------------------------------------
def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().strip())


def h128(t: str) -> str:
    return hashlib.blake2b(t.encode("utf-8", "ignore"), digest_size=16).hexdigest()


def content_hash(s: dict) -> str:
    parts = []
    for m in s["messages"]:
        parts.append(m["role"] + ":" + norm_text(m["content"]))
        for c in m.get("tool_calls") or []:
            parts.append("call:" + c["function"]["name"] + norm_text(c["function"]["arguments"]))
    if s.get("tools"):
        parts.append("tools:" + json.dumps(s["tools"], ensure_ascii=False, sort_keys=True))
    return h128("\n".join(parts))


def prompt_hash(s: dict) -> str:
    for m in s["messages"]:
        if m["role"] == "user":
            return h128(norm_text(m["content"]))
    return ""


def make_minhash_template(num_perm: int):
    """A MinHash whose permutation tables are built once and reused (building them per sample is slow)."""
    try:
        from datasketch import MinHash
    except ImportError:
        return None
    return MinHash(num_perm=num_perm)


def minhash_signature(text: str, template) -> Optional[Any]:
    """MinHash hash values of the first 300 words' 5-shingles; None for very short texts (never deduped)."""
    words = norm_text(text).split()[:300]
    if len(words) < 8:
        return None
    shingles = {" ".join(words[i:i + 5]) for i in range(max(1, len(words) - 4))}
    m = template.copy()
    if hasattr(m, "update_batch"):
        m.update_batch([s.encode("utf-8") for s in shingles])
    else:
        for s in shingles:
            m.update(s.encode("utf-8"))
    return m.hashvalues.copy()


class Deduper:
    """Global exact + near (MinHash-LSH) dedup. Not thread/process safe: only the parent process uses it.

    begin()/commit()/rollback() make one dataset's additions atomic, so a dataset that fails midway
    cannot make later datasets lose duplicates of samples that were never written.
    """

    def __init__(self, near: bool, threshold: float, num_perm: int):
        self.seen: set[str] = set()
        self.lsh = None
        self.num_perm = num_perm
        self.minhash_template = None
        self._tx: Optional[tuple[list, list]] = None
        if near:
            try:
                from datasketch import MinHashLSH
                self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
                self.minhash_template = make_minhash_template(num_perm)
            except ImportError:
                log.warning("datasketch not installed -> near-dedup disabled (pip install datasketch)")

    def begin(self) -> None:
        self._tx = ([], [])

    def commit(self) -> None:
        self._tx = None

    def rollback(self) -> None:
        if self._tx is not None:
            hashes, keys = self._tx
            self.seen.difference_update(hashes)
            if self.lsh is not None:
                for k in keys:
                    self.lsh.remove(k)
        self._tx = None

    def exact_dup(self, h: str) -> bool:
        if h in self.seen:
            return True
        self.seen.add(h)
        if self._tx is not None:
            self._tx[0].append(h)
        return False

    def signature(self, text: str) -> Optional[Any]:
        if self.lsh is None or self.minhash_template is None:
            return None
        return minhash_signature(text, self.minhash_template)

    def near_dup_sig(self, key: str, sig: Any, scope: str = "") -> bool:
        """LSH lookup/insert for a precomputed signature (see minhash_signature)."""
        if self.lsh is None or sig is None:
            return False
        m = self.minhash_template.copy()
        m.hashvalues = sig
        if any(candidate[0] == scope for candidate in self.lsh.query(m)):
            return True
        self.lsh.insert((scope, key), m)
        if self._tx is not None:
            self._tx[1].append((scope, key))
        return False

    def near_dup(self, key: str, text: str, scope: str = "") -> bool:
        return self.near_dup_sig(key, self.signature(text), scope)


def near_text(s: dict) -> str:
    u = next((m["content"] for m in s["messages"] if m["role"] == "user"), "")
    a = next((m["content"] or json.dumps(m.get("tool_calls"), ensure_ascii=False)
              for m in s["messages"] if m["role"] == "assistant"), "")
    return u + " " + a


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def oasst_threads(cfg: dict, args) -> Iterator[dict]:
    """Rebuild OASST message trees into linear conversations (rank-0 assistant replies only)."""
    from datasets import load_dataset
    msgs, children = {}, defaultdict(list)
    for split in cfg.get("splits", ["train", "validation"]):
        for r in load_dataset(cfg["repo"], split=split, token=HF_TOKEN, revision=cfg.get("revision")):
            if r.get("deleted") or r.get("review_result") is False:
                continue
            msgs[r["message_id"]] = r
            children[r["parent_id"]].append(r)
    def walk(path):
        kids = [k for k in children.get(path[-1]["message_id"], [])
                if k["role"] == "prompter" or k.get("rank") in (None, 0)]
        if not kids:
            if path[-1]["role"] == "assistant":
                yield path
            return
        for k in kids:
            yield from walk(path + [k])

    for root in [m for m in msgs.values() if m["parent_id"] is None]:
        for path in walk([root]):
            yield {"id": path[-1]["message_id"], "lang": root.get("lang"),
                   "messages": [{"role": m["role"], "content": m["text"]} for m in path]}


def load_rows(cfg: dict, args) -> Iterator[dict]:
    loader = cfg.get("loader", "hf")
    if loader == "jsonl":
        files = sorted(glob.glob(cfg["path"]))
        if not files:
            raise FileNotFoundError(f"no files match {cfg['path']}")
        for f in files:
            op = gzip.open if f.endswith(".gz") else open
            with op(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)
        return
    if loader == "oasst":
        yield from oasst_threads(cfg, args)
        return

    from datasets import get_dataset_split_names, load_dataset
    configs = cfg.get("configs") or [cfg.get("config")]
    for conf in configs:
        splits = cfg.get("splits", ["train"])
        if splits == "all":
            splits = get_dataset_split_names(cfg["repo"], conf, token=HF_TOKEN,
                                             revision=cfg.get("revision"))
        elif isinstance(splits, str):
            splits = [splits]
        for sp in splits:
            kwargs = {"data_files": cfg["data_files"]} if cfg.get("data_files") else {}
            ds = load_dataset(cfg["repo"], conf, split=sp, streaming=cfg.get("streaming", not args.no_streaming),
                              token=HF_TOKEN, revision=cfg.get("revision"), **kwargs)
            for row in ds:
                row = dict(row)
                row["__split"], row["__config"] = sp, conf
                yield row


_MISSING = object()


def get_path(row: Any, path: str, default: Any = None) -> Any:
    """Dotted lookup into nested dicts, e.g. 'response_quality_assessment.overall_score'.
    Any level that is a JSON string (common in HF datasets) is parsed on the fly."""
    cur = row
    for key in path.split("."):
        cur = maybe_json(cur)
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def passes_where(row: dict, conds: Optional[list]) -> bool:
    for c in conds or []:
        v, t, op = get_path(row, c["field"]), c.get("value"), c.get("op", "eq")
        if op not in {"eq", "ne", "gt", "lt", "ge", "le", "in", "truthy"}:
            raise ValueError(f"Unsupported where operator: {op!r}")
        if v is None:  # missing / null field never passes
            return False
        if isinstance(v, str) and isinstance(t, bool):
            boolean = v.strip().lower()
            if boolean not in {"true", "false"}:
                return False
            v = boolean == "true"
        try:
            if op == "eq":
                ok = v == t
            elif op == "ne":
                ok = v != t
            elif op == "gt":
                ok = float(v) > float(t)
            elif op == "lt":
                ok = float(v) < float(t)
            elif op == "ge":
                ok = float(v) >= float(t)
            elif op == "le":
                ok = float(v) <= float(t)
            elif op == "in":
                ok = v in t
            else:
                ok = bool(v)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return False
    return True


# ----------------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, directory: Path, shard_size: int, gz: bool):
        self.dir, self.size, self.gz = directory, shard_size, gz
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in self.dir.glob("part-*"):
            old.unlink()
        self.i, self.n, self.fh, self.name = -1, 0, None, ""

    def _open(self):
        if self.fh:
            self.fh.close()
        self.i += 1
        self.name = f"part-{self.i:05d}.jsonl" + (".gz" if self.gz else "")
        p = self.dir / self.name
        self.fh = gzip.open(p, "wt", encoding="utf-8") if self.gz else open(p, "w", encoding="utf-8")
        self.n = 0

    def write(self, rec: dict) -> tuple[str, int]:
        return self.write_line(json.dumps(rec, ensure_ascii=False))

    def write_line(self, line: str) -> tuple[str, int]:
        """Write one already-serialized JSON record (workers serialize in parallel)."""
        if self.fh is None or self.n >= self.size:
            self._open()
        self.fh.write(line + "\n")
        self.n += 1
        return self.name, self.n - 1

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None

    def discard(self):
        """Remove every shard written so far (used when a build fails midway)."""
        self.close()
        for old in self.dir.glob("part-*"):
            old.unlink(missing_ok=True)


class IndexWriter:
    def __init__(self, directory: Path, batch_size: int):
        import pyarrow as pa
        import pyarrow.parquet as pq
        integer_fields = [
            "row", "n_messages", "n_user_turns", "n_assistant_turns", "n_tool_msgs",
            "n_tool_calls", "n_tools_defined", "tokens_total", "tokens_trainable",
            "tokens_system", "tokens_user", "tokens_assistant", "tokens_tool_result",
            "tokens_tool_calls", "tokens_tools_schema", "n_chars",
        ]
        string_fields = [
            "id", "source", "group", "shard", "license", "orig_id", "token_method",
            "lang", "lang_user", "lang_assistant", "content_hash", "prompt_hash", "extra",
        ]
        self.schema = pa.schema(
            [(k, pa.string()) for k in string_fields]
            + [(k, pa.int64()) for k in integer_fields]
            + [(k, pa.float64()) for k in ("thanglish_score", "tamil_ratio", "code_ratio")]
            + [("tool_valid", pa.bool_())]
        )
        self.path = directory / "_index.parquet"
        self.temp_path = directory / "_index.parquet.tmp"
        self.writer = pq.ParquetWriter(self.temp_path, self.schema)
        self.batch_size = batch_size
        self.buffer: list[dict] = []

    def write(self, row: dict):
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if self.buffer:
            import pyarrow as pa
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()

    def close(self, complete: bool):
        try:
            if complete:
                self.flush()
        finally:
            self.writer.close()
        if complete:
            self.temp_path.replace(self.path)
        else:
            self.temp_path.unlink(missing_ok=True)


def safe_extra(row: dict, fields: list[str]) -> dict:
    """Copy scalar fields (dotted paths allowed) into meta.extra, e.g. 'a.b' -> 'a_b'."""
    out = {}
    for f in fields + ["__split", "__config"]:
        v = get_path(row, f, _MISSING)
        if v is not _MISSING and (v is None or isinstance(v, (str, int, float, bool))):
            out[f.strip("_").replace(".", "_")] = v
    return out


# ----------------------------------------------------------------------------
# Per-chunk work: runs in worker processes (or in-process with --workers 1)
# ----------------------------------------------------------------------------
_END = object()
PROGRESS_EVERY = 20000


def process_chunk(items: list, cfg: dict, args, tok: TokenCounter, template) -> dict:
    """Stateless, CPU-heavy part of the pipeline for one chunk of (row_index, row) pairs.

    normalize -> validate -> tokenize -> length filters -> hashes -> MinHash signature -> language tags
    -> JSON serialization. Everything order-dependent (exact/near dedup, shard writing) stays in the
    parent, so parallel output is identical to a serial run.
    """
    name, group = cfg["name"], cfg.get("group", "misc")
    adapt = ADAPTERS[cfg.get("adapter", "generic")]
    max_tokens = cfg.get("max_tokens", args.max_tokens)  # 0 = disabled
    min_asst = cfg.get("min_assistant_tokens", args.min_assistant_tokens)
    near_ok = cfg.get("near_dedup", True) and not args.no_near_dedup and template is not None
    stats: Counter = Counter()
    cpu: Counter = Counter()
    errors: list[str] = []

    started = time.perf_counter()
    prepared = []
    for i, row in items:
        try:
            s = adapt(row, cfg)
            if s is None:
                stats["drop_adapter_none"] += 1
                continue
            s, reason = finalize(s, cfg, args)
        except Exception as e:  # noqa - one odd row must not abort a multi-hour build
            stats["drop_adapter_error"] += 1
            if len(errors) < 3:
                errors.append(repr(e))
            continue
        if s is None:
            stats[f"drop_{reason}"] += 1
            continue
        oid = row.get(cfg.get("id_field", "id"), row.get("uuid", i))
        prepared.append((i, oid, s, safe_extra(row, cfg.get("extra_fields", []))))
    cpu["normalize"] += time.perf_counter() - started

    started = time.perf_counter()
    texts: list[str] = []
    all_pieces = []
    for _, _, s, _ in prepared:
        p = pieces(s)
        all_pieces.append(p)
        texts.extend(p[k] for k in PIECE_KEYS)
    counts = tok.count(texts) if texts else []
    cpu["tokenize"] += time.perf_counter() - started

    kept = []
    n_keys = len(PIECE_KEYS)
    for j, (i, oid, s, row_extra) in enumerate(prepared):
        p = all_pieces[j]
        c = dict(zip(PIECE_KEYS, counts[j * n_keys:(j + 1) * n_keys]))
        n_msgs = len(s["messages"])
        total = sum(c.values()) + args.msg_overhead * n_msgs
        n_asst = sum(m["role"] == "assistant" for m in s["messages"])
        trainable = c["assistant"] + c["tool_calls"] + args.msg_overhead * n_asst
        if max_tokens and total > max_tokens:
            stats["drop_too_long"] += 1
            continue
        if c["assistant"] + c["tool_calls"] < min_asst:
            stats["drop_too_short"] += 1
            continue

        started = time.perf_counter()
        ch = content_hash(s)
        scope = h128(json.dumps(s["tools"], ensure_ascii=False, sort_keys=True)) if s["tools"] else ""
        sig = minhash_signature(near_text(s), template) if near_ok else None
        cpu["hash"] += time.perf_counter() - started

        started = time.perf_counter()
        lu, la = lang_stats(p["user"]), lang_stats(p["assistant"])
        lall = lang_stats(p["user"] + "\n" + p["assistant"])
        cpu["language"] += time.perf_counter() - started
        prose_chars = len(p["user"]) + len(p["assistant"])
        meta = {
            "license": cfg.get("license"),
            "orig_id": str(oid),
            "n_messages": n_msgs,
            "n_user_turns": sum(m["role"] == "user" for m in s["messages"]),
            "n_assistant_turns": n_asst,
            "n_tool_msgs": sum(m["role"] == "tool" for m in s["messages"]),
            "n_tool_calls": sum(len(m.get("tool_calls") or []) for m in s["messages"]),
            "n_tools_defined": len(s["tools"]),
            "tool_valid": s["extra"].get("tool_valid", True),
            "tokens_total": total,
            "tokens_trainable": trainable,
            "tokens_system": c["system"], "tokens_user": c["user"],
            "tokens_assistant": c["assistant"], "tokens_tool_result": c["tool_result"],
            "tokens_tool_calls": c["tool_calls"], "tokens_tools_schema": c["tools_schema"],
            "token_method": tok.name,
            "lang": lall["lang"], "lang_user": lu["lang"], "lang_assistant": la["lang"],
            "thanglish_score": lall["thanglish_score"], "tamil_ratio": lall["tamil_ratio"],
            "code_ratio": round(lall["code_chars"] / prose_chars, 4) if prose_chars else 0.0,
            "n_chars": prose_chars,
            "content_hash": ch, "prompt_hash": prompt_hash(s),
            "extra": json.dumps(s["extra"] | row_extra, ensure_ascii=False),
        }
        sid = f"{name}:{i}"
        started = time.perf_counter()
        line = json.dumps({"id": sid, "source": name, "group": group, "messages": s["messages"],
                           "tools": s["tools"], "meta": meta}, ensure_ascii=False)
        cpu["serialize"] += time.perf_counter() - started
        kept.append({"sid": sid, "ch": ch, "scope": scope, "sig": sig, "line": line, "meta": meta,
                     "total": total, "trainable": trainable})
    return {"stats": stats, "cpu": cpu, "errors": errors, "items": kept}


# ----------------------------------------------------------------------------
# Worker pool (spawn: fork is unsafe with the Rust tokenizers library)
# ----------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(tokenizer: str, num_perm: int, near: bool) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # parallelism comes from the process pool
    try:
        _WORKER["tok"] = TokenCounter(tokenizer)
        _WORKER["template"] = make_minhash_template(num_perm) if near else None
    except Exception as e:  # noqa - a raising initializer makes Pool respawn workers forever
        _WORKER["error"] = repr(e)


def _pool_task(task: tuple) -> dict:
    if "error" in _WORKER:
        raise RuntimeError(f"worker initialization failed: {_WORKER['error']}")
    chunk, cfg, args = task
    return process_chunk(chunk, cfg, args, _WORKER["tok"], _WORKER["template"])


class ChunkPool:
    def __init__(self, workers: int, tokenizer: str, num_perm: int, near: bool):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.size = workers
        self._pool = mp.get_context("spawn").Pool(
            workers, initializer=_init_worker, initargs=(tokenizer, num_perm, near))

    def submit(self, task: tuple):
        return self._pool.apply_async(_pool_task, (task,))

    def close(self, terminate: bool = False) -> None:
        if terminate:
            self._pool.terminate()
        else:
            self._pool.close()
        self._pool.join()


def resolve_workers(requested: int) -> int:
    """0 = auto: leave a core for the parent (reading, dedup, writing), cap to keep the parent the bottleneck-free."""
    if requested and requested > 0:
        return requested
    return max(1, min((os.cpu_count() or 2) - 1, 12))


# ----------------------------------------------------------------------------
# Per-dataset processing
# ----------------------------------------------------------------------------
def process_dataset(cfg: dict, args, tok: TokenCounter, dd: Deduper, out: Path,
                    pool: Optional[ChunkPool] = None) -> dict:
    name, group = cfg["name"], cfg.get("group", "misc")
    near_ok = cfg.get("near_dedup", True) and not args.no_near_dedup
    conds = cfg.get("where") or []
    tag = paint(f"[{name}]", "cyan")

    stats: Counter = Counter()
    timings: Counter = Counter()
    ddir = out / group / name
    writer = ShardWriter(ddir, args.shard_size, args.gzip)
    for stale in ("_index.parquet", "_index.csv", "_manifest.json"):
        (ddir / stale).unlink(missing_ok=True)
    index = IndexWriter(ddir, args.index_batch_size)
    shards: dict[str, Counter] = defaultdict(Counter)
    logged_errors = 0
    t0 = time.perf_counter()

    def progress():
        rate = stats["seen"] / max(time.perf_counter() - t0, 1e-9)
        log.info("%s seen=%s kept=%s tokens=%s  %s", tag, paint(fmt_int(stats["seen"]), "bold"),
                 paint(fmt_int(stats["kept"]), "green"), fmt_tokens(stats["tokens_kept"]),
                 paint(f"{rate:,.0f} rows/s", "magenta"))

    def chunks() -> Iterator[list]:
        """Parent-side: stream rows, apply `where` (cheap, avoids shipping dropped rows to workers)."""
        rows = iter(islice(load_rows(cfg, args), args.limit or None))
        chunk: list = []
        i = -1
        while True:
            started = time.perf_counter()
            row = next(rows, _END)
            timings["load"] += time.perf_counter() - started
            if row is _END:
                break
            i += 1
            stats["seen"] += 1
            if stats["seen"] % PROGRESS_EVERY == 0:
                progress()
            started = time.perf_counter()
            failed = [c for c in conds if not passes_where(row, [c])]
            timings["filter"] += time.perf_counter() - started
            if failed:
                stats["drop_where"] += 1
                for c in failed:
                    stats[f"where_fail:{c['field']}"] += 1
                continue
            chunk.append((i, row))
            if len(chunk) >= args.batch_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def results() -> Iterator[dict]:
        if pool is None:
            template = dd.minhash_template if dd.lsh is not None else None
            for chunk in chunks():
                yield process_chunk(chunk, cfg, args, tok, template)
            return
        pending: deque = deque()  # bounded in-flight window = backpressure on the row stream

        def wait(async_result) -> dict:
            started = time.perf_counter()
            res = async_result.get()
            timings["wait_workers"] += time.perf_counter() - started
            return res

        for chunk in chunks():
            pending.append(pool.submit((chunk, cfg, args)))
            if len(pending) >= pool.size * 3:
                yield wait(pending.popleft())
        while pending:
            yield wait(pending.popleft())

    def consume(res: dict) -> None:
        nonlocal logged_errors
        stats.update(res["stats"])
        for k, v in res["cpu"].items():
            timings[f"cpu_{k}"] += v
        for err in res["errors"]:
            if logged_errors < 3:
                logged_errors += 1
                log.warning("%s adapter error: %s", tag, err)
        for it in res["items"]:
            started = time.perf_counter()
            if dd.exact_dup(it["ch"]):
                timings["dedup"] += time.perf_counter() - started
                stats["drop_exact_dup"] += 1
                continue
            if near_ok and dd.near_dup_sig(it["ch"], it["sig"], it["scope"]):
                timings["dedup"] += time.perf_counter() - started
                stats["drop_near_dup"] += 1
                continue
            timings["dedup"] += time.perf_counter() - started
            started = time.perf_counter()
            shard, row_i = writer.write_line(it["line"])
            index.write({"id": it["sid"], "source": name, "group": group, "shard": shard,
                         "row": row_i} | it["meta"])
            shards[shard].update(n_samples=1, tokens_total=it["total"], tokens_trainable=it["trainable"])
            timings["write"] += time.perf_counter() - started
            stats["kept"] += 1
            stats["tokens_kept"] += it["total"]

    complete = False
    dd.begin()
    try:
        for res in results():
            consume(res)
        eligible = stats["seen"] - stats["drop_where"]
        if eligible and stats["drop_adapter_none"] + stats["drop_adapter_error"] == eligible:
            raise ValueError(
                f"[{name}] adapter could not normalize any rows; run inspect and configure "
                "messages_field or user_field/assistant_field"
            )
        complete = True
    finally:
        try:
            if complete:
                dd.commit()
                writer.close()
            else:
                dd.rollback()  # samples of a failed dataset must not suppress later duplicates
                writer.discard()  # and no partial shards are left behind
        finally:
            started = time.perf_counter()
            index.close(complete)
            timings["write"] += time.perf_counter() - started

    manifest = {
        "source": name, "group": group, "n_samples": stats["kept"],
        "tokens_total": stats["tokens_kept"],
        "tokens_trainable": sum(s["tokens_trainable"] for s in shards.values()),
        "tokenizer": tok.name, "shard_size": args.shard_size,
        "content_hash_version": 2, "id_method": "source:raw_row_index",
        "shards": [{"file": file, **counts} for file, counts in shards.items()],
    }
    (ddir / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    result = dict(stats)
    result["workers"] = pool.size if pool else 1
    result["timings_seconds"] = {k: round(v, 3) for k, v in timings.items()}
    return result


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def load_config(path: str) -> list[dict]:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["datasets"]


def select(cfgs: list[dict], only: Optional[list[str]]) -> list[dict]:
    return [c for c in cfgs if not only or c["name"] in only]


def cmd_inspect(args):
    for cfg in select(load_config(args.config), args.only):
        print(paint("=" * 100, "gray"))
        print(paint(cfg["name"], "bold", "cyan"), paint(f"({cfg.get('repo', cfg.get('path'))})", "gray"))
        adapt = ADAPTERS[cfg.get("adapter", "generic")]
        for i, row in enumerate(load_rows(cfg, args)):
            if i >= args.n:
                break
            print(paint(f"\n--- raw row {i}: keys={list(row.keys())}", "blue"))
            print(json.dumps(row, ensure_ascii=False, default=str)[:1200])
            if not passes_where(row, cfg.get("where")):
                print(paint(">> filtered by `where`", "yellow"))
                continue
            try:
                s = adapt(row, cfg)
                if s is None:
                    print(paint(">> ADAPTER RETURNED None  (fix adapter / messages_field in YAML)", "red", "bold"))
                    continue
                s, reason = finalize(s, cfg, args)
            except Exception as e:  # noqa - show the failing row instead of aborting the inspection
                print(paint(f">> ADAPTER/FINALIZE ERROR: {e!r}", "red", "bold"))
                continue
            if s is None:
                print(paint(f">> dropped by finalize: {reason}", "yellow"))
            else:
                print(paint(">> normalized:", "green", "bold"), json.dumps(s, ensure_ascii=False)[:1500])


def flatten_paths(obj: Any, prefix: str = "", depth: int = 4) -> Iterator[tuple[str, str, str]]:
    obj = maybe_json(obj)
    if isinstance(obj, dict) and depth > 0:
        for k, v in obj.items():
            yield from flatten_paths(v, f"{prefix}.{k}" if prefix else str(k), depth - 1)
    else:
        yield prefix, type(obj).__name__, str(obj).replace("\n", " ")[:60]


def cmd_wherecheck(args):
    """Diagnose `where` filters on a sample: real nested keys, value distributions, per-condition pass rates."""
    for cfg in select(load_config(args.config), args.only):
        conds = cfg.get("where") or []
        if not conds:
            continue
        print(paint("=" * 100, "gray"))
        print(paint(cfg["name"], "bold", "cyan") + f": sampling first {args.sample} rows")
        stat = [Counter() for _ in conds]
        vals: list[list] = [[] for _ in conds]
        all_pass = n = 0
        shown_roots: set[str] = set()
        columns: list = []
        for i, row in enumerate(load_rows(cfg, args)):
            if i >= args.sample:
                break
            n += 1
            columns = list(row.keys())
            ok_all = True
            for k, c in enumerate(conds):
                root = c["field"].split(".")[0]
                if root not in shown_roots and root in row:
                    shown_roots.add(root)
                    print(f"\n[schema of '{root}' in first row that has it]")
                    for path, typ, ex in flatten_paths(row[root], root):
                        print(f"   {path:70s} {typ:6s} e.g. {ex!r}")
                v = get_path(row, c["field"])
                if v is None:
                    stat[k]["missing"] += 1
                    ok_all = False
                    continue
                vals[k].append(v)
                if passes_where(row, [c]):
                    stat[k]["pass"] += 1
                else:
                    ok_all = False
            all_pass += ok_all
        if shown_roots != {c["field"].split(".")[0] for c in conds}:
            print(paint("\n!! some root fields are not top-level columns of the row. Columns seen:", "red", "bold"),
                  columns)
        print(paint(f"\n{'condition':85s} {'pass%':>7s} {'missing%':>9s}  values", "bold"))
        for k, c in enumerate(conds):
            desc = f"{c['field']} {c.get('op', 'eq')} {c.get('value')}"
            v = vals[k]
            nums = []
            for x in v:
                try:
                    nums.append(float(x))
                except (TypeError, ValueError):
                    pass
            if v and len(nums) == len(v) and len(set(v)) > 8:
                summ = f"min={min(nums):.3g} mean={sum(nums) / len(nums):.3g} max={max(nums):.3g}"
            else:
                summ = str(Counter(map(str, v)).most_common(8))
            pass_pct = 100 * stat[k]["pass"] / max(n, 1)
            pass_txt = paint(f"{pass_pct:6.1f}%", "green" if pass_pct >= 50 else "yellow" if pass_pct > 0 else "red")
            print(f"{desc:85s} {pass_txt} {100 * stat[k]['missing'] / max(n, 1):8.1f}%  {summ}")
        print(f"\nALL conditions pass: {all_pass}/{n} = {100 * all_pass / max(n, 1):.1f}%")


def run_summary_line(name: str, st: dict) -> str:
    """One colored, human-readable line per dataset (the full dict stays in build_stats.json)."""
    if "error" in st:
        return f"{paint('<<<', 'red', 'bold')} {paint(name, 'bold')} {paint('FAILED', 'red', 'bold')} {paint(st['error'], 'red')}"
    drops = {k[5:]: v for k, v in st.items() if k.startswith("drop_") and v}
    seconds = st.get("seconds", 0) or 0
    rate = st.get("seen", 0) / seconds if seconds else 0
    drop_txt = " ".join(f"{k}={fmt_int(v)}" for k, v in sorted(drops.items(), key=lambda kv: -kv[1])) or "none"
    return (f"{paint('<<<', 'green', 'bold')} {paint(name, 'bold')}  "
            f"kept {paint(fmt_int(st.get('kept', 0)), 'green', 'bold')}  tokens {paint(fmt_tokens(st.get('tokens_kept', 0)), 'cyan')}  "
            f"seen {fmt_int(st.get('seen', 0))}  drops: {paint(drop_txt, 'yellow')}  "
            f"{paint(fmt_duration(seconds), 'magenta')} ({rate:,.0f} rows/s, workers={st.get('workers', 1)})")


def cmd_build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfgs = select(load_config(args.config), args.only)
    tok = TokenCounter(args.tokenizer)
    dd = Deduper(not args.no_near_dedup, args.near_threshold, args.num_perm)
    stats_path = out / "build_stats.json"
    all_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    todo = [c for c in cfgs if c.get("enabled", True) is not False or args.only]
    workers = resolve_workers(args.workers)
    pool = None
    if workers > 1 and todo:
        log.info("starting %s worker processes (tokenizer=%s)", paint(workers, "bold", "magenta"), args.tokenizer)
        pool = ChunkPool(workers, args.tokenizer, args.num_perm, not args.no_near_dedup)
    else:
        log.info("running single-process (--workers %s)", args.workers)

    run: list[tuple[str, dict]] = []
    ok = False
    try:
        for cfg in cfgs:
            name = cfg["name"]
            if cfg.get("enabled", True) is False and not args.only:
                log.info("skip %s (enabled: false)", paint(name, "gray"))
                continue
            log.info("%s %s", paint(">>>", "blue", "bold"), paint(name, "bold"))
            rev = cfg.get("revision")
            if api and cfg.get("repo"):
                try:
                    rev = api.dataset_info(cfg["repo"], revision=rev).sha
                except Exception as e:
                    log.warning("[%s] could not resolve dataset revision: %s", name, e)
            try:
                resolved_cfg = cfg | {"revision": rev} if rev else cfg
                st = process_dataset(resolved_cfg, args, tok, dd, out, pool)
            except Exception as e:  # keep going on failures (gated / schema change / etc.)
                log.exception("FAILED %s", name)
                st = {"error": repr(e)}
            st |= {"repo": cfg.get("repo"), "revision": rev, "group": cfg.get("group"),
                   "run_at": datetime.now(timezone.utc).isoformat()}
            all_stats[name] = st
            stats_path.write_text(json.dumps(all_stats, indent=2))
            run.append((name, st))
            log.info("%s", run_summary_line(name, st))
        ok = True
    finally:
        if pool is not None:
            pool.close(terminate=not ok)

    if run:
        print(paint("\nRun summary", "bold"))
        for name, st in run:
            print("  " + run_summary_line(name, st))
    cmd_report(args)


def cmd_report(args):
    import pandas as pd
    out = Path(args.out)
    by_directory = {f.parent: f for f in sorted(out.glob("*/*/_index.csv"))}
    by_directory.update({f.parent: f for f in sorted(out.glob("*/*/_index.parquet"))})
    files = list(by_directory.values())
    if not files:
        print("no analysis files found")
        return
    columns = ["source", "group", "tokens_total", "tokens_trainable", "n_tool_calls", "n_messages", "lang"]
    summaries, languages = [], []
    q = lambda p: (lambda s: s.quantile(p))  # noqa
    for f in files:
        df = pd.read_parquet(f, columns=columns) if f.suffix == ".parquet" else pd.read_csv(f, usecols=columns)
        if df.empty:
            continue
        summaries.append(df.groupby("source").agg(
            group=("group", "first"), samples=("tokens_total", "size"),
            tokens=("tokens_total", "sum"), trainable=("tokens_trainable", "sum"),
            mean_tok=("tokens_total", "mean"), p50=("tokens_total", q(0.5)), p95=("tokens_total", q(0.95)),
            tool_call_rate=("n_tool_calls", lambda s: (s > 0).mean()),
            avg_turns=("n_messages", "mean")))
        languages.append(pd.crosstab(df["source"], df["lang"]))
        del df
    if not summaries:
        print("no samples found in analysis files")
        return
    per = pd.concat(summaries).sort_index().round(2)
    lang = pd.concat(languages).fillna(0).astype("int64").sort_index().sort_index(axis=1)
    cmp_ = per.groupby("group")[["samples", "tokens"]].sum()
    cmp_["%tokens_of_total"] = (cmp_.tokens / cmp_.tokens.sum() * 100).round(1)
    lines = [f"# SFT build report ({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC)\n",
             f"Total samples: {per.samples.sum():,}   total tokens: {per.tokens.sum():,}\n",
             "## Per source\n```", per.to_string(), "```\n",
             "## Language distribution (samples per source)\n```", lang.to_string(), "```\n",
             "## Group totals\n```", cmp_.to_string(), "```\n"]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(paint(ln, "bold", "cyan") if ln.startswith("#") else ln for ln in lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("inspect", "wherecheck", "build", "report"):
        p = sub.add_parser(n)
        p.add_argument("--config", default="datasets.yaml")
        p.add_argument("--out", default="out")
        p.add_argument("--only", nargs="*", help="dataset names from the YAML")
        p.add_argument("--n", type=int, default=2, help="(inspect) rows to show")
        p.add_argument("--sample", type=int, default=2000, help="(wherecheck) rows to sample")
        p.add_argument("--limit", type=int, default=0, help="max raw rows per dataset (testing)")
        p.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct", help="HF tokenizer id or 'approx'")
        p.add_argument("--max-tokens", type=int, default=0,
                       help="optional hard per-sample cap (0 = keep everything; the loader can filter on meta.tokens_total)")
        p.add_argument("--min-assistant-tokens", type=int, default=2)
        p.add_argument("--msg-overhead", type=int, default=4, help="chat-template tokens added per message")
        p.add_argument("--no-near-dedup", action="store_true")
        p.add_argument("--no-streaming", action="store_true",
                       help="use disk-backed HF datasets (initial download/prepare required); "
                            "per-source streaming settings override this")
        p.add_argument("--near-threshold", type=float, default=0.85)
        p.add_argument("--num-perm", type=int, default=64)
        p.add_argument("--keep-invalid-tools", dest="drop_invalid_tools", action="store_false")
        p.add_argument("--gzip", action="store_true")
        p.add_argument("--shard-size", type=int, default=10000)
        p.add_argument("--batch-size", type=int, default=256,
                       help="rows per work chunk (tokenizer batch and unit of parallel work)")
        p.add_argument("--index-batch-size", type=int, default=10000,
                       help="rows buffered per Parquet index row group (bounded memory)")
        p.add_argument("--workers", type=int, default=0,
                       help="worker processes for normalize/tokenize/hash/language work "
                            "(0 = auto: min(cpu-1, 12); 1 = single process). Output is identical for any value")
        p.add_argument("--color", choices=["auto", "always", "never"], default="auto",
                       help="colored terminal output (auto: only on a TTY; NO_COLOR disables)")
    args = ap.parse_args()
    for key in ("batch_size", "index_batch_size", "shard_size", "num_perm"):
        if getattr(args, key) <= 0:
            ap.error(f"--{key.replace('_', '-')} must be positive")
    if args.limit < 0:
        ap.error("--limit must be non-negative")
    if args.workers < 0:
        ap.error("--workers must be >= 0")
    setup_logging(args.color)
    {"inspect": cmd_inspect, "wherecheck": cmd_wherecheck, "build": cmd_build, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()