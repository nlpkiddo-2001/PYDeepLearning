"""Hermetic fixtures: a tiny ChatML tokenizer and a fake `out/` tree in the build_sft_dataset.py layout."""
from __future__ import annotations

import hashlib
import json
import random
import tempfile
from pathlib import Path

import pandas as pd

CHATML = (
    "{%- if tools %}{{ '<|im_start|>system\\n' }}{% for t in tools %}{{ t | tojson }}{{ '\\n' }}{% endfor %}{{ '<|im_end|>\\n' }}{%- endif %}"
    "{%- for m in messages %}{{ '<|im_start|>' + m.role + '\\n' + m.content }}"
    "{% if m.tool_calls %}{% for c in m.tool_calls %}{{ '\\n<tool_call>\\n' }}{{ c.function | tojson }}{{ '\\n</tool_call>' }}{% endfor %}{% endif %}"
    "{{ '<|im_end|>\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
)
# renders the final message differently from the same message in a prefix -> not prefix-consistent
INCONSISTENT = CHATML.replace("{{ '<|im_start|>' + m.role + '\\n' + m.content }}",
                              "{{ '<|im_start|>' + m.role + '\\n' + m.content + ('!' if loop.last and m.role == 'assistant' else '') }}")


def make_tokenizer_dir(path: Path, template: str = CHATML) -> str:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    corpus = [f"word{i} answer question tool call result {i} " * 3 for i in range(200)]
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=500, special_tokens=["<unk>", "<|im_start|>", "<|im_end|>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<|im_end|>", unk_token="<unk>")
    fast.chat_template = template
    fast.save_pretrained(path)
    return str(path)


def make_sample(source: str, i: int, rng: random.Random, kind: str = "chat", scale: int = 1, turns: int = 0) -> dict:
    words = lambda n: " ".join(f"word{rng.randrange(500)}" for _ in range(n * scale))
    msgs = [{"role": "system", "content": "You are helpful."}] if i % 3 == 0 else []
    tools: list = []
    turns = turns or rng.randint(1, 3)
    for t in range(turns):
        msgs.append({"role": "user", "content": f"question {i}.{t} " + words(rng.randint(3, 12))})
        if kind == "tool":
            tools = [{"type": "function", "function": {"name": "lookup", "description": "d",
                                                         "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{t}", "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"q": f"x{i}"})}}]})
            msgs.append({"role": "tool", "content": f"result {i}.{t} " + words(rng.randint(3, 20)), "tool_call_id": f"c{t}", "name": "lookup"})
        msgs.append({"role": "assistant", "content": f"answer {i}.{t} " + words(rng.randint(3, 25))})
    body = json.dumps(msgs)
    h = hashlib.blake2b(f"{source}{i}{body}".encode(), digest_size=16).hexdigest()
    return {"id": f"{source}:{i}", "source": source, "group": "g", "messages": msgs, "tools": tools,
            "meta": {"content_hash": h, "tool_valid": True, "lang": "en", "n_tool_calls": turns if kind == "tool" else 0,
                     "n_messages": len(msgs), "license": "MIT"}}


def write_source(root: Path, group: str, name: str, n: int, shard_size: int = 7, kind: str = "chat", scale: int = 1,
                 seed: int = 0, license_name: str = "MIT", mutate=None, turns: int = 0) -> list[dict]:
    d = root / group / name
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"{name}{seed}")
    samples = [make_sample(name, i, rng, kind, scale, turns) for i in range(n)]
    if mutate:
        for s in samples:
            mutate(s)
    shards = []
    for k in range(0, n, shard_size):
        chunk = samples[k:k + shard_size]
        fn = f"part-{k // shard_size:05d}.jsonl"
        (d / fn).write_text("".join(json.dumps(s) + "\n" for s in chunk), encoding="utf-8")
        shards.append({"file": fn, "n_samples": len(chunk)})
    (d / "_manifest.json").write_text(json.dumps({"source": name, "group": group, "n_samples": n, "shards": shards}))
    pd.DataFrame({"id": [s["id"] for s in samples], "license": license_name}).to_parquet(d / "_index.parquet", index=False)
    return samples


class Workspace:
    """Temp dir with tokenizer + fake corpus + config dict; use as a context manager."""

    def __init__(self, template: str = CHATML):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tokenizer = make_tokenizer_dir(self.root / "tok", template)
        self.data = self.root / "out"
        self.cache = self.root / "cache"

    def cfg_dict(self, **over) -> dict:
        base = {"data_root": str(self.data), "cache_dir": str(self.cache), "tokenizer": self.tokenizer,
                "seq_len": 256, "pack_window": 64, "val_fraction": 0.1, "val_max_samples_per_source": 50,
                "domains": {
                    "chat": {"weight": 0.5, "sources": {"short": {}}},
                    "tools": {"weight": 0.3, "sources": {"toolish": {}}},
                    "long": {"weight": 0.2, "sources": {"longdocs": {}}}}}
        base.update(over)
        return base

    def populate(self, n_short=120, n_tool=80, n_long=30):
        self.short = write_source(self.data, "g1", "short", n_short, 7)
        self.toolish = write_source(self.data, "g2", "toolish", n_tool, 9, kind="tool")
        self.longdocs = write_source(self.data, "g3", "longdocs", n_long, 5, turns=3)
        return self

    def compile(self, cfg, workers: int = 1, chunk_rows: int = 11, force: bool = False):
        from Instruction_taining.sft_data.catalog import discover
        from Instruction_taining.sft_data.compile import CompileJob, compile_sources, make_cache_spec
        spec, tok, template = make_cache_spec(cfg.tokenizer, cfg.chat_template, cfg.cache_dir)
        found = discover(cfg.data_root)
        jobs = [CompileJob(s.name, found[s.name].path, found[s.name].license, found[s.name].shard_rows)
                for _, s in cfg.enabled_sources()]
        compile_sources(jobs, spec, workers=workers, chunk_rows=chunk_rows, force=force)
        return spec, tok, template

    def close(self):
        self.tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
