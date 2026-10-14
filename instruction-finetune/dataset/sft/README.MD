# Data Model

Two tools turn raw instruction datasets into training batches for the 4 domains (chat, coding, tool calling, roleplay):

* **`dataset/`**: `build_dataset.py` downloads and normalizes every source into one chat+tools schema, and
  `decontaminate.py` flags training samples that overlap evaluation benchmarks.
* **`loader/`**: compiles the corpus to a token cache and serves **token-budget domain mixing, assistant-only loss masks,
  sequence packing, deterministic + exactly resumable streams, held-out validation, and a proxy-model mixture sweep.**
  Framework-agnostic core (numpy); thin torch adapter. The Python package is `my_training_codes.loader`.

```
datasets.yaml --build--> out/<group>/<source>/part-*.jsonl --decontaminate--> _contamination.parquet
                                      |                                              |
                                      +---------------- compile ---------------------+
                                                         v
                          token cache (ids + loss mask, memory-mapped)   <- sft_cache/
                                      |  catalog: filters, benchmark-overlap drop, exact cuts, val split
                                      v
        mixture sampler (token deficit round-robin, epoch caps, stages, rank sharding)
                                      v
              packer (best-fit window, doc boundaries) --> batches --> your trainer
```

## Layout and conventions
```
Instruction_taining/
├── my_training_codes/
│   ├── dataset/   build_sft_dataset.py  decontaminate.py  datasets.yaml  benchmarks.yaml  testing/
│   └── loader/    cli.py  loader.py  config.py  catalog.py  compile.py  render.py  sampler.py  packer.py  sweep.py  mixes/  tests/
├── out/           built corpus: <group>/<source>/{part-*.jsonl,_index.parquet,_manifest.json,_contamination.parquet}, report.md, decontam_report.md
└── sft_cache/     token cache (safe to delete; rebuilt by `compile`)
```
**Run every command in this README from `Instruction_taining/`.** The mix config uses relative paths (`data_root: out`,
`cache_dir: sft_cache`) and `decontaminate.py`'s default `--out ../out` is only right from the old `dataset/` location, so
always pass `--out out`.

Dependencies: `datasets`, `transformers`, `huggingface_hub`, `pyyaml`, `pandas`, `pyarrow`, `numpy`, `datasketch` (near-dedup);
`torch` only for `PackedDataset` / training. Set `HF_TOKEN` for gated datasets.

## Pipeline at a glance
| step | command | writes | run again when |
|---|---|---|---|
| 1. build | `python my_training_codes/dataset/build_sft_dataset.py build ...` | `out/<group>/<source>/` | `datasets.yaml` or an adapter changed |
| 2. decontaminate | `python my_training_codes/dataset/decontaminate.py --out out` | `<source>/_contamination.parquet`, `out/decontam_report.md` | **a source was rebuilt with different content**, or `benchmarks.yaml` changed |
| 3. compile | `python -m my_training_codes.loader.cli compile --config <mix>` | `sft_cache/<fingerprint>/` | tokenizer / template / `token_config` changed, or a source was rebuilt |
| 4. check | `... cli stats / validate / preview --config <mix>` | nothing | before every real run |
| 5. train | your code (below) | checkpoints | |

After rebuilding a source: **build → decontaminate → compile**, in that order (the loader checks the token cache first, so
a stale cache error hides a stale decontamination result until you fix it).

## 1. Build the dataset
`datasets.yaml` lists the sources in dedup priority order (an earlier source keeps a duplicate, later ones lose it). Entries
with `enabled: false` are skipped. Always look at a source before a full build:
```bash
cd ${PROJECT_DIR}
python -m dataset.sft.build_dataset inspect    --config dataset/sft/configs/sft_mix_1.0.yaml --only swe_smith --n 3      # raw row, normalized row, why a row would be dropped
python -m dataset.sft.build_dataset wherecheck --config dataset/sft/configs/sft_mix_1.0.yaml --only toucan_sft --sample 2000   # does the `where:` filter pass what you expect
python -m dataset.sft.build_dataset build      --config dataset/sft/configs/sft_mix_1.0.yaml --out out --workers 16 --batch-size 512
python -m dataset.sft.build_dataset report  --config dataset/sft/configs/sft_mix_1.0.yaml --out out                                           # rewrite out/report.md from the index files
```
* `--only a b` builds just those sources (it does **not** dedup against sources built in earlier invocations; rebuild all
  sources in YAML order when dedup priority matters). `--limit N` caps raw rows per source: use it for dry runs.
* `--workers N` (0 = auto) parallelizes normalize/tokenize/hash; output is byte-identical for any N. Download and global
  dedup stay single-threaded, so check `timings_seconds` in `out/build_stats.json` before adding workers.
* Useful flags: `--no-near-dedup` (faster, keeps near-duplicates), `--max-tokens N` (hard per-sample cap),
  `--tokenizer Qwen/Qwen2.5-7B-Instruct` (the counting tokenizer; `approx` for dry runs only), `--keep-invalid-tools`,
  `--no-streaming`, `--color auto|always|never`.
* **Do not use `--gzip` if you will decontaminate**: `decontaminate.py` skips gzip shards, and the loader then has no result for
  that source.
* Don't run two builds into the same `--out` at once. The first serial build of opencodeinstruct (5M rows) took ~2 h.
* A source that fails (gated, schema change) is recorded as `error` in `out/build_stats.json` and the build continues; check it.
----

Per sample (JSONL) the schema is OpenAI-style: `messages` (system/user/assistant with `tool_calls`/tool), `tools`, and `meta`
(`tokens_*`, `lang`, `content_hash`, `tool_valid`, `license`, ...). IDs are `<source>:<raw row index>`. Counts in `meta` are
estimates; the loader's `compile` step computes exact token counts.

Per-source YAML options live in the `build_sft_dataset.py` docstring (`adapter`, `where`, `max_tokens`, `near_dedup`,
`splits`, `data_files`, `default_tools`, ...). Two that matter for agent data: `adapter: function_tags` (SWE-smith: text tool
definitions and `<function=..>` calls become structured `tools` / `tool_calls` / `tool` messages) and
`drop_if_last_assistant_matches` (drops harness lines such as "Exit due to ...").

## 2. Decontaminate against benchmarks
`decontaminate.py` never edits the corpus. It scans every shard and writes, per source, `_contamination.parquet` (one row
per flagged sample: `id, n_hits, benchmarks, items, max_item_coverage`) plus `out/decontam_report.md` and
`out/decontam_stats.json`. The loader then drops flagged samples at catalog time.

```bash
python -m dataset.sft.decontaminate --out out --benchmarks dataset/sft/configs/benchmarks.yaml                           # all sources, all enabled benchmarks (~35 s for 2 small sources incl. downloads)
python -m dataset.sft.decontaminate --out out --benchmarks dataset/sft/configs/benchmarks.yaml  --only when2call         # just one source (after rebuilding it)
python -m dataset.sft.decontaminate --out out --benchmarks dataset/sft/configs/benchmarks.yaml  --out out --enable livecodebench   # also run benchmarks marked `enabled: false` (livecodebench is a 4.2 GB download)
```
`--workers N` (0 = cpu-1), `--benchmarks path.yaml` (default: `benchmarks.yaml` next to the script). It needs network access
to fetch benchmark data and uses a `fork` pool (Linux/macOS).

**What counts as overlap.** A sample is flagged when any message (content **and** tool-call arguments) shares a run of
`ngram` (13) lowercased words with a benchmark item's text. Items shorter than `ngram` words but at least `min_words` (8)
are matched whole. N-grams shared by more than `max_items_per_ngram` (5) benchmark items are treated as template text and
ignored, so IFEval-style boilerplate does not flag every instruction-following sample.

**`benchmarks.yaml` entries** (only put text the model must NOT have seen: prompts and reference solutions, not tool schemas):
```yaml
- name: humaneval                      # used in reports
  repo: openai/openai_humaneval
  split: test                          # or `all`; `config:` for HF configs
  fields: [prompt, canonical_solution] # dotted paths; every string under them is benchmark text
  id_field: task_id
# loader: hf_files + files: [...]      # for repos shipping plain JSON/JSONL (BFCL, LiveCodeBench)
# enabled: false                       # opt-in
```
A benchmark that fails to load **aborts the run** (an incomplete check must not look clean): fix it, or set
`enabled: false` and accept that it is unchecked. Currently covered: humaneval, mbpp, mbppplus, swebench_verified,
when2call_test, bfcl_v3, ifeval, mt_bench, arena_hard, alpaca_eval, gsm8k (livecodebench opt-in).

**Choose the threshold from the report.** `max_item_coverage` is the share of the matched benchmark item's n-grams found in
the sample: ~1.0 is a copied benchmark problem, a small value is one shared sentence. Read `out/decontam_report.md`, then set
`decontam_min_coverage` in the mix config (0 drops any hit). Example from the 2026-10-12 run on the full corpus:

| source | samples | flagged (any hit) | coverage ≥ 0.1 | coverage ≥ 0.5 |
|---|---:|---:|---:|---:|
| smoltalk | 1,032,149 | 20,456 | 1,425 | 306 |
| opencodeinstruct | 1,581,744 | 27,750 | 9,477 | 4,744 |
| swe_smith | 9,284 | 1,244 | 163 | 0 |
| toucan_sft | 90,320 | 214 | 95 | 91 |
| glaive_fc | 90,509 | 299 | 124 | 99 |

Most of smoltalk's 20K hits are IFEval-style instruction sentences with low coverage, which is why `decontam_min_coverage: 0`
would discard far more than `0.1` does.

**How the loader uses it** (`decontam:` in the mix config):

| value | behaviour |
|---|---|
| `required` | every source in the mix needs a current result, otherwise the loader raises (use this for real runs) |
| `if_present` | uses results where they exist, warns `training on unchecked data` where they don't |
| `off` | ignore decontamination |

A result is *current* when its recorded hash of the source's `_manifest.json` matches. Rebuilding a source with different
content makes it stale and the loader raises `_contamination.parquet predates the current build; rerun dataset/decontaminate.py`.
A deterministic rebuild that produces a byte-identical manifest does not. Dropped samples show up as `benchmark_overlap=N`
in the `rejected` column of `loader.cli stats`.

**Limits.** This is lexical overlap only: paraphrased or translated benchmark items, benchmarks missing from
`benchmarks.yaml`, and benchmark text outside the listed `fields` are not caught. The validation split the loader holds out
of the training sources is not a decontaminated benchmark; evaluate on real benchmarks you have checked here.

## 3. Compile, check and inspect the mix
```bash
python -m dataset.sft.loader.cli compile \
  --config dataset/sft/configs/sft_dataloader.yaml \
  --workers 3    # once per tokenizer/template (1.9B tokens took ~2.5 min with 64 workers); resumable
python -m dataset.sft.loader.cli stats    --config dataset/sft/configs/sft_dataloader.yaml --budget-tokens 1e9 --world-size 8   # realized shares, epochs per source, what survives at each seq_len
python -m my_training_codes.loader.cli validate --config $M                       # cache == fresh render, packing invariants
python -m dataset.sft.loader.cli preview  --config dataset/sft/configs/sft_dataloader.yaml                    # decode a pack, trained tokens in green
```
`compile` flags: `--sources a b` (subset), `--force` (recompile), `--chunk-rows N`. `stats` builds the catalog, so it is also
the quickest way to see decontamination, license and filter problems (`rejected` column, `skipped ...` lines).
`stats` reports `!! budget NOT reachable` when the epoch caps cannot supply the token budget you asked for.

## 4. Load the data in your training code
Import the package from `Instruction_taining/` (a script placed there, `python -m ...` from there, or
`PYTHONPATH=/path/to/Instruction_taining`). `my_training_codes` is a namespace package, so no install is needed. The config's
relative `data_root`/`cache_dir` resolve against the **current working directory**; use absolute paths if you launch from elsewhere.

```python
import json
from torch.utils.data import DataLoader
from dataset.sft.loader import MixConfig, PackedStream, PackedDataset

cfg    = MixConfig.from_yaml("my_training_codes/loader/mixes/baseline.yaml")   # validates; raises ConfigError with a fix hint
stream = PackedStream.from_config(cfg, rank=rank, world_size=world)           # each rank draws its own disjoint documents
ds     = PackedDataset(stream, prefetch=2)                                     # torch IterableDataset; as_torch=False -> numpy dicts
loader = DataLoader(ds, batch_size=None, num_workers=0)                        # num_workers MUST be 0 (resumable state lives in-process)
val    = {v.name: stream.eval_batches(v.name) for v in stream.views}           # fixed held-out batches per source; same count on every rank

for batch in loader:        # tensors shaped [cfg.micro_batch_size, cfg.seq_len]
    ...                     # input_ids, labels, position_ids, doc_ids, cu_seqlens, max_seqlen, n_trainable, n_real

ckpt_state = json.dumps(ds.state_dict())                  # save per rank, next to the model checkpoint
ds.load_state_dict(json.loads(ckpt_state))                # resume: BEFORE iterating; raises StateMismatchError if world size/config/sources changed
```
Settings can be changed in code before building the stream: `cfg = MixConfig.from_yaml(p); cfg.micro_batch_size = 1`
(or build one from a dict with `MixConfig.from_dict(...)`).

Batch keys: `input_ids, labels, position_ids` (restart per document), `doc_ids` (-1 = padding; build a block-diagonal mask),
`cu_seqlens`/`max_seqlen` (flash-attn varlen over the flattened batch), `n_trainable`, `n_real`.
`labels` are not pre-shifted: score `logits[:, :-1]` against `labels[:, 1:]` with `ignore_index=-100`, sum the loss, and divide by
`n_trainable` summed over all ranks and accumulation steps.

## Tests
```bash
cd /workspace/RAG/experiment      # the directory ABOVE Instruction_taining/ (the tests import it by absolute name)
python -m unittest Instruction_taining.my_training_codes.dataset.testing.test_build_sft_dataset
python -m unittest Instruction_taining.my_training_codes.dataset.testing.test_decontaminate
```

## How multi-domain SFT works (and what this implements)
1. **Render** each sample with the model's chat template (tools, tool calls, tool results) -> one tokenization.
2. **Loss mask**: train on assistant text and tool-call tokens only. System/user/tool-result/tool-schema tokens are context.
   Derived from the template itself (prefix diffs + tokenizer offsets), and a sample whose template is not
   prefix-consistent is *rejected*, never mis-masked.
3. **Mix by tokens, not samples.** Weights are shares of packed tokens: `domain weight × within-domain weight`.
   Sample counts mislead (swe_smith is 0.5% of samples, 10% of tokens). Within a domain, sources get explicit
   weights, or — if none are given — `tokens^alpha` smoothing (alpha 0.3-0.7 flattens a dominant source).
4. **Epoch caps** (`max_epochs`, per source override). A capped source stops; the others renormalize *immediately*.
5. **Pack** documents into fixed-length sequences (best-fit over a lookahead window), never splitting a document;
   position ids restart and attention must not cross documents.
6. **Normalize the loss by trained tokens** over the global batch (above), otherwise long agent traces or short chats dominate.
7. **Stages** (optional): e.g. broad mix first, then a final annealing stage on high-quality / long-agentic data.
8. **Find the mix empirically** with proxy models (`my_training_codes.loader.sweep`), then confirm 2-3 candidates at a larger size.
   Mixture optima transfer across scale only approximately.

## Config (`loader/mixes/baseline.yaml`)
| key | meaning |
|---|---|
| `seq_len`, `over_length` | training length; docs longer are cut at the **last complete assistant turn** that fits (`truncate`) or dropped (`drop`). Unusable docs are rejected in the catalog (reported), so sampler accounting is exact |
| `domains.<d>.weight`, `.sources.<s>.weight` | token shares; source weights all-or-none per domain (none = `alpha` smoothing) |
| `max_epochs`, `sources.<s>.max_epochs` | pass cap per source (fractions allowed) |
| `stages: [{until_tokens, domain_weights, source_weights}]` | weight schedule by *global* tokens consumed; deficits reset at every boundary |
| `val_fraction`, `val_max_samples_per_source`, `split_seed` | validation split = pure function of the content hash: identical for every mix, never in train |
| per source: `min_trainable_tokens, max_tokens, langs, require_tool_valid, enabled` | eligibility filters |
| `license_deny: [...]` | case-insensitive substrings of a source's license (e.g. `CC-BY-NC` skips apigen_mt) |
| `decontam: required\|if_present\|off`, `decontam_min_coverage` | drop samples flagged by `dataset/decontaminate.py` (benchmark n-gram overlap, `<source>/_contamination.parquet`); a result older than the source's build is an error |
| `token_config: {instruct_start, assistant_token, instruct_end}` | the special tokens that delimit turns (`<\|im_start\|>`, `assistant`, `<\|im_end\|>` for Qwen). Omit it and the renderer derives them from the chat template, which needs `assistant` to be one token |
| `pack_window` (512), `micro_batch_size`, `seed`, `tokenizer`, `chat_template` | |

## Guarantees (all covered by tests; mutation-checked)
* **Realized mix = target.** Absolute token error never exceeds one long document; at horizon N the relative error is about
  `window_tokens / N` (e.g. 0.108 → 0.002 as a run goes from 26K to 1.6M tokens with a deliberately small window).
* **Exact resume**: sampler + packer window + counters; refuses a different world size / config / sources (`StateMismatchError`).
  Prefetch never loses batches: state is the state after the last batch *you received*.
* **Disjoint, complete ranks**: each rank draws `perm[rank::world]` of the same seeded epoch permutation (no communication);
  draining all ranks at `max_epochs=1` yields every training doc exactly once.
* **Cache correctness**: keyed by tokenizer vocab + merges + template + render version; stale/incomplete caches are detected
  against the shards' bytes and the manifest row counts; `validate` re-renders random samples and compares.

## Findings on the current corpus (snapshot: 1.89B tokens, 8 sources, Qwen2.5 tokenizer, 8K context)
* Compiled token counts match the builder's estimates (swe_smith 227.3M vs 227.2M) — the estimates are reliable.
* **swe_smith loses most of its signal at 8K**: only 5% of trajectories fit whole and 26% of its tokens survive
  (97%/98% at 64K). At 8K it is repeated ~2.5× per 1B tokens for 14% weight — consider a lower weight at 8K plus a
  long-context stage (`stages:`) with `seq_len` 32K+.
* toucan_sft keeps 84% of tokens at 8K; smoltalk loses 2.7K samples whose first assistant turn alone exceeds 8K.
* Epoch caps bind early for small sources (when2call hits its 3-epoch cap at a 2.5% target; roleplay ~2 epochs at 8%) — watch
  these for overfitting in the sweep.
* Packing fill 0.999, ~2.6M tokens/s single process (far above GPU consumption); trained fraction ~45% of packed tokens.
* 1.4% of smoltalk replies start with a newline that the tokenizer merges with the template's `\n` into one token
  that straddles the context/target boundary; that single token is left untrained (counted as `misaligned spans`).
* apigen_mt is excluded by the baseline `license_deny` (CC-BY-NC).

## Proxy-model mixture sweep
```bash
python -m my_training_codes.loader.sweep generate --base my_training_codes/loader/mixes/baseline.yaml --out sweep/r1 --n 24      # row 0 = baseline
python -m my_training_codes.loader.sweep run --dir sweep/r1 --gpus 0,1 \
    --cmd "python train_proxy.py --mix {mix} --out {out}"        # your script writes {out}/metrics.json = {"val_loss/chat": ..., "bfcl": ...}
python -m my_training_codes.loader.sweep fit --dir sweep/r1 --objective sweep/objective.yaml
```
`objective.yaml`: `metrics: {val_loss/chat: {direction: min, weight: 1}, bfcl: {direction: max, weight: 2}}`,
optional `constraints: {min_domain: 0.03, max_domain: {roleplay: 0.15}, min_domain_each: {chat: 0.2}}`.
`fit` runs one ridge regression per metric (linear/quadratic by leave-one-out CV), searches the simplex inside a trust
region around mixes you actually ran, and writes `fit_report.md`, `fit.json`, `recommended/rec_*.yaml` (runnable configs).
It tells you when a metric is **not predictable** from the mix — then more runs or a different metric is needed.
Use identical steps, seed, learning rate and the same validation split for every proxy run; hold the token budget fixed.

## Caveats
* Exact resume requires the same world size and config; `num_workers=0` (use `prefetch`).
* The validation split is by content hash, so near-duplicates of a validation doc can remain in train (dedup is the
  builder's job).
* `labels` are not pre-shifted. First token of every document is always masked.
* The cache is ~5.8 bytes/token (int32 ids + uint8 mask); `sft_cache/<fingerprint>/` can be deleted and rebuilt.
* torch is optional; the loader core runs without it (`PackedDataset(..., as_torch=False)` yields numpy).
* `loader/tests/` still import the old `Instruction_taining.sft_data` package, not `my_training_codes.loader`, so they do not
  test this copy yet. The dataset tools' tests (`dataset/testing/`) do test this copy; see "Tests" below.
