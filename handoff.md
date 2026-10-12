# SFT pipeline handoff

_Last updated: 2026-10-12_

## Goal
Write our own SFT training loop for 2–4B dense models on agentic tasks (tool calling, coding, reasoning; English only). DPO/APO and RLVR loops come after SFT. Another dev owns data collection and the input stream; we own the training loops.

- **First model:** Qwen3-4B (`Qwen/Qwen3-4B`)
- **Hardware:** 1 node, 8× H200
- **Training:** full fine-tuning with FSDP2
- **Python env:** `/root/miniconda3/envs/vllm` (transformers 5.11, torch 2.11). `flash_attn` is **not** installed.

## Decisions already made
- **Data format:** canonical OpenAI/HF chat records (`schema_version`, `id`, `metadata.{source,task_type}`, `tools`, `messages`). Assistant messages may carry `reasoning_content` and a per-turn `train` flag. Tool-call `arguments` are JSON objects. The example record is in the docstring of `sft/data/chat_tokenizer.py`.
- **Tokenization:** render the conversation with the chat template, find each assistant turn's character span by rendering prefixes (with and without the turn), then map spans to tokens with offset mappings.
  - `auto` mode tries one sequence per record and falls back to one sample per turn when the template rewrites history (Qwen3) or the record is too long.
  - Labels are unshifted (`-100` = no loss). Overlong samples are dropped by default.
- **Packing:** concatenate samples into rows of `pack_len = max_seq_len`.
  - `position_ids` restart at 0 for each sample, and each sample's first label is `-100`.
  - The model must get **no `attention_mask`** and **`use_cache=False`**, or `transformers` won't detect the sample boundaries and samples will attend to each other.
- **Best-fit packer: postponed.** Greedy packing reaches about 85% fill on synthetic agentic lengths. Best-fit is an optimization for later.

## Current files

| File | State |
|---|---|
| `sft/data/chat_tokenizer.py` | Done: tokenizer, Qwen3 and default adapters, `render_labels`, `PROBE_RECORD`, `probe_template`. |
| `sft/data/packing.py` | Done: `PackedRow`, `make_row`, `pack_greedy`. |
| `sft/data/normalize.py` | Docstring only. Not needed, because the data team sends the canonical format. |
| `tests/test_chat_tokenizer.py` | 17 tests, all passing. |
| `tests/test_packing.py` | 8 tests, all passing. |
| `tests/fixtures/sample.jsonl` | 3 test records. |
| `conftest.py` | Empty, so pytest can `import sft`. |
| `practice.ipynb` | Scratch notebook. Its kernel uses `/opt/conda/envs/vllm_env`, a different env from the terminal. |

Run all tests: `cd /root/finetune && python -m pytest tests/ -q` (25 pass).

The folder is a git repo now: commit after each step that passes.

## Pending work
None blocking. Optional: switch the notebook kernel to `/root/miniconda3/envs/vllm/bin/python` and add `%load_ext autoreload` / `%autoreload 2` in its first cell.

## Next: the SFT training loop
Build it in steps, each runnable before the next. Start with a stand-in for the stream (a JSONL file read on each GPU). The real streaming loader replaces it later without changing the loop.

| Step | File | Outcome |
|---|---|---|
| 1 | `sft/train.py` (1 GPU) | Load Qwen3-4B. Collate `PackedRow`s to tensors. Forward with `use_cache=False` and no `attention_mask`. Loss = sum of token losses ÷ supervised-token count. AdamW, warmup + cosine LR, gradient clipping. |
| 2 | `sft/train.py` | Gradient accumulation, normalizing by the total supervised tokens over all micro-batches (not a per-micro-batch mean). |
| 3 | `tests/` | Packed vs unpacked loss match; proof that `position_ids` block attention across samples (including the `use_cache` case); overfit 32 samples on a small model. |
| 4 | `sft/parallel.py` | FSDP2 on 8 GPUs with `torchrun`: `fully_shard` per decoder layer and the root, `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`, supervised-token count all-reduced across GPUs. Watch for tied embeddings. |
| 5 | `sft/model.py` | Choose the attention backend (flash-attn install, a kernels-hub flash-attention, or `flex_attention`), activation checkpointing, fused linear cross-entropy for the 151K vocabulary (Liger). |
| 6 | `sft/checkpoint.py` | DCP save/resume (model, optimizer, scheduler, data state) and export to HF `save_pretrained`. |
| 7 | `sft/data/loader.py` | Real streaming loader: per-rank sharding, dataloader workers, `state_dict()` for resume, handling streams that end unevenly across ranks. |
| 8 | `sft/metrics.py` | Loss (also per `task_type`), gradient norm, LR, tokens/sec, MFU, tokenizer rejection counts. |

## Later
- Best-fit buffered packer, to raise fill from about 85% to above 95%.
- Adapters for the other two target models, each checked with `probe_template` and the test suite.
- DPO/APO loop (APO-zero vs APO-down), then RLVR.

## Known gotchas
- **Qwen3 template:** keeps reasoning only after the last user query, and renders the final turn differently (an empty `<think>` block). Records without reasoning, or with several user queries, fall back to one sample per turn, which costs more tokens. Consider `Qwen3-4B-Instruct-2507` if much of the data has no reasoning.
- **Llama-3.2 template:** rejects parallel tool calls and drops `content` on tool-call turns.
- **`ChatTokenizer` catches every exception** so one bad record can't kill a dataloader worker. This also hides bugs in our own code, so watch the rejection counts, and use `raise_errors=True` while debugging.
