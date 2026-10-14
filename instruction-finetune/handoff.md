# SFT pipeline handoff

_Last updated: 2026-10-13_

## Working agreement (read first)
- **Ask before running anything on the GPUs** (torchrun, training, GPU tests), even when they're free.
- **The user is learning to write this pipeline.** Teach: explain why and how, give hints and specs, review what they write. Write or edit repo code only when they explicitly ask. CPU-only checks of example snippets go in a scratch directory, not the repo.

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
| `sft/train/train.py` | Steps 1–2 done, step 4 started: `TrainConfig`, `row_stream` (round-robin sharding by rank), `collate`, `load_model`, AdamW + warmup/cosine, `compute_loss_sum(..., autocast=)`, gradient accumulation with one denominator per step. Run with `python -m sft.train.train`. |
| `tests/conftest.py` | `tok` fixture (Qwen3-4B tokenizer) and `make_tiny_model` (random 2-layer Qwen3, real vocab). |
| `tests/test_chat_tokenizer.py`, `tests/test_packing.py` | 17 + 8 tests. |
| `tests/test_grad_accum.py` | Accumulated grads == one backward over the whole step; the mean-of-means way differs. |
| `tests/test_packing_attention.py` | Packed == unpacked loss and grads (fp32); no cross-sample attention; negative control: `use_cache=True` or an `attention_mask` leaks. sdpa, eager, flex. |
| `tests/test_overfit.py` | `train()` memorizes 32 samples. Slow (~2 min CPU), runs only with `RUN_SLOW=1`. |
| `tests/test_sharding.py` | Ranks get disjoint records that cover the file; a rank with no usable samples raises. |
| `tests/fixtures/sample.jsonl` | 3 records; `all-frozen` has no train turns (always rejected). |
| `practice/` | The user's exercises (e.g. `hello_dist.py`). Untracked on purpose. |
| `practice.ipynb` | Scratch notebook. Its kernel uses `/opt/conda/envs/vllm_env`, a different env from the terminal. |

Run all tests on CPU: `cd /root/finetune && CUDA_VISIBLE_DEVICES= python -m pytest tests/ -q` (41 pass, 1 skipped). With `RUN_SLOW=1`: 42 pass.

Commit after each step that passes. No git remote yet: the history exists only on this machine.

## Pending work: step 4 (FSDP2), in progress
The user is writing this section by section (lesson order):
1. ~~torchrun + process group~~ (exercise `practice/hello_dist.py`). Done.
2. ~~Data sharding~~ in `row_stream`. Done, committed.
3. **Next: model sharding (user writes it).** `sft/parallel.py`: `fully_shard` per decoder layer then the root, `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`, tied `embed_tokens`/`lm_head` left to the root unit (assert the tie after sharding). Under FSDP call `compute_loss_sum(..., autocast=False)`.
4. Global supervised count (all-reduce before any backward) and loss scale `world_size / num_supervised`, because FSDP averages grads over ranks.
5. `set_requires_gradient_sync` only on the last micro-batch; a flag to sync every micro-batch.
6. `clip_grad_norm_` returns a DTensor: `.full_tensor()` it. The norm is global, so every rank skips together.
7. Rank-0 logging, all-reduced loss; HF export via `get_model_state_dict(full_state_dict=True, cpu_offload=True)` on every rank, rank 0 writes.
8. Tests: 2-GPU fp32 grad check vs an unsharded reference (with a negative control that drops `× world_size`), then an 8-GPU smoke run on a JSONL with at least 8 lines.

Reference numbers from an earlier implementation that was reverted so the user can write it (re-verify yours against them):
- Without the `× world_size` scale, every grad is exactly 0.5× the reference on 2 GPUs.
- Qwen3-4B, 8 GPUs, `max_seq_len=4096`, accum 2: about 63K tok/s, rows about 97% full, the tie survives sharding, the bf16 HF export is 8.3 GB and reloads.

Optional: switch the notebook kernel to `/root/miniconda3/envs/vllm/bin/python` and add `%load_ext autoreload` / `%autoreload 2` in its first cell.

## Next: the SFT training loop
Build it in steps, each runnable before the next. Start with a stand-in for the stream (a JSONL file read on each GPU). The real streaming loader replaces it later without changing the loop.

| Step | File | Outcome |
|---|---|---|
| 1 ✅ | `sft/train/train.py` (1 GPU) | Load Qwen3-4B. Collate `PackedRow`s to tensors. Forward with `use_cache=False` and no `attention_mask`. Loss = sum of token losses ÷ supervised-token count. AdamW, warmup + cosine LR, gradient clipping. |
| 2 ✅ | `sft/train/train.py` | Gradient accumulation, normalizing by the total supervised tokens over all micro-batches (not a per-micro-batch mean). |
| 3 ✅ | `tests/` | Packed vs unpacked loss match; proof that `position_ids` block attention across samples (including the `use_cache` case); overfit 32 samples on a small model. |
| 4 🚧 | `sft/parallel.py` | FSDP2 on 8 GPUs with `torchrun`: `fully_shard` per decoder layer and the root, `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`, supervised-token count all-reduced across GPUs. Watch for tied embeddings. |
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
- **bf16 hides small bugs:** packed vs unpacked grads differ by about 2e-3 relative under bf16 autocast and by about 1e-7 in fp32. Correctness tests run in fp32 (`autocast=False`).
- **FlexAttention has no backward on CPU**, so CPU tests check only its forward pass.
- **Collectives:** every rank must call the same collectives in the same order, or the job hangs without an error. Every skip decision (zero supervised tokens, non-finite grad norm) must use a global value.
- **`torch.cuda.set_device(local_rank)` is required** next to `init_process_group(device_id=...)`. Without it, a plain `"cuda"` means GPU 0 on every rank.
- **`ChatTokenizer` catches every exception** so one bad record can't kill a dataloader worker. This also hides bugs in our own code, so watch the rejection counts, and use `raise_errors=True` while debugging.
