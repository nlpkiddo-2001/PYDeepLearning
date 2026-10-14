"""
Run under torchrun (tests/test_fsdp.py does that). Checks that FSDP2 + our loss normalization gives
exactly the gradient of ONE process seeing every row:

    grad = d/dp [ sum of token losses over all ranks and micro-batches / total supervised tokens ]

fp32 params (no bf16) so the comparison tests the math, not rounding. Exits non-zero on failure.

    CHECK_DEVICE=cpu CUDA_VISIBLE_DEVICES= torchrun --standalone --nproc_per_node=2 -m tests.fsdp_grad_check   # gloo
    torchrun --standalone --nproc_per_node=2 -m tests.fsdp_grad_check                                        # 2 GPUs
"""
import copy
import os
import sys
from collections import Counter

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from sft.data.chat_tokenizer import ChatTokenizer, TokenizerConfig
from sft.train.parallel import full_tensor, init_distributed, shard_model
from sft.train.train import accumulate_grads, compute_loss_sum, collate, row_stream
from tests.conftest import make_tiny_model

ACCUM = 2


def main():
    info = init_distributed(os.environ.get("CHECK_DEVICE", "cuda"))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    # the same rows on every rank (rank=0, world_size=1), then each rank takes its own slice
    chat_tok = ChatTokenizer(tok, TokenizerConfig(max_seq_len=512))
    stream = row_stream("tests/fixtures/sample.jsonl", chat_tok, 512, Counter())
    all_rows = [next(stream) for _ in range(ACCUM * info.world_size)]
    counts = {r.num_supervised for r in all_rows}
    assert len(counts) > 1, "rows need different supervised counts, or a wrong normalization could still pass"
    my_rows = all_rows[info.rank * ACCUM:(info.rank + 1) * ACCUM]

    model = make_tiny_model(len(tok)).to(info.device)
    reference = copy.deepcopy(model)  # unsharded copy, same init on every rank
    shard_model(model, param_dtype=torch.float32, reduce_dtype=torch.float32)

    accumulate_grads(model, my_rows, info, info.device, autocast=False)
    fsdp_grads = {n: full_tensor(p.grad) for n, p in model.named_parameters()}

    ok = True
    if info.is_main:
        n_total = sum(r.num_supervised for r in all_rows)
        total = sum(compute_loss_sum(reference, collate(r, info.device), info.device, autocast=False) for r in all_rows)
        (total / n_total).backward()

        for n, p in reference.named_parameters():
            if not torch.allclose(fsdp_grads[n], p.grad, rtol=1e-4, atol=1e-6):
                diff = (fsdp_grads[n] - p.grad).abs().max().item()
                ratio = (fsdp_grads[n].norm() / p.grad.norm()).item()
                print(f"MISMATCH {n}: max diff {diff:.3e}, norm ratio fsdp/ref {ratio:.4f}")
                ok = False
        print(f"world_size={info.world_size} rows={len(all_rows)} sup_counts={sorted(counts)} grads_match={ok}")

    flag = torch.tensor(int(ok), device=info.device)
    dist.broadcast(flag, src=0)
    dist.destroy_process_group()
    sys.exit(0 if flag.item() else 1)


if __name__ == "__main__":
    main()
