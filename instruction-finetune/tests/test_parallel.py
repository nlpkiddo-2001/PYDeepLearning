import argparse

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from sft.train.parallel import cleanup_distributed, init_distributed, shard_model
from tests.conftest import make_tiny_model


def gb(device):
    return torch.cuda.memory_allocated(device) / 1e9 if device.startswith("cuda") else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--real", action="store_true", help="Qwen3-4B instead of the tiny model")
    args = p.parse_args()

    info = init_distributed(args.device)
    r = info.rank

    if args.real:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.float32)
        vocab = model.config.vocab_size
    else:
        vocab = 151936      # the real vocab, so the embedding/lm_head are realistic; tokenizer not needed
        model = make_tiny_model(vocab)
    model.config.use_cache = False
    model.to(info.device)
    before = gb(info.device)

    shard_model(model)
    after = gb(info.device)

    w = model.model.layers[0].mlp.down_proj.weight
    emb = model.model.embed_tokens.weight
    lines = [
        f"[rank {r}] down_proj  type={type(w).__name__}  global={tuple(w.shape)}  local={tuple(w.to_local().shape)}",
        f"[rank {r}] embed      global={tuple(emb.shape)}  local={tuple(emb.to_local().shape)}  stored dtype={emb.dtype}",
        f"[rank {r}] tied: {model.lm_head.weight is model.model.embed_tokens.weight}",
        f"[rank {r}] all params DTensor: {all(isinstance(q, DTensor) for q in model.parameters())}",
        f"[rank {r}] memory GB before={before:.2f} after={after:.2f}",
    ]

    ids = torch.randint(0, vocab, (1, 16), device=info.device)
    pos = torch.arange(16, device=info.device).unsqueeze(0)
    logits = model(input_ids=ids, position_ids=pos, use_cache=False).logits
    lines.append(f"[rank {r}] forward ok, logits dtype={logits.dtype} shape={tuple(logits.shape)}")

    # print rank by rank so lines from different processes don't interleave
    for i in range(info.world_size):
        if i == r:
            print("\n".join(lines), flush=True)
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()