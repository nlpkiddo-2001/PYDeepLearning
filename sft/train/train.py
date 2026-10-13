"""
SFT TRAINING LOOP

step 1: single GPU.

    JSONL -> ChatTokenizer -> pack_greedy -> PackedRow -> collate -> model -> loss -> AdamW

- One PackedRow per micro-batch. Rows have different lengths and we can't pass an attention_mask
  (it would break the packed-sequence detection), so we don't batch several rows together.
- fp32 master weights + bf16 autocast. bf16 weights would round small AdamW updates to zero.
- Loss = sum of token losses / supervised-token count. Not a per-row mean, so every supervised
  token has the same weight. Step 2 (grad accumulation) only changes the denominator.

Run:
    python -m sft.train --data_path tests/fixtures/sample.jsonl --total_steps 20
"""

import argparse
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft.data.chat_tokenizer import IGNORE_INDEX, ChatTokenizer, TokenizerConfig
from sft.data.packing import PackedRow, pack_greedy

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    model_name : str = "Qwen/Qwen3-4B"
    data_path : str = "tests/fixtures/sample.jsonl"
    lr : float = 1e-5
    min_lr_ratio : float = 0.1
    weight_decay : float = 0.1
    grad_clip : float = 1.0
    log_every : int = 1
    device : str = "cuda"
    grad_checkpointing : bool = False
    total_steps : int = 100
    seed : int = 0
    warmup_steps : int = 10
    overfit_one_row : bool = False
    max_seq_len : int = 4096
    grad_accum_steps : int = 1


def row_stream(path: str, chat_tok: ChatTokenizer, pack_len: int, rejections: Counter) -> Iterator[PackedRow]:
    """
    Reads data → tokenizes conversations → packs samples into rows
    """
    def samples():
        while True:
            with open(path) as file:
                for line in file:
                    if not line.strip():
                        continue

                    res = chat_tok(json.loads(line))
                    for r in res.rejections:
                        rejections[r.reason.value] += 1
                    yield from res.samples
    yield from pack_greedy(samples(), pack_len)


def collate(row: PackedRow, device: str):

    def t(x):
        return torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)

    return {
        "input_ids" : t(row.input_ids),
        "position_ids" : t(row.position_ids),
        "labels" : t(row.labels),
        "num_supervised" : row.num_supervised
    }


def load_model(cfg: TrainConfig):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype = torch.float32, # float 32 contains master weights, and when doing compute we can autocast it to bf16 aka mixed precision training
        attn_implementation="sdpa"
    ).to(cfg.device)
    model.config.use_cache = False
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    return model

def build_optimiser(model, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >=2 else no_decay).append(p)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ],
        lr=cfg.lr,
        betas=(0.9,0.95),
        eps=1e-8,
        fused=cfg.device.startswith("cuda"),
    )

def build_scheduler(opt: torch.optim.Optimizer, cfg:TrainConfig):

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        progress = min(progress, 1.0)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)) # cosine formula is -> f(p) = r + (1 - r) * 1 + cos(pi*progress) / 2

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def compute_loss_sum(model, batch: dict, device: str, autocast: bool = True) -> torch.Tensor:
    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=autocast):
        logits = model(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            use_cache=False
        ).logits

    shift_logits = logits[:, :-1].float()
    shift_labels = batch["labels"][:, 1:]

    assert (shift_labels != IGNORE_INDEX).sum().item() == batch["num_supervised"] # checking that valid tokens are correct, because we are using reduciton as "sum" here.

    return F.cross_entropy(
        input=shift_logits.reshape(-1, shift_logits.size(-1)),
        target=shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum"
    )


def train(cfg: TrainConfig, model=None, tokenizer=None) -> list[float]:
    torch.manual_seed(cfg.seed)

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name)
    chat_tok = ChatTokenizer(tokenizer, TokenizerConfig(max_seq_len=cfg.max_seq_len))
    rejections: Counter = Counter()
    rows = row_stream(path=cfg.data_path, chat_tok=chat_tok, pack_len=cfg.max_seq_len, rejections=rejections)

    model = model or load_model(cfg)
    opt = build_optimiser(model, cfg)
    scd = build_scheduler(opt, cfg)

    losses = []
    fixed_rows = next(rows) if cfg.overfit_one_row else None

    for step in range(cfg.total_steps):
        t0 = time.perf_counter()

        # step 1: fetch micro batches, num_supervised must be known
        micro_rows = [fixed_rows or next(rows) for _ in range(cfg.grad_accum_steps)]
        num_supervised = sum(r.num_supervised for r in micro_rows)
        if num_supervised == 0:
            logger.warning(f"num supervised token is 0 for step {step}")
            continue

        # step 2: forward/backwar per batch , all divided by same step level count
        step_loss_sum = 0.0 # just overall loss for logging
        for row in micro_rows:
            batch = collate(row, cfg.device)
            loss_sum = compute_loss_sum(model, batch, cfg.device)
            (loss_sum / num_supervised).backward()
            step_loss_sum += loss_sum.item() # just overall loss for logging

        # step 3: one optimiser step per step, not for per batch
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not torch.isfinite(grad_norm):
            logger.warning(f"step {step}: non-finite grad norm {grad_norm}, skipping update")
            opt.zero_grad(set_to_none=True)
            continue

        # opt.step is placed outside, because we want all micro-batches to calculate gradients using the same model weights.
        opt.step()
        scd.step()
        opt.zero_grad(set_to_none=True)

        loss = step_loss_sum / num_supervised # just overall loss for logging
        losses.append(loss)
        if step % cfg.log_every == 0:
            dt = time.perf_counter() - t0
            num_tokens = sum(len(r.input_ids) for r in micro_rows)
            logger.info(
                "step %4d | loss %.4f | gnorm %.3f | lr %.2e | rows %d | tokens %d | sup %d | %.0f tok/s | rejected %s",
                step, loss, grad_norm.item(), scd.get_last_lr()[0],
                len(micro_rows), num_tokens, num_supervised, num_tokens / dt, dict(rejections),
            )

    return losses 
    # losses = []
    # fixed_rows = next(rows) if cfg.overfit_one_row else None

    # for step in range(cfg.total_steps):
    #     t0 = time.perf_counter()
    #     row = fixed_rows or next(rows)
    #     batch = collate(row, cfg.device)

    #     loss_sum = compute_loss_sum(model, batch, cfg.device)
    #     loss = loss_sum / max(1, batch["num_supervised"]) # actual sum loss, dividing by number of valid tokens, will change in distributed training
    #     loss.backward()

    #     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    #     if not torch.isfinite(grad_norm): # returns boolean whether gradients given is finite or not
    #         logger.warning(f"step {step}: non-finite grad norm {grad_norm}, skipping update")
    #         opt.zero_grad(set_to_none=True)
    #         continue

    #     opt.step()
    #     scd.step()
    #     opt.zero_grad(set_to_none=True)

    #     losses.append(loss.item())
    #     if step % cfg.log_every == 0:
    #         dt = time.perf_counter() - t0
    #         logger.info(
    #             "step %4d | loss %.4f | gnorm %.3f | lr %.2e | tokens %d | sup %d | samples %d | %.0f tok/s | rejected %s",
    #             step, losses[-1], grad_norm.item(), scd.get_last_lr()[0],
    #             len(row.input_ids), row.num_supervised, len(row.seq_lens),
    #             len(row.input_ids) / dt, dict(rejections),
    #         )

    # return losses


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        if f.type is bool or f.type == "bool":
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train(parse_args())
