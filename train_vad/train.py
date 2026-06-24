"""VAD Training script

Usage -> python3 train.py --model crnn --epochs 5 --batch_size 32
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prepare_dataset import VADDataset, find_audio_files
from vad_models import build_model, count_params

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def split_speakers(librispeech_root: str, val_speaker_count: int = 10, seed:int = 10):
    root = Path(librispeech_root)
    speakers = sorted([p.name for p in root.iterdir() if p.is_dir()])
    rng = random.Random(seed)
    rng.shuffle(speakers)
    val = set(speakers[:val_speaker_count])
    train = set(speakers[val_speaker_count:])
    return train, val

class SpeakerFilterDataset(VADDataset):
    
    def __init__(self, allowed_speakers: set[str], **kwargs):
        super().__init__(**kwargs)

        before = len(self.speech_files)
        self.speech_files = [
            p for p in self.speech_files
            if any(part in allowed_speakers for part in p.parts)
        ]
        after = len(self.speech_files)
        print(f" Speaker files : {before} -> {after} speech files"
              f"({len(allowed_speakers)} speakers)")


def get_lr(step: int, peak_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, labels: torch.Tesnor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return p, r , f1

@torch.no_grad()
def validate(model, loader, loss_fn, device, max_batches: int = 50):
    model.eval()
    losses = []
    all_logits , all_labels = [], []
    for i , (feat, lab) in enumerate(loader):
        if i >= max_batches:
            break
        feat = feat.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)
        logits = model(feat)
        loss = loss_fn(logits, lab.float())
        losses.append(loss.item())
        all_logits.append(logits.flatten())
        all_labels.append(lab.flatten())
    model.train()
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    p, r, f1 = compute_metrics(logits_cat, labels_cat)
    return float(np.mean(losses)), p, r, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "cnn", "crnn"], default="crnn")
    ap.add_argument("--speech-dir", default="/workspace/vad_train/LibriSpeech/train-clean-100")
    ap.add_argument("--noise-dir",  default="/workspace/vad_train/musan/noise")
    ap.add_argument("--music-dir",  default="/workspace/vad_train/musan/music")
    ap.add_argument("--out-dir",    default="/workspace/vad_train/checkpoints")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--epoch-size", type=int, default=5000,
                    help="Samples per epoch. Bump to 50000+ for a serious run.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--pos-weight", type=float, default=0.5,
                    help="BCE positive weight. 0.5 mildly favours precision; "
                         "increase toward 1.0 for higher recall.")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type=="cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    

    print(f"Setting up datasets")
    train_speakers, val_speakers = split_speakers(args.speech_dir, val_speaker_count=10)
    print(f"{len(train_speakers)} train_speakers, {len(val_speakers)} validation speakers")

    common_kwargs = dict(
        speech_dir=args.speech_dir,
        noise_dir=args.noise_dir,
        music_dir=args.music_dir,
        clip_seconds=4.0,
        sample_rate=16000,
        snr_range_db=(-5, 25),
    )

    train_ds = SpeakerFilterDataset(
        allowed_speakers = train_speakers,
        epoch_size=args.epoch_size,
        seed = args.seed,
        **common_kwargs
    )

    val_ds = SpeakerFilterDataset(
        allowed_speakers = val_speakers,
        epoch_size=2000,
        seed = args.seed + 9999,
        **common_kwargs
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, persistent_workers=True, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=max(2, args.num_workers // 2), shuffle=False, persistent_workers=True, pin_memory=True
    )

    model = build_model(args.model).to(device)
    n_params = count_params(model)
    print(f"Model: {args.model.upper()} with {n_params:,} parameters")

    pos_weight_tensor = torch.tensor([args.pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * (args.epoch_size // args.batch_size)
    print(f"Total training steps: {total_steps}")
    print(f"Validation every {args.val_every} steps, "
          f"logging every {args.log_every} steps")
    print("-" * 70)

    best_f1 = 0.0
    step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        for feat, lab in train_loader:

            lr = get_lr(step, args.lr, args.warmup_steps, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr
            
            feat = feat.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)

            logits = model(feat)
            loss = loss_fn(logits, lab.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                speech_frac = lab.float().mean().item()
                print(f"step {step:6d} | epoch {epoch} | "
                      f"loss {loss.item():.4f} | lr {lr:.2e} | "
                      f"speech_frac {speech_frac:.2f} | "
                      f"elapsed {elapsed:.0f}s")
                
            if step > 0 and step % args.val_every == 0:
                val_loss, p, r, f1 = validate(model, val_loader, loss_fn, device)
                print(f"  [VAL] step {step:6d} | val_loss {val_loss:.4f} | "
                      f"P {p:.3f} | R {r:.3f} | F1 {f1:.3f}")


                if f1 > best_f1:
                    best_f1 = f1
                    ckpt_path = os.path.join(args.out_dir, f"best_{args.model}.pt")
                    torch.save({
                        "model_state": model.state_dict(),
                        "model_name": args.model,
                        "step": step,
                        "f1": f1,
                        "args": vars(args),
                    }, ckpt_path)
                    print(f"  [VAL] new best F1 {f1:.3f}, saved {ckpt_path}")
            
            step += 1
        
    val_loss, p, r, f1 = validate(model, val_loader, loss_fn, device, max_batches=100)
    print("-" * 70)
    print(f"FINAL | val_loss {val_loss:.4f} | P {p:.3f} | R {r:.3f} | F1 {f1:.3f}")
    print(f"Best F1 during training: {best_f1:.3f}")

    final_path = os.path.join(args.out_dir, f"final_{args.model}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "model_name": args.model,
        "step": step,
        "f1": f1,
        "args": vars(args),
    }, final_path)
    print(f"Saved final checkpoint: {final_path}")

        
if __name__ == "__main__":
    main()
