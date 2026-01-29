#!/usr/bin/env python3
"""
Hyperparameter Calculator for Pretrain and Midtrain

This script:
1. Counts tokens in pretrain and midtrain datasets
2. Suggests hyperparameters (learning_rate, warmup_steps, max_steps)

Usage:
    python scripts/calculate_hyperparams.py \
        --tokenizer_path ./data/tokenizer.json \
        --pretrain_dir ./data/pretraining \
        --midtrain_config ./config.midtrain.yaml \
        --batch_size 12 \
        --grad_accum 8 \
        --world_size 8 \
        --context_window 4096
"""

import os
import sys
import glob
import json
import argparse
import yaml
from typing import Dict, List, Tuple, Optional


def count_tokens_in_file(filepath: str, tokenizer) -> int:
    """Count tokens in a single file."""
    token_count = 0
    
    try:
        if filepath.endswith('.bin'):
            # Binary file: uint16 tokens
            import numpy as np
            data = np.memmap(filepath, dtype=np.uint16, mode='r')
            token_count = len(data)
            
        elif filepath.endswith('.jsonl'):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = ""
                        if 'messages' in data:
                            # Conversational format
                            for msg in data['messages']:
                                text += msg.get('content', '') + " "
                        elif 'text' in data:
                            text = data['text']
                        elif 'content' in data:
                            text = data['content']
                        
                        if text.strip():
                            encoded = tokenizer.encode(text)
                            token_count += len(encoded.ids)
                    except json.JSONDecodeError:
                        continue
                        
        elif filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            if text.strip():
                encoded = tokenizer.encode(text)
                token_count = len(encoded.ids)
                
    except Exception as e:
        print(f"  Warning: Error processing {filepath}: {e}")
    
    return token_count


def count_tokens_in_directory(data_dir: str, tokenizer) -> Tuple[int, int]:
    """
    Count total tokens in a directory.
    Returns: (total_tokens, num_files)
    """
    if not os.path.exists(data_dir):
        print(f"  Warning: Directory does not exist: {data_dir}")
        return 0, 0
    
    files = []
    for ext in ['*.bin', '*.jsonl', '*.txt']:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    total_tokens = 0
    for filepath in files:
        tokens = count_tokens_in_file(filepath, tokenizer)
        total_tokens += tokens
        print(f"  {os.path.basename(filepath)}: {tokens:,} tokens")
    
    return total_tokens, len(files)


def count_midtrain_tokens(config_path: str, tokenizer) -> Dict[str, int]:
    """
    Count tokens for midtraining datasets from config.
    Returns dict: {dataset_name: token_count}
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets = config.get('datasets', [])
    if not datasets:
        # Fallback to single data_dir
        data_dir = config.get('data_dir', './data/midtraining')
        tokens, _ = count_tokens_in_directory(data_dir, tokenizer)
        return {'midtrain': tokens}
    
    result = {}
    for ds in datasets:
        name = ds.get('name', 'unnamed')
        path = ds['path']
        weight = ds.get('weight', 1.0)
        
        print(f"\n[{name}] (weight: {weight})")
        tokens, num_files = count_tokens_in_directory(path, tokenizer)
        result[name] = tokens
        print(f"  Total: {tokens:,} tokens in {num_files} files")
    
    return result


def suggest_hyperparameters(
    total_tokens: int,
    batch_size: int = 12,
    grad_accum: int = 8,
    world_size: int = 8,
    context_window: int = 4096,
    model_params: int = 1_100_000_000,  # ~1.1B
    is_midtrain: bool = False,
    pretrain_steps: int = 0
) -> Dict:
    """
    Suggest hyperparameters based on token count and training config.
    
    Rules of thumb:
    - Total training tokens = 20x to 100x model parameters (Chinchilla scaling)
    - Warmup = 1-5% of total steps
    - LR: ~3e-4 to 6e-4 for 1B models
    - Midtrain: 20-30% of pretrain tokens, lower LR (0.5x pretrain LR)
    """
    
    # Tokens per step
    tokens_per_step = batch_size * grad_accum * world_size * context_window
    
    # Calculate steps
    max_steps = total_tokens // tokens_per_step
    
    # Warmup: 2% of total steps, min 500, max 5000
    warmup_steps = max(500, min(5000, int(max_steps * 0.02)))
    
    # Learning rate suggestions
    if is_midtrain:
        # Midtrain uses lower LR (typically 0.3x-0.5x of pretrain)
        learning_rate = 3.0e-4
        lr_min = 3.0e-5
        # Max steps should be pretrain_steps + midtrain_steps
        suggested_max_steps = pretrain_steps + max_steps
    else:
        # Pretrain
        learning_rate = 6.0e-4
        lr_min = 6.0e-5
        suggested_max_steps = max_steps
    
    # Chinchilla optimal tokens
    chinchilla_optimal = model_params * 20  # Conservative estimate
    
    return {
        'tokens_per_step': tokens_per_step,
        'total_tokens': total_tokens,
        'max_steps': max_steps,
        'suggested_max_steps': suggested_max_steps,
        'warmup_steps': warmup_steps,
        'learning_rate': learning_rate,
        'lr_min': lr_min,
        'chinchilla_optimal_tokens': chinchilla_optimal,
        'tokens_vs_optimal_ratio': total_tokens / chinchilla_optimal if chinchilla_optimal > 0 else 0,
    }


def print_suggestions(
    stage: str,
    suggestions: Dict,
    pretrain_steps: int = 0
):
    """Pretty print hyperparameter suggestions."""
    print("\n" + "=" * 70)
    print(f"  {stage.upper()} HYPERPARAMETER SUGGESTIONS")
    print("=" * 70)
    
    print(f"\n📊 Token Statistics:")
    print(f"  Total Tokens:        {suggestions['total_tokens']:,}")
    print(f"  Tokens per Step:     {suggestions['tokens_per_step']:,}")
    print(f"  Chinchilla Optimal:  {suggestions['chinchilla_optimal_tokens']:,}")
    print(f"  Tokens/Optimal:      {suggestions['tokens_vs_optimal_ratio']:.2f}x")
    
    print(f"\n⚙️  Suggested Hyperparameters:")
    print(f"  max_steps:           {suggestions['suggested_max_steps']:,}")
    if stage == 'midtrain':
        print(f"    (pretrain steps:   {pretrain_steps:,})")
        print(f"    (midtrain steps:   {suggestions['max_steps']:,})")
    print(f"  warmup_steps:        {suggestions['warmup_steps']:,}")
    print(f"  learning_rate:       {suggestions['learning_rate']:.1e}")
    print(f"  lr_min:              {suggestions['lr_min']:.1e}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Calculate tokens and suggest hyperparameters")
    parser.add_argument("--tokenizer_path", type=str, default="./data/tokenizer.json",
                        help="Path to tokenizer.json")
    parser.add_argument("--pretrain_dir", type=str, default="./data/pretraining",
                        help="Path to pretrain data directory")
    parser.add_argument("--midtrain_config", type=str, default="./config.midtrain.yaml",
                        help="Path to midtrain config YAML")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--context_window", type=int, default=4096)
    parser.add_argument("--model_params", type=int, default=1_100_000_000,
                        help="Model parameter count (default: 1.1B)")
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"\n🔤 Loading tokenizer from {args.tokenizer_path}")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    print(f"  Vocab size: {tokenizer.get_vocab_size():,}")
    
    # =================================================================
    # PRETRAIN TOKEN COUNT
    # =================================================================
    print("\n" + "=" * 70)
    print("  PRETRAIN DATA")
    print("=" * 70)
    
    pretrain_tokens, pretrain_files = count_tokens_in_directory(args.pretrain_dir, tokenizer)
    print(f"\n📁 Total: {pretrain_tokens:,} tokens in {pretrain_files} files")
    
    pretrain_suggestions = suggest_hyperparameters(
        total_tokens=pretrain_tokens,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        world_size=args.world_size,
        context_window=args.context_window,
        model_params=args.model_params,
        is_midtrain=False
    )
    
    print_suggestions("pretrain", pretrain_suggestions)
    
    # =================================================================
    # MIDTRAIN TOKEN COUNT
    # =================================================================
    print("\n" + "=" * 70)
    print("  MIDTRAIN DATA")
    print("=" * 70)
    
    midtrain_token_counts = count_midtrain_tokens(args.midtrain_config, tokenizer)
    midtrain_total = sum(midtrain_token_counts.values())
    
    print(f"\n📊 Midtrain Dataset Summary:")
    for name, count in midtrain_token_counts.items():
        pct = (count / midtrain_total * 100) if midtrain_total > 0 else 0
        print(f"  {name}: {count:,} tokens ({pct:.1f}%)")
    print(f"  TOTAL: {midtrain_total:,} tokens")
    
    pretrain_steps = pretrain_suggestions['max_steps']
    
    midtrain_suggestions = suggest_hyperparameters(
        total_tokens=midtrain_total,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        world_size=args.world_size,
        context_window=args.context_window,
        model_params=args.model_params,
        is_midtrain=True,
        pretrain_steps=pretrain_steps
    )
    
    print_suggestions("midtrain", midtrain_suggestions, pretrain_steps)
    
    # =================================================================
    # MIDTRAIN RATIO CHECK
    # =================================================================
    print("\n" + "=" * 70)
    print("  MIDTRAIN RATIO ANALYSIS")
    print("=" * 70)
    
    midtrain_ratio = (midtrain_total / pretrain_tokens * 100) if pretrain_tokens > 0 else 0
    print(f"\n  Midtrain tokens / Pretrain tokens: {midtrain_ratio:.1f}%")
    
    if midtrain_ratio < 15:
        print("  ⚠️  WARNING: Midtrain data might be too small (<15%)")
        print("     Recommendation: Add more domain data or reduce pretrain steps")
    elif midtrain_ratio > 40:
        print("  ⚠️  WARNING: Midtrain data might be too large (>40%)")
        print("     Risk: Catastrophic forgetting of general knowledge")
    else:
        print("  ✅ Ratio looks good (15-40% range)")
    
    print("\n")


if __name__ == "__main__":
    main()
