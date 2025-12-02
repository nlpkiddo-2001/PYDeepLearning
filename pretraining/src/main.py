import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import argparse
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tokenizers import Tokenizer

from model import GPT, ModelConfig
from trainer import Trainer
from dataloader import DistributedDataLoader
from muon import Muon

torch.serialization.add_safe_globals([ModelConfig]) # torch 2.6.0+ requires this

def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

import yaml
import glob
import time

def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    if not checkpoints:
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(latest_path):
            return latest_path
        return None
    
    return max(checkpoints, key=os.path.getmtime)

def load_checkpoint_safe(checkpoint_path, map_location, rank=0):
    """
    Safely load checkpoint with proper error handling for PyTorch 2.6+.
    
    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to map tensors to
        rank: Process rank for logging
    
    Returns:
        Loaded checkpoint dictionary
    """
    try:
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=map_location,
            weights_only=False
        )
        return checkpoint
    except Exception as e:
        if rank == 0:
            print(f"Error loading checkpoint: {e}")
            print(f"Attempting alternative loading method...")
        
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=False
            )
            return checkpoint
        except Exception as e2:
            if rank == 0:
                print(f"Failed to load checkpoint: {e2}")
            raise

def print_model_stats(model, config, rank, world_size):
    """Print comprehensive model and training statistics."""
    if rank != 0:
        return
    
    total_params, trainable_params = count_parameters(model)
    param_size_mb = total_params * 4 / (1024 ** 2)
    
    print("="*70)
    print("MODEL CONFIGURATION")
    print("="*70)
    
    if 'model' in config:
        print(f"Architecture:")
        print(f"  Dimension:        {config['model'].get('dim', 'N/A')}")
        print(f"  Layers:           {config['model'].get('n_layers', 'N/A')}")
        print(f"  Attention Heads:  {config['model'].get('n_heads', 'N/A')}")
        print(f"  Vocab Size:       {config['model'].get('vocab_size', 'N/A')}")
        print(f"  Context Window:   {config.get('context_window', 'N/A')}")
    
    print(f"\nModel Size:")
    print(f"  Total Parameters:      {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable Parameters:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  Model Size (FP32):     {param_size_mb:.2f} MB")
    
    batch_size_per_rank = config['batch_size']
    grad_accum = config['grad_accum']
    global_batch_size = batch_size_per_rank * world_size * grad_accum
    seq_len = config['context_window']
    tokens_per_batch = global_batch_size * seq_len
    
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Distributed Setup:")
    print(f"  World Size:            {world_size}")
    print(f"  Gradient Accumulation: {grad_accum}")
    print(f"\nBatch Configuration:")
    print(f"  Per-Rank Batch Size:   {batch_size_per_rank}")
    print(f"  Global Batch Size:     {global_batch_size}")
    print(f"  Tokens per Batch:      {tokens_per_batch:,}")
    print(f"\nOptimization:")
    print(f"  Learning Rate:         {config['learning_rate']:.2e}")
    print(f"  Min Learning Rate:     {config.get('lr_min', config['learning_rate']*0.1):.2e}")
    print(f"  Warmup Steps:          {config['warmup_steps']}")
    print(f"  Max Steps:             {config['max_steps']}")
    print(f"  Optimizer:             {'Muon + AdamW' if config.get('use_muon', False) else 'AdamW'}")
    print("="*70)
    print("")

def get_config():
    parser = argparse.ArgumentParser(description="H100 LLM Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    config = get_config()
    rank, world_size, local_rank = setup_ddp()
    
    if config.get('resume_from') == 'auto':
        stage = config.get('stage', 'pretrain')
        if stage == 'midtrain':
            checkpoint_dir = './checkpoints/pretrain'
        elif stage == 'finetune':
            checkpoint_dir = './checkpoints/midtrain'
        else:
            checkpoint_dir = None
        
        if checkpoint_dir:
            auto_checkpoint = find_latest_checkpoint(checkpoint_dir)
            if auto_checkpoint:
                config['resume_from'] = auto_checkpoint
                if rank == 0:
                    print(f"Auto-detected checkpoint: {auto_checkpoint}")
            else:
                if rank == 0:
                    print(f"Warning: No checkpoint found in {checkpoint_dir} for auto-resume")
                config['resume_from'] = None
        else:
            config['resume_from'] = None
    
    tokenizer_path = os.path.join(os.path.dirname(config['data_dir']), "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(config['data_dir'])), "tokenizer.json")
    
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "./data/tokenizer.json"
    
    vocab_size = 56000  
    tokenizer = None
    
    if os.path.exists(tokenizer_path):
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            vocab_size = tokenizer.get_vocab_size()
            if rank == 0:
                print(f"Loaded tokenizer with vocab size: {vocab_size}")
        except Exception as e:
            if rank == 0:
                print(f"Error loading tokenizer: {e}. Using default vocab size: {vocab_size}")
    else:
        if rank == 0:
            print(f"Warning: Tokenizer not found at {tokenizer_path}. Using default vocab size: {vocab_size}")
    
    if 'model' in config:
        model_cfg = config['model']
        model_config = ModelConfig(
            vocab_size=model_cfg.get('vocab_size', vocab_size),
            dim=model_cfg.get('dim', 2048),
            n_layers=model_cfg.get('n_layers', 20),
            n_heads=model_cfg.get('n_heads', 32),
            n_kv_heads=model_cfg.get('n_kv_heads'),
            multiple_of=model_cfg.get('multiple_of', 256),
            ffn_dim_multiplier=model_cfg.get('ffn_dim_multiplier'),
            norm_eps=model_cfg.get('norm_eps', 1e-5),
            max_seq_len=config['context_window'],
            dropout=model_cfg.get('dropout', 0.0),
            gradient_checkpointing=model_cfg.get('gradient_checkpointing', False)
        )
    else:
        model_config = ModelConfig(
            vocab_size=vocab_size,
            max_seq_len=config['context_window'],
        )
    
    setattr(model_config, 'grad_accum_steps', config['grad_accum'])
    setattr(model_config, 'optimization', config.get('optimization', {}))
    
    model = GPT(model_config)
    
    if rank == 0:
        print(f"Initialized model with vocab_size={model_config.vocab_size}")
    
    if config.get('resume_from'):
        if os.path.exists(config['resume_from']):
            if rank == 0:
                print(f"Loading checkpoint from {config['resume_from']}")
            
            checkpoint = load_checkpoint_safe(
                config['resume_from'], 
                map_location=f"cuda:{local_rank}",
                rank=rank
            )
            
            state_dict = checkpoint.get("model", checkpoint)
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict)
            if rank == 0:
                print("Successfully loaded model weights from checkpoint")
        else:
            if rank == 0:
                print(f"Checkpoint {config['resume_from']} not found! Starting from scratch.")

    loader = DistributedDataLoader(
        data_dir=config['data_dir'],
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        batch_size=config['batch_size'],
        context_window=config['context_window']
    )
    
    if config.get('use_muon', False):
        muon_params = []
        adam_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 2:
                muon_params.append(param)
            else:
                adam_params.append(param)
        
        opt_muon = Muon(muon_params, lr=0.02, momentum=0.95)
        opt_adam = AdamW(adam_params, lr=config['learning_rate'], betas=(0.9, 0.95), weight_decay=0.1)
        
        optimizer = [opt_muon, opt_adam]
        
        warmup_adam = LinearLR(opt_adam, start_factor=0.01, end_factor=1.0, total_iters=config['warmup_steps'])
        decay_adam = CosineAnnealingLR(opt_adam, T_max=config['max_steps'] - config['warmup_steps'], 
                                       eta_min=config.get('lr_min', config['learning_rate']*0.1))
        sched_adam = SequentialLR(opt_adam, schedulers=[warmup_adam, decay_adam], milestones=[config['warmup_steps']])
        
        warmup_muon = LinearLR(opt_muon, start_factor=0.01, end_factor=1.0, total_iters=config['warmup_steps'])
        decay_muon = CosineAnnealingLR(opt_muon, T_max=config['max_steps'] - config['warmup_steps'], eta_min=0.002)
        sched_muon = SequentialLR(opt_muon, schedulers=[warmup_muon, decay_muon], milestones=[config['warmup_steps']])
        
        scheduler = [sched_muon, sched_adam]
        
    else:
        optimizer = AdamW(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.95), weight_decay=0.1)
        
        opt_conf = config.get('optimization', {})
        sched_type = opt_conf.get('scheduler_type', 'cosine')
        
        lr_min = config.get('lr_min')
        if lr_min is None:
            lr_min = config['learning_rate'] * 0.1
        
        if sched_type == 'wsd':
            decay_steps = int(config['max_steps'] * 0.2)
            stable_steps = config['max_steps'] - config['warmup_steps'] - decay_steps
            
            warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=config['warmup_steps'])
            stable = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=stable_steps)
            decay = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=lr_min)
            
            scheduler = SequentialLR(optimizer, schedulers=[warmup, stable, decay], 
                                   milestones=[config['warmup_steps'], config['warmup_steps'] + stable_steps])
            
        elif sched_type == 'linear':
            warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=config['warmup_steps'])
            decay = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, 
                           total_iters=config['max_steps'] - config['warmup_steps'])
            scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[config['warmup_steps']])
            
        else:
            warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=config['warmup_steps'])
            decay = CosineAnnealingLR(optimizer, T_max=config['max_steps'] - config['warmup_steps'], eta_min=lr_min)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[config['warmup_steps']])
    
    start_step = 0
    if config.get('resume_from') and os.path.exists(config['resume_from']):
        checkpoint = load_checkpoint_safe(config['resume_from'], map_location="cpu", rank=rank)
        ckpt_config = checkpoint.get("config", None)
        
        ckpt_stage = None
        if ckpt_config:
            if hasattr(ckpt_config, 'stage'):
                ckpt_stage = ckpt_config.stage
            elif isinstance(ckpt_config, dict):
                ckpt_stage = ckpt_config.get('stage')
        
        if ckpt_stage == config.get('stage'):
            if rank == 0:
                print(f"Resuming training state for stage {config['stage']}")
            
            if isinstance(optimizer, list):
                if isinstance(checkpoint.get("optimizer"), list):
                    for opt, state in zip(optimizer, checkpoint["optimizer"]):
                        opt.load_state_dict(state)
                    for sched, state in zip(scheduler, checkpoint["scheduler"]):
                        sched.load_state_dict(state)
                else:
                    if rank == 0:
                        print("Warning: Checkpoint optimizer state format mismatch. Skipping optimizer load.")
            else:
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "scheduler" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                
            start_step = checkpoint.get("step", 0)
            if rank == 0:
                print(f"Resuming from step {start_step}")
        else:
            if rank == 0:
                print(f"Starting new stage {config.get('stage')} from checkpoint weights")
    
    trainer = Trainer(
        config=model_config,
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=config['output_dir'],
        save_every=config['save_every'],
        stage=config.get('stage', 'pretrain')
    )
    
    trainer.set_batch_config(config['batch_size'], config['grad_accum'])
    
    print_model_stats(model, config, rank, world_size)
    
    trainer.fit(config['max_steps'], start_step=start_step)
    
    cleanup_ddp()

if __name__ == "__main__":
    main()