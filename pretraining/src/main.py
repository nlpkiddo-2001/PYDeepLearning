import os
import argparse
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.model import GPT, ModelConfig
from src.trainer import Trainer
from src.dataloader import DistributedDataLoader
from src.muon import Muon

def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser(description="H100 LLM Training")
    parser.add_argument("--stage", type=str, required=True, choices=["pretrain", "midtrain", "finetune"], help="Training stage")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps") # Default might need adjustment per stage if not passed
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Maximum learning rate")
    parser.add_argument("--lr_min", type=float, default=None, help="Minimum learning rate (default: 10% of max)")
    parser.add_argument("--context_window", type=int, default=4096, help="Context window size")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--use_muon", action="store_true", help="Use Muon optimizer for 2D parameters")
    return parser.parse_args()

def main():
    args = get_args()
    rank, world_size, local_rank = setup_ddp()
    
    # 1. Initialize Model
    config = ModelConfig(
        max_seq_len=args.context_window,
        # Add other config overrides if needed
    )
    # Inject grad_accum into config for Trainer
    setattr(config, 'grad_accum_steps', args.grad_accum)
    
    model = GPT(config)
    
    # 2. Checkpoint Loading Logic
    # If resuming from a checkpoint (either for continuation or new stage)
    if args.resume_from:
        if os.path.exists(args.resume_from):
            if rank == 0:
                print(f"Loading checkpoint from {args.resume_from}")
            
            # Map location is important for DDP
            checkpoint = torch.load(args.resume_from, map_location=f"cuda:{local_rank}")
            
            # Handle both full checkpoint dict and raw state_dict
            state_dict = checkpoint.get("model", checkpoint)
            
            # Fix for DDP keys if saved with "module." prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict)
        else:
            if rank == 0:
                print(f"Checkpoint {args.resume_from} not found! Starting from scratch.")

    # 3. Initialize DataLoader
    tokenizer_path = os.path.join(os.path.dirname(args.data_dir), "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        # Try looking one level up if data_dir is a subdir (e.g. data/pretraining)
        tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(args.data_dir)), "tokenizer.json")
    
    if not os.path.exists(tokenizer_path):
         # Fallback default
         tokenizer_path = "./data/tokenizer.json"

    loader = DistributedDataLoader(
        data_dir=args.data_dir,
        tokenizer_path=tokenizer_path,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        context_window=args.context_window
    )
    
    # 4. Optimizer & Scheduler
    # Split parameters for Muon (2D) and AdamW (others)
    if args.use_muon:
        muon_params = []
        adam_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Muon for 2D weights (Linear, Embedding, etc. if 2D)
            # Usually Muon is used for Linear layers in Transformers
            if param.ndim == 2:
                muon_params.append(param)
            else:
                adam_params.append(param)
        
        # Create optimizers
        # Note: We need a way to step both. Trainer usually takes one optimizer.
        # We can wrap them in a list or a custom wrapper class.
        # For simplicity, let's use a custom wrapper or just pass list to Trainer.
        # But standard PyTorch schedulers expect one optimizer.
        # We might need to chain them or handle them separately.
        
        # Let's use a list of optimizers and handle it in Trainer.
        # But main.py needs to create schedulers.
        
        opt_muon = Muon(muon_params, lr=0.02, momentum=0.95) # Muon LR is usually higher (0.02)
        opt_adam = AdamW(adam_params, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
        
        optimizer = [opt_muon, opt_adam]
        
        # Scheduler: Muon usually has its own schedule or same as Adam?
        # Let's assume same schedule shape but different base LRs.
        # We can use ChainedScheduler or just create two schedulers.
        
        # Warmup + Cosine for Adam
        warmup_adam = LinearLR(opt_adam, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_steps)
        decay_adam = CosineAnnealingLR(opt_adam, T_max=args.max_steps - args.warmup_steps, eta_min=args.lr_min if args.lr_min else args.learning_rate*0.1)
        sched_adam = SequentialLR(opt_adam, schedulers=[warmup_adam, decay_adam], milestones=[args.warmup_steps])
        
        # Warmup + Cosine for Muon (if needed, or constant/simple decay)
        # Muon often works well with simple constant or cosine. Let's match Adam's shape.
        warmup_muon = LinearLR(opt_muon, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_steps)
        decay_muon = CosineAnnealingLR(opt_muon, T_max=args.max_steps - args.warmup_steps, eta_min=0.002) # Min Muon LR
        sched_muon = SequentialLR(opt_muon, schedulers=[warmup_muon, decay_muon], milestones=[args.warmup_steps])
        
        scheduler = [sched_muon, sched_adam]
        
    else:
        # Standard AdamW for all
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
        
        lr_min = args.lr_min if args.lr_min is not None else args.learning_rate * 0.1
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_steps)
        decay = CosineAnnealingLR(optimizer, T_max=args.max_steps - args.warmup_steps, eta_min=lr_min)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[args.warmup_steps])
    
    # Resume Logic for Optimizers (if needed)
    start_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location="cpu")
        ckpt_config = checkpoint.get("config", None)
        
        ckpt_stage = ckpt_config.stage if hasattr(ckpt_config, 'stage') else ckpt_config.get('stage') if isinstance(ckpt_config, dict) else None
        
        if ckpt_stage == args.stage:
            if rank == 0:
                print(f"Resuming training state for stage {args.stage}")
            
            # Handle list of optimizers
            if isinstance(optimizer, list):
                # We assume checkpoint has list of states if we were using Muon
                # Or we need to be careful if switching from Adam-only to Muon.
                # For now, assume consistency: if resuming, structure matches.
                if isinstance(checkpoint["optimizer"], list):
                    for opt, state in zip(optimizer, checkpoint["optimizer"]):
                        opt.load_state_dict(state)
                    for sched, state in zip(scheduler, checkpoint["scheduler"]):
                        sched.load_state_dict(state)
                else:
                    print("Warning: Checkpoint optimizer state format mismatch (expected list). Skipping optimizer load.")
            else:
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                
            start_step = checkpoint["step"]
        else:
            if rank == 0:
                print(f"Starting new stage {args.stage} from checkpoint weights")
    
    # 5. Trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=args.output_dir,
        save_every=args.save_every,
        stage=args.stage
    )
    
    # 6. Train
    if rank == 0:
        print(f"Starting Stage: {args.stage}")
        print(f"Output Directory: {args.output_dir}")
        print(f"Max Steps: {args.max_steps}")
        print(f"Batch Size: {args.batch_size} (Global: {args.batch_size * world_size * args.grad_accum})")
        print(f"Optimizer: {'Muon + AdamW' if args.use_muon else 'AdamW'}")
        
    trainer.fit(args.max_steps, start_step=start_step)
    
    cleanup_ddp()

if __name__ == "__main__":
    main()
