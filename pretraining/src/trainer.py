import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import glob

class Trainer:
    def __init__(self, config, model, train_loader, optimizer, scheduler, save_dir, save_every=1000, stage="pretrain"):
        self.config = config
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = f"cuda:{self.gpu_id}"
        
        # Move model to GPU
        self.model = model.to(self.device)
        self.loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_every = save_every
        self.stage = stage
        self.grad_accum_steps = config.grad_accum_steps
        
        # 1. Wrap model in DDP
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
        # 2. Compile model (PyTorch 2.0 feature)
        # Note: In some environments, compilation might need specific backend or options
        # self.model = torch.compile(self.model) # Commented out for safety in unknown envs, enable if needed
        
        # Mixed Precision Context
        self.ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        # No scaler needed for bfloat16
        
        # Create save directory if rank 0
        if self.gpu_id == 0:
            os.makedirs(self.save_dir, exist_ok=True)

    def train_step(self):
        # 1. Get Batch
        try:
            input_ids, targets = next(self.loader)
        except StopIteration:
            # Handle end of epoch if loader is not infinite, though user spec implies infinite/streaming
            return None
            
        # 2. Forward Pass (in BF16)
        with self.ctx:
            logits, loss = self.model(input_ids, targets)
            # Scale loss if using Gradient Accumulation
            loss = loss / self.grad_accum_steps
            
        # 3. Backward Pass
        loss.backward()
        
        return loss.item()

    def save_checkpoint(self, step):
        if self.gpu_id != 0:
            return
            
        # Save current checkpoint
        ckpt_path = os.path.join(self.save_dir, f"step_{step}.pt")
        
        # Ensure config has stage info
        if not hasattr(self.config, 'stage'):
            setattr(self.config, 'stage', self.stage)
        
        # Handle list of optimizers/schedulers
        if isinstance(self.optimizer, list):
            opt_state = [opt.state_dict() for opt in self.optimizer]
            sched_state = [sched.state_dict() for sched in self.scheduler]
        else:
            opt_state = self.optimizer.state_dict()
            sched_state = self.scheduler.state_dict()
            
        checkpoint = {
            "model": self.model.module.state_dict(),
            "optimizer": opt_state,
            "scheduler": sched_state,
            "step": step,
            "config": self.config,
        }
        torch.save(checkpoint, ckpt_path)
        
        # Save "latest" pointer
        torch.save(checkpoint, os.path.join(self.save_dir, "latest.pt"))
        
        # Rotation: Keep only last 3 checkpoints
        # Pattern matches step_*.pt
        checkpoints = sorted(glob.glob(os.path.join(self.save_dir, "step_*.pt")), key=os.path.getmtime)
        if len(checkpoints) > 3:
            for ckpt in checkpoints[:-3]:
                os.remove(ckpt)
        
        print(f"Saved checkpoint to {ckpt_path}")

    def fit(self, max_steps, start_step=0):
        self.model.train()
        
        step = start_step
        print(f"[Rank {self.gpu_id}] Starting training from step {step} to {max_steps}")
        
        while step < max_steps:
            accum_loss = 0.0
            
            # A. Gradient Accumulation Loop
            for _ in range(self.grad_accum_steps):
                # We need to handle DDP sync properly. 
                # Gradients should only sync on the last accumulation step.
                if _ < self.grad_accum_steps - 1:
                    with self.model.no_sync():
                        loss_val = self.train_step()
                else:
                    loss_val = self.train_step()
                
                if loss_val is not None:
                    accum_loss += loss_val
            
            # B. Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # C. Optimizer Step
            if isinstance(self.optimizer, list):
                for opt in self.optimizer:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                for sched in self.scheduler:
                    sched.step()
            else:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            step += 1
            
            # D. Logging (Only Rank 0 prints)
            if self.gpu_id == 0 and step % 10 == 0: # Log more frequently for debugging/demo
                # accum_loss is sum of (loss/grad_accum), so it approximates average loss
                # Get LR (handle list)
                if isinstance(self.scheduler, list):
                    lr_val = self.scheduler[0].get_last_lr()[0] # Just show first one
                else:
                    lr_val = self.scheduler.get_last_lr()[0]
                    
                print(f"Step {step} | Loss: {accum_loss:.4f} | LR: {lr_val:.2e}")
                
            # E. Checkpointing (Only Rank 0 saves)
            if self.gpu_id == 0 and step % self.save_every == 0:
                self.save_checkpoint(step)
        
        # Save final model
        if self.gpu_id == 0:
            self.save_checkpoint(max_steps)
            torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, "final_model.pt"))
