import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import glob
import time

from model import ModelConfig
torch.serialization.add_safe_globals([ModelConfig])

class BatchSizeScheduler:
    def __init__(self, config, trainer):
        self.config = config or {}
        self.trainer = trainer
        self.strategy = self.config.get('strategy', 'fixed')
        self.start_bs = self.config.get('start_batch_size', 1)
        self.max_bs = self.config.get('max_batch_size', 100)
        self.interval = self.config.get('interval_steps', 1000)
        self.milestones = self.config.get('milestones', []) # For 'step' strategy
        
    def step(self, current_step):
        if self.strategy == 'fixed':
            return

        current_effective_bs = self.trainer.batch_size * self.trainer.current_grad_accum
        target_grad_accum = self.trainer.current_grad_accum
        
        if self.strategy == 'doubling':
            if current_step > 0 and current_step % self.interval == 0:
                if current_effective_bs < self.max_bs:
                    target_grad_accum = self.trainer.current_grad_accum * 2
                    
        elif self.strategy == 'linear':
            total_steps = self.trainer.config.max_steps
            progress = min(1.0, current_step / total_steps)
            target_bs = self.start_bs + (self.max_bs - self.start_bs) * progress
            
            base_bs = self.trainer.batch_size * self.trainer.world_size
            target_grad_accum = max(1, int(round(target_bs / base_bs)))
            
        elif self.strategy == 'step':
            for step, multiplier in self.milestones:
                if current_step == step:
                    target_grad_accum = int(self.trainer.current_grad_accum * multiplier)
        
        if target_grad_accum != self.trainer.current_grad_accum:
            if self.max_bs:
                base_bs = self.trainer.batch_size * self.trainer.world_size
                if target_grad_accum * base_bs > self.max_bs:
                    target_grad_accum = max(1, self.max_bs // base_bs)
            
            if target_grad_accum != self.trainer.current_grad_accum:
                self._update_grad_accum(target_grad_accum, current_step)

    def _update_grad_accum(self, new_grad_accum, step):
        scale = (new_grad_accum / self.trainer.current_grad_accum) ** 0.5
        self.trainer._scale_lr(scale)
        
        old_accum = self.trainer.current_grad_accum
        self.trainer.current_grad_accum = new_grad_accum
        self.trainer.grad_accum_steps = new_grad_accum
        
        if self.trainer.gpu_id == 0:
            base_bs = self.trainer.batch_size * self.trainer.world_size
            print(f"[Step {step}] Batch Size Update: {old_accum} -> {new_grad_accum} Grad Accum Steps")
            print(f"          Effective Batch Size: {old_accum * base_bs} -> {new_grad_accum * base_bs}")
            print(f"          Scaling LR by {scale:.4f}")

class Trainer:
    def __init__(self, config, model, train_loader, optimizer, scheduler, save_dir, save_every=1000, stage="pretrain"):
        self.config = config
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = f"cuda:{self.gpu_id}"
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        

        self.model = model.to(self.device)
        self.loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_every = save_every
        self.stage = stage
        self.grad_accum_steps = config.grad_accum_steps
        
        self.seq_len = config.max_seq_len if hasattr(config, 'max_seq_len') else 4096
        self.batch_size = None
        self.step_times = []
        self.ema_tokens_per_sec = None
        self.ema_alpha = 0.1
        
        self.vocab_size = config.vocab_size if hasattr(config, 'vocab_size') else None
        
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
        self.ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        
        if self.gpu_id == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            
        self.opt_config = getattr(config, 'optimization', {})
        self.batch_scheduler = BatchSizeScheduler(self.opt_config.get('batch_size_schedule', {}), self)
        self.ctx_schedule = self.opt_config.get('context_length_schedule', None)
        
        self.current_grad_accum = self.grad_accum_steps

    def set_batch_config(self, batch_size_per_rank, grad_accum):
        self.batch_size = batch_size_per_rank
        self.grad_accum_steps = grad_accum
        self.current_grad_accum = grad_accum

    def validate_tokens(self, input_ids, targets):
        if self.vocab_size is None:
            return  # Skip validation if vocab_size not set
        
        invalid_mask = (input_ids < 0) | (input_ids >= self.vocab_size)
        if invalid_mask.any():
            min_val = input_ids.min().item()
            max_val = input_ids.max().item()
            num_invalid = invalid_mask.sum().item()
            raise ValueError(
                f"[Rank {self.gpu_id}] Invalid token IDs in input_ids!\n"
                f"  Vocab size: {self.vocab_size}\n"
                f"  Min token ID: {min_val}\n"
                f"  Max token ID: {max_val}\n"
                f"  Number of invalid tokens: {num_invalid}\n"
                f"  Invalid indices: {torch.where(invalid_mask)}"
            )
        
        valid_targets = targets != -100
        invalid_mask = valid_targets & ((targets < 0) | (targets >= self.vocab_size))
        if invalid_mask.any():
            min_val = targets[valid_targets].min().item()
            max_val = targets[valid_targets].max().item()
            num_invalid = invalid_mask.sum().item()
            raise ValueError(
                f"[Rank {self.gpu_id}] Invalid token IDs in targets!\n"
                f"  Vocab size: {self.vocab_size}\n"
                f"  Min token ID: {min_val}\n"
                f"  Max token ID: {max_val}\n"
                f"  Number of invalid tokens: {num_invalid}"
            )

    def train_step(self):
        try:
            input_ids, targets = next(self.loader)
        except StopIteration:
            return None
        
        self.validate_tokens(input_ids, targets)
            
        with self.ctx:
            logits, loss = self.model(input_ids, targets)
            loss = loss / self.grad_accum_steps
            
        loss.backward()
        
        loss_val = loss.item()
        
        if loss_val == 0.0:
            print(f"[Rank {self.gpu_id}] WARNING: Loss is exactly 0.0!")
            if (targets == -100).all():
                 print(f"[Rank {self.gpu_id}] All targets are -100 (ignore_index). This explains 0 loss.")
        
        return loss_val

    def save_checkpoint(self, step):
        if self.gpu_id != 0:
            return
            
        ckpt_path = os.path.join(self.save_dir, f"step_{step}.pt")
        
        if not hasattr(self.config, 'stage'):
            setattr(self.config, 'stage', self.stage)
        
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
        
        torch.save(checkpoint, os.path.join(self.save_dir, "latest.pt"))
        
        checkpoints = sorted(glob.glob(os.path.join(self.save_dir, "step_*.pt")), key=os.path.getmtime)
        if len(checkpoints) > 3:
            for ckpt in checkpoints[:-3]:
                os.remove(ckpt)
        
        print(f"Saved checkpoint to {ckpt_path}")

    def fit(self, max_steps, start_step=0):
        self.model.train()
        
        step = start_step
        if self.gpu_id == 0:
            print(f"[Rank {self.gpu_id}] Starting training from step {step} to {max_steps}")
            if self.vocab_size:
                print(f"[Rank {self.gpu_id}] Vocabulary size: {self.vocab_size}")
        
        while step < max_steps:
            self._check_schedules(step)
            
            accum_loss = 0.0
            step_start_time = time.time()
            
            for micro_step in range(self.grad_accum_steps):
                if micro_step < self.grad_accum_steps - 1:
                    with self.model.no_sync():
                        loss_val = self.train_step()
                else:
                    loss_val = self.train_step()
                
                if loss_val is not None:
                    accum_loss += loss_val
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
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
            step_time = time.time() - step_start_time
            
            tokens_this_step = self.batch_size * self.seq_len * self.world_size * self.grad_accum_steps
            tokens_per_sec = tokens_this_step / step_time if step_time > 0 else 0
            
            if self.ema_tokens_per_sec is None:
                self.ema_tokens_per_sec = tokens_per_sec
            else:
                self.ema_tokens_per_sec = self.ema_alpha * tokens_per_sec + (1 - self.ema_alpha) * self.ema_tokens_per_sec
            
            if self.gpu_id == 0 and step % 10 == 0:
                if isinstance(self.scheduler, list):
                    lr_val = self.scheduler[0].get_last_lr()[0]
                else:
                    lr_val = self.scheduler.get_last_lr()[0]
                
                samples_per_sec = tokens_per_sec / self.seq_len
                    
                print(f"Step {step:6d} | Loss: {accum_loss:.4f} | LR: {lr_val:.2e} | "
                      f"Tokens/sec: {self.ema_tokens_per_sec:,.0f} | "
                      f"Samples/sec: {samples_per_sec:.1f} | "
                      f"Time: {step_time:.2f}s")
                
            if self.gpu_id == 0 and step % self.save_every == 0:
                self.save_checkpoint(step)
        
        if self.gpu_id == 0:
            self.save_checkpoint(max_steps)
            torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, "final_model.pt"))

    def _check_schedules(self, step):
        """Check and apply dynamic updates to batch size and context length."""
        if self.batch_scheduler:
            self.batch_scheduler.step(step)
        if self.ctx_schedule:
            for trigger_step, new_len in self.ctx_schedule:
                if step == trigger_step:
                    if hasattr(self.loader, 'set_context_window'):
                        self.loader.set_context_window(new_len)
                    self.seq_len = new_len
                    if self.gpu_id == 0:
                        print(f"[Step {step}] Updating Context Length to {new_len}")

    def _scale_lr(self, scale):
        """Scale learning rate of optimizer."""
        if isinstance(self.optimizer, list):
            for opt in self.optimizer:
                for param_group in opt.param_groups:
                    param_group['lr'] *= scale
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= scale