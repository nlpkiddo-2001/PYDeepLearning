"""
Distributed Streaming Data Loader for LLM Pretraining (Binary Format)

This module implements a file-level sharding strategy for distributed training using binary packed data.
Each GPU (rank) reads different .bin files to maximize disk I/O throughput.
"""

import os
import glob
import random
import numpy as np
import torch
from typing import List, Optional, Iterator

class DistributedDataLoader:
    """
    Distributed Data Loader for Binary Packed Data (.bin)
    
    Args:
        data_dir: Directory containing .bin files (uint16 numpy arrays)
        rank: GPU rank (0 to world_size-1)
        world_size: Total number of GPUs
        batch_size: Number of sequences per batch
        context_window: Maximum sequence length (tokens)
        device: Target device (e.g., 'cuda:0')
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str, # Kept for API compatibility, but not used for loading
        rank: int = 0,
        world_size: int = 1,
        batch_size: int = 8,
        context_window: int = 4096,
        device: Optional[str] = None
    ):
        self.data_dir = data_dir
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.context_window = context_window
        
        # Set device
        if device is None:
            self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Shard files across ranks
        self.files = self._get_sharded_files()
        print(f"[Rank {self.rank}] Assigned {len(self.files)} binary files")
        
        if not self.files:
            print(f"[Rank {self.rank}] WARNING: No files assigned!")
            
        self.current_file_idx = 0
        self.current_data = None
        self.current_pos = 0
        
        # Statistics
        self.batches_produced = 0
        
        # Load first file
        self._load_next_file()
    
    def _get_sharded_files(self) -> List[str]:
        """Get files assigned to this rank using round-robin sharding"""
        all_files = sorted(glob.glob(os.path.join(self.data_dir, "*.bin")))
        
        if not all_files:
            # Fallback to checking if there are jsonl files, if so warn user
            if glob.glob(os.path.join(self.data_dir, "*.jsonl")):
                raise ValueError(f"No .bin files found in {self.data_dir}, but .jsonl files exist. Please run src/data_packing.py first.")
            raise ValueError(f"No .bin files found in {self.data_dir}")
        
        # Round-robin assignment
        sharded_files = [f for i, f in enumerate(all_files) if i % self.world_size == self.rank]
        
        # Shuffle files initially
        random.shuffle(sharded_files)
        
        return sharded_files
    
    def _load_next_file(self):
        """Load the next binary file into memory (mmap)"""
        if not self.files:
            return

        filename = self.files[self.current_file_idx]
        
        # Use memmap for efficiency with large files
        # We assume uint16 tokens
        self.current_data = np.memmap(filename, dtype=np.uint16, mode='r')
        self.current_pos = 0
        
        # Advance file index (cycle)
        self.current_file_idx = (self.current_file_idx + 1) % len(self.files)
        
        # If we cycled back to start, shuffle again
        if self.current_file_idx == 0:
            random.shuffle(self.files)
            
    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data.
        Returns:
            input_ids: (batch_size, context_window)
            target_ids: (batch_size, context_window)
        """
        input_list = []
        target_list = []
        
        req_len = self.context_window + 1
        
        while len(input_list) < self.batch_size:
            if self.current_data is None:
                # Should not happen unless no files
                return torch.zeros((self.batch_size, self.context_window), dtype=torch.long), torch.zeros((self.batch_size, self.context_window), dtype=torch.long)

            # Check if we have enough data left in current file
            if self.current_pos + req_len > len(self.current_data):
                self._load_next_file()
                continue
                
            # Get chunk
            # Note: converting to int64 (long) for PyTorch
            chunk = self.current_data[self.current_pos : self.current_pos + req_len].astype(np.int64)
            self.current_pos += req_len
            
            # Convert to tensor
            chunk_tensor = torch.from_numpy(chunk)
            
            input_list.append(chunk_tensor[:-1])
            target_list.append(chunk_tensor[1:])
            
        # Stack
        input_ids = torch.stack(input_list)
        target_ids = torch.stack(target_list)
        
        return input_ids, target_ids
    
    def __iter__(self):
        return self
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, target_ids = self._get_batch()
        
        # Move to device
        input_ids = input_ids.to(self.device, non_blocking=True)
        target_ids = target_ids.to(self.device, non_blocking=True)
        
        self.batches_produced += 1
        
        return input_ids, target_ids