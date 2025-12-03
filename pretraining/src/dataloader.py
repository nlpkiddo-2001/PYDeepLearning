import os
import glob
import random
import json
import numpy as np
import torch
from typing import List, Optional, Iterator

class DistributedDataLoader:
    """
    Distributed Data Loader for LLM Pretraining and Fine-tuning
    Supports:
    - Binary packed data (.bin)
    - Raw text data (.jsonl with 'text' or 'content' fields)
    - Conversational data (.jsonl with 'messages' field)
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        tokenizer=None,
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
        
        if device is None:
            self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            from tokenizers import Tokenizer
            if os.path.exists(tokenizer_path):
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
            else:
                print(f"[Rank {self.rank}] Warning: Tokenizer not found at {tokenizer_path}. Raw data loading will fail.")

        self.files = self._get_sharded_files()
        print(f"[Rank {self.rank}] Assigned {len(self.files)} files")
        
        if not self.files:
            print(f"[Rank {self.rank}] WARNING: No files assigned!")
            
        self.current_file_idx = 0
        self.current_data = None 
        self.current_pos = 0
        self.is_raw = False
        
        self.token_buffer = [] 
        
        self.batches_produced = 0
        
        self._load_next_file()
    
    def _get_sharded_files(self) -> List[str]:
        """Get files assigned to this rank using round-robin sharding"""
        bin_files = sorted(glob.glob(os.path.join(self.data_dir, "*.bin")))
        jsonl_files = sorted(glob.glob(os.path.join(self.data_dir, "*.jsonl")))
        txt_files = sorted(glob.glob(os.path.join(self.data_dir, "*.txt")))
        
        all_files = bin_files + jsonl_files + txt_files
        
        if not all_files:
            raise ValueError(f"No .bin, .jsonl, or .txt files found in {self.data_dir}")
        
        sharded_files = [f for i, f in enumerate(all_files) if i % self.world_size == self.rank]
        
        random.shuffle(sharded_files)
        
        return sharded_files
    
    def _load_next_file(self):
        """Load the next file into memory"""
        if not self.files:
            return

        filename = self.files[self.current_file_idx]
        
        if filename.endswith('.jsonl') or filename.endswith('.txt'):
            self._load_raw_file(filename)
        elif filename.endswith('.bin'):
            try:
                file_size = os.path.getsize(filename)
                if file_size % 2 != 0:
                    print(f"[Rank {self.rank}] WARNING: Skipping corrupted binary file {filename} (Size {file_size} not even).")
                    self.current_file_idx = (self.current_file_idx + 1) % len(self.files)
                    return self._load_next_file()

                self.current_data = np.memmap(filename, dtype=np.uint16, mode='r')
                self.current_pos = 0
            except Exception as e:
                print(f"[Rank {self.rank}] Error loading binary file {filename}: {e}")
                self.current_file_idx = (self.current_file_idx + 1) % len(self.files)
                return self._load_next_file()
        else:
            print(f"[Rank {self.rank}] Unknown file extension: {filename}")
        
        self.current_file_idx = (self.current_file_idx + 1) % len(self.files)
        
        if self.current_file_idx == 0:
            random.shuffle(self.files)

    def _format_messages_to_text(self, messages: list) -> str:
        """
        Convert messages format to a single text string for tokenization.
        Format: <|user|>user_msg<|assistant|>assistant_msg<|endoftext|>
        """
        formatted_parts = []
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'user':
                formatted_parts.append(f"<|user|>\n{content}")
            elif role == 'assistant':
                formatted_parts.append(f"<|assistant|>\n{content}")
            elif role == 'system':
                formatted_parts.append(f"<|system|>\n{content}")
        
        # Join all parts and add end token
        return "\n".join(formatted_parts) + "\n<|endoftext|>"

    def _load_raw_file(self, filename: str):
        """Read raw text file, tokenize, and fill buffer"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for loading raw files but was not initialized.")
        
        new_tokens = []
        eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        
        # Handle case where special tokens might not exist
        if eos_id is None:
            print(f"[Rank {self.rank}] Warning: <|endoftext|> token not found, using 0")
            eos_id = 0
        
        lines_processed = 0
        lines_skipped = 0
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    text = ""
                    
                    if filename.endswith('.jsonl'):
                        try:
                            data = json.loads(line)
                            
                            # CRITICAL FIX: Handle 'messages' format
                            if 'messages' in data:
                                text = self._format_messages_to_text(data['messages'])
                            # Fallback to text/content format
                            elif 'text' in data or 'content' in data:
                                text = data.get('text', '') or data.get('content', '')
                            else:
                                lines_skipped += 1
                                continue
                                
                        except json.JSONDecodeError as e:
                            if self.rank == 0 and line_num < 5:
                                print(f"[Rank {self.rank}] JSON decode error on line {line_num}: {e}")
                            lines_skipped += 1
                            continue
                    else:
                        # Plain text files
                        text = line
                    
                    if not text.strip():
                        lines_skipped += 1
                        continue
                    
                    # Encode
                    try:
                        encoded = self.tokenizer.encode(text)
                        new_tokens.extend(encoded.ids)
                        new_tokens.append(eos_id)
                        lines_processed += 1
                    except Exception as e:
                        if self.rank == 0 and line_num < 5:
                            print(f"[Rank {self.rank}] Encoding error on line {line_num}: {e}")
                        lines_skipped += 1
                        continue
                        
        except Exception as e:
            print(f"[Rank {self.rank}] Error reading {filename}: {e}")
        
        if self.rank == 0:
            print(f"[Rank {self.rank}] Loaded {filename}:")
            print(f"  - Lines processed: {lines_processed}")
            print(f"  - Lines skipped: {lines_skipped}")
            print(f"  - Total tokens: {len(new_tokens):,}")
            
            if len(new_tokens) == 0:
                print(f"  ⚠️  WARNING: No tokens extracted from file!")
            
        self.current_data = np.array(new_tokens, dtype=np.uint16)
        self.current_pos = 0

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
            if self.current_data is None or len(self.current_data) == 0:
                self._load_next_file()
                if self.current_data is None or len(self.current_data) == 0:
                    print(f"[Rank {self.rank}] ⚠️  WARNING: Returning zero tensors - no data available!")
                    return torch.zeros((self.batch_size, self.context_window), dtype=torch.long), torch.zeros((self.batch_size, self.context_window), dtype=torch.long)

            if self.current_pos + req_len > len(self.current_data):
                self._load_next_file()
                continue
                
            chunk = self.current_data[self.current_pos : self.current_pos + req_len].astype(np.int64)
            self.current_pos += req_len
            
            chunk_tensor = torch.from_numpy(chunk)
            
            input_list.append(chunk_tensor[:-1])
            target_list.append(chunk_tensor[1:])
            
        input_ids = torch.stack(input_list)
        target_ids = torch.stack(target_list)
        
        return input_ids, target_ids
    
    def __iter__(self):
        return self
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, target_ids = self._get_batch()
        
        input_ids = input_ids.to(self.device, non_blocking=True)
        target_ids = target_ids.to(self.device, non_blocking=True)
        
        self.batches_produced += 1
        
        return input_ids, target_ids

    def set_batch_size(self, new_batch_size: int):
        """Update batch size dynamically."""
        if new_batch_size != self.batch_size:
            print(f"[Rank {self.rank}] Updating batch size: {self.batch_size} -> {new_batch_size}")
            self.batch_size = new_batch_size

    def set_context_window(self, new_context_window: int):
        """Update context window dynamically."""
        if new_context_window != self.context_window:
            print(f"[Rank {self.rank}] Updating context window: {self.context_window} -> {new_context_window}")
            self.context_window = new_context_window
