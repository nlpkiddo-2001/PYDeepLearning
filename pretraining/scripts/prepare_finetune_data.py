import os
import argparse
import json
import random
from pathlib import Path

def split_dataset(input_file, output_dir, chunk_size=10000, seed=42):
    """
    Splits a large JSONL file into smaller chunks.
    
    Args:
        input_file: Path to the input .jsonl file
        output_dir: Directory to save the chunks
        chunk_size: Number of lines per chunk
        seed: Random seed for shuffling (optional)
    """
    print(f"Processing {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Read all lines first to shuffle (if dataset fits in memory)
    # For massive datasets, we might want to stream and distribute round-robin,
    # but for finetuning (usually < 10GB), reading lines is often okay.
    # If it's huge, we'll stream. Let's stream to be safe.
    
    # Actually, for finetuning, we often want a train/val split too.
    # But the prompt just says "Split the Finetuning file".
    # Let's assume we just chunk it for the distributed loader.
    
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                lines.append(line)
    
    print(f"Total lines: {len(lines)}")
    
    if seed is not None:
        random.seed(seed)
        random.shuffle(lines)
        
    # Split into chunks
    total_chunks = (len(lines) + chunk_size - 1) // chunk_size
    
    for i in range(total_chunks):
        chunk_lines = lines[i * chunk_size : (i + 1) * chunk_size]
        output_filename = f"finetune_chunk_{i:03d}.jsonl"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in chunk_lines:
                f.write(line)
        
        print(f"Saved {output_path} ({len(chunk_lines)} lines)")

    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Split finetuning dataset into chunks")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for chunks")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Lines per chunk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    split_dataset(args.input_file, args.output_dir, args.chunk_size, args.seed)

if __name__ == "__main__":
    main()
