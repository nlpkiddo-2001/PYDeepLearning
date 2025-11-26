import os
import json
import argparse
import glob
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer

def pack_data(input_path, output_dir, tokenizer_path, chunk_size=1024*1024*100): # 100M tokens per file
    """
    Reads JSONL files, tokenizes them, and packs them into binary .bin files (uint16).
    Supports directory (recursive) or single file input.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("</s>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<|endoftext|>")
    
    buffer = []
    file_idx = 0
    total_tokens = 0
    
    # Resolve input files
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        # Recursive search for .jsonl
        files = sorted(glob.glob(os.path.join(input_path, "**/*.jsonl"), recursive=True))
    else:
        # Try as glob pattern
        files = sorted(glob.glob(input_path))
        
    if not files:
        print(f"No files found at {input_path}")
        return
    
    print(f"Found {len(files)} files to process.")
    
    for filepath in files:
        print(f"Processing {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        line = line.strip()
                        if not line:
                            continue
                            
                        data = json.loads(line)
                        text = data.get('text', data.get('content', ''))
                        
                        if not text:
                            continue
                            
                        # Tokenize
                        ids = tokenizer.encode(text).ids
                        ids.append(eos_id)
                        
                        buffer.extend(ids)
                        
                        # If buffer is large enough, write to file
                        while len(buffer) >= chunk_size:
                            chunk = buffer[:chunk_size]
                            buffer = buffer[chunk_size:]
                            
                            arr = np.array(chunk, dtype=np.uint16)
                            out_path = os.path.join(output_dir, f"data_{file_idx:04d}.bin")
                            with open(out_path, 'wb') as out_f:
                                out_f.write(arr.tobytes())
                            
                            print(f"Wrote {out_path} ({len(chunk)} tokens)")
                            file_idx += 1
                            total_tokens += len(chunk)
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
                    
    # Write remaining buffer
    if buffer:
        arr = np.array(buffer, dtype=np.uint16)
        out_path = os.path.join(output_dir, f"data_{file_idx:04d}.bin")
        with open(out_path, 'wb') as out_f:
            out_f.write(arr.tobytes())
        print(f"Wrote {out_path} ({len(buffer)} tokens)")
        total_tokens += len(buffer)
        
    print(f"Done. Total tokens: {total_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input directory, file, or glob pattern")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .bin files")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer.json")
    args = parser.parse_args()
    
    pack_data(args.input, args.output_dir, args.tokenizer_path)
