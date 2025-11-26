import argparse
import os
import json
import time
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer

# Replace with your token or use 'huggingface-cli login' in terminal
login("***REMOVED***ESWJGUGdFGbShXfXYdosaeFzzXpeeykKsX")

def estimate_tokens(text):
    # Rough estimation: 1 token ~= 4 characters
    return len(text) // 4

def main():
    parser = argparse.ArgumentParser(description="Mid-training Data Preparation Script (English Only)")
    parser.add_argument("--output_dir", type=str, default="./data/midtraining", help="Directory to save the processed dataset")
    parser.add_argument("--token_limit", type=int, default=5_000_000_000, help="Total token limit (default: 5B)")
    parser.add_argument("--dry_run", action="store_true", help="Run with a small limit for verification")
    parser.add_argument("--chunk_size", type=int, default=50_000, help="Number of samples per output file chunk")
    args = parser.parse_args()

    if args.dry_run:
        args.token_limit = 1_000_000  # 1M tokens for dry run
        print("--- DRY RUN MODE ACTIVATED ---")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Starting Mid-training Data Preparation...")
    print(f"Target Token Limit: {args.token_limit:,}")
    print(f"Output Directory: {args.output_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except:
        print("Warning: Could not load gpt2 tokenizer. Using character heuristic (len/4).")
        tokenizer = None

    current_tokens = 0
    file_index = 0
    buffer = []
    
    def save_chunk(data, index):
        output_file = os.path.join(args.output_dir, f"midtraining_data_part_{index:04d}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved chunk {index} ({len(data)} samples) to {output_file}")

    # Datasets to iterate for Mid-training (High Quality)
    # 1. Wikipedia (English) - High quality knowledge
    # 2. CulturaX (English) - Supplement if needed
    
    dataset_configs = [
        {"name": "wikipedia", "subset": "20220301.en", "split": "train"},
        {"name": "uonlp/CulturaX", "subset": "en", "split": "train"}
    ]

    for config in dataset_configs:
        if current_tokens >= args.token_limit:
            break
            
        ds_name = config["name"]
        subset = config["subset"]
        print(f"\nProcessing {ds_name} ({subset if subset else 'default'})...")
        
        try:
            if subset:
                ds = load_dataset(ds_name, subset, split=config["split"], streaming=True)
            else:
                ds = load_dataset(ds_name, split=config["split"], streaming=True)
            
            for sample in ds:
                if current_tokens >= args.token_limit:
                    break
                
                text = sample.get("text", sample.get("content", ""))
                if not text:
                    continue
                
                # Count tokens
                if tokenizer:
                    tokens = len(tokenizer.encode(text))
                else:
                    tokens = estimate_tokens(text)
                
                current_tokens += tokens
                buffer.append({"text": text, "source": ds_name})
                
                if len(buffer) >= args.chunk_size:
                    save_chunk(buffer, file_index)
                    file_index += 1
                    buffer = []
                    print(f"Progress: {current_tokens:,} / {args.token_limit:,} tokens")

        except Exception as e:
            print(f"Error processing {ds_name}: {e}")

    # Save remaining buffer
    if buffer:
        save_chunk(buffer, file_index)
        print(f"Saved final chunk {file_index}")

    print(f"\nProcessing Complete!")
    print(f"Total Tokens Collected: {current_tokens:,}")

if __name__ == "__main__":
    main()
