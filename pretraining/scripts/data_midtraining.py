import argparse
import os
import json
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login

# Replace with your token or use 'huggingface-cli login' in terminal
login("***REMOVED***ESWJGUGdFGbShXfXYdosaeFzzXpeeykKsX")

def format_text_only(example):
    """
    Standardizes to a simple 'text' column.
    """
    # Most pre-training datasets have 'text' or 'content'.
    # We try to find the content field and rename it to 'text'.
    text = example.get('text', example.get('content', example.get('summary', '')))
    return {"text": text}

def main():
    parser = argparse.ArgumentParser(description="Mid-training Data Preparation Script")
    parser.add_argument("--output_dir", type=str, default="./data/midtraining", help="Directory to save the processed dataset")
    
    # New arguments for Dry Run logic
    parser.add_argument("--dry_run", action="store_true", help="If set, runs a test with a small sample size.")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of samples to use per dataset if dry_run is active.")
    
    args = parser.parse_args()

    # Determine the limit based on the flag
    if args.dry_run:
        limit = args.sample_size
        mode_name = "DRY RUN"
    else:
        limit = None # None means 'all data'
        mode_name = "PRODUCTION (Full Download)"

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "midtraining_data.jsonl")

    print(f"--- Starting Mid-Training Data Preparation ---")
    print(f"--- Mode: {mode_name} ---")
    
    datasets_list = []

    # ---------------------------------------------------------
    # 1. CulturaX (Multilingual - High Quality)
    # ---------------------------------------------------------
    print("\n1. Loading CulturaX (Multilingual)...")
    try:
        # CulturaX is HUGE. We MUST use streaming for dry run to avoid downloading TBs.
        # For production, we might want full download, but even then streaming/sharding is often better.
        # We'll use streaming=True if dry_run is set, or if the user wants to stream.
        # To match the user's pattern, we'll try to respect the structure, but safety first.
        use_streaming = args.dry_run 
        
        # Loading a subset of languages for demonstration. In production, you might want 'all' or specific list.
        # We'll pick 'en' and 'zh' (Chinese) as examples since user is interested in Chinese.
        langs = ['en', 'zh']
        for lang in langs:
            print(f"   -> Fetching language: {lang}")
            ds = load_dataset("uonlp/CulturaX", lang, split="train", streaming=use_streaming)
            
            if args.dry_run:
                ds = ds.take(limit)
                # Materialize to list to work with concatenate_datasets later if mixing streaming/non-streaming
                # However, concatenate_datasets works best with same type. 
                # We will convert everything to an Arrow dataset (in memory) for dry run consistency.
                from datasets import Dataset
                ds_list_data = list(ds)
                ds = Dataset.from_list(ds_list_data)
                print(f"      -> DRY RUN: Retrieved {len(ds)} samples.")
            
            # Format
            ds = ds.map(format_text_only, remove_columns=ds.column_names)
            datasets_list.append(ds)
            
    except Exception as e:
        print(f"   -> Error loading CulturaX: {e}")

    # ---------------------------------------------------------
    # 2. CommonCrawl Chn (SkyPile-150B or similar)
    # ---------------------------------------------------------
    print("\n2. Loading CommonCrawl Chn (SkyPile-150B)...")
    try:
        # SkyPile-150B is a high-quality Chinese dataset derived from CommonCrawl.
        ds_name = "skywork/SkyPile-150B"
        # This is also huge.
        use_streaming = args.dry_run
        
        ds = load_dataset(ds_name, split="train", streaming=use_streaming)
        
        if args.dry_run:
            ds = ds.take(limit)
            from datasets import Dataset
            ds_list_data = list(ds)
            ds = Dataset.from_list(ds_list_data)
            print(f"   -> DRY RUN: Retrieved {len(ds)} samples.")
            
        ds = ds.map(format_text_only, remove_columns=ds.column_names)
        datasets_list.append(ds)
        
    except Exception as e:
        print(f"   -> Error loading CommonCrawl Chn: {e}")

    # ---------------------------------------------------------
    # 3. BaiduBaike (Chinese Encyclopedia)
    # ---------------------------------------------------------
    print("\n3. Loading BaiduBaike...")
    try:
        ds_name = "xu-song/baidu-baike-2023"
        # This is smaller but still significant.
        use_streaming = args.dry_run
        
        ds = load_dataset(ds_name, split="train", streaming=use_streaming)
        
        if args.dry_run:
            ds = ds.take(limit)
            from datasets import Dataset
            ds_list_data = list(ds)
            ds = Dataset.from_list(ds_list_data)
            print(f"   -> DRY RUN: Retrieved {len(ds)} samples.")
            
        ds = ds.map(format_text_only, remove_columns=ds.column_names)
        datasets_list.append(ds)
        
    except Exception as e:
        print(f"   -> Error loading BaiduBaike: {e}")

    # ---------------------------------------------------------
    # 4. Merge and Shuffle
    # ---------------------------------------------------------
    print("\n4. Merging Datasets...")
    if datasets_list:
        combined_dataset = concatenate_datasets(datasets_list)
        
        print(f"   -> Total samples: {len(combined_dataset)}")
        if not args.dry_run:
             # Shuffle only makes sense if we have the full dataset or a large buffer.
             # For dry run (in memory), it's fine.
             print("   -> Shuffling dataset...")
             combined_dataset = combined_dataset.shuffle(seed=42)

        # ---------------------------------------------------------
        # 5. Save to JSONL
        # ---------------------------------------------------------
        print(f"\n5. Saving to {output_file}...")
        
        # If it's a Dataset object (from dry run list conversion), we can iterate.
        with open(output_file, "w", encoding="utf-8") as f:
            for item in combined_dataset:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

        print(f"\nSUCCESS! Dataset saved at: {output_file}")
        print(f"Format: JSONL (text field)")
        print(f"Total Count: {len(combined_dataset)}")
    else:
        print("No datasets loaded successfully.")

if __name__ == "__main__":
    main()
