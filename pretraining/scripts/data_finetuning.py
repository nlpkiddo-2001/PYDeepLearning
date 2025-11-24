import argparse
import os
import json
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login

# Replace with your token or use 'huggingface-cli login' in terminal
login("***REMOVED***ESWJGUGdFGbShXfXYdosaeFzzXpeeykKsX")

def format_aya(example):
    """
    Converts Aya (inputs/targets) to Generic Messages format.
    """
    return {
        "messages": [
            {"role": "user", "content": example['inputs']},
            {"role": "assistant", "content": example['targets']}
        ]
    }

def format_ultrachat(example):
    """
    Ensures UltraChat is in the correct Generic Messages format.
    """
    return {
        "messages": example['messages']
    }

def main():
    parser = argparse.ArgumentParser(description="Production Data Preparation Script")
    parser.add_argument("--output_dir", type=str, default="./data/finetuning", help="Directory to save the processed dataset")
    
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
    output_file = os.path.join(args.output_dir, "train_dataset.jsonl")

    print(f"--- Starting Data Preparation ---")
    print(f"--- Mode: {mode_name} ---")
    
    # ---------------------------------------------------------
    # 1. Process UltraChat 200k (Multi-turn)
    # ---------------------------------------------------------
    print("\n1. Loading UltraChat 200k...")
    ds_ultra = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    
    if limit:
        # Ensure we don't ask for more samples than exist
        actual_limit = min(len(ds_ultra), limit)
        ds_ultra = ds_ultra.select(range(actual_limit))
        print(f"   -> DRY RUN: Limited UltraChat to {actual_limit} samples.")

    print("   -> Formatting UltraChat columns...")
    ds_ultra = ds_ultra.map(
        format_ultrachat, 
        remove_columns=ds_ultra.column_names
    )

    # ---------------------------------------------------------
    # 2. Process Aya Collection (Multilingual / Single-turn)
    # ---------------------------------------------------------
    print("\n2. Loading Aya Collection...")
    ds_aya = load_dataset("CohereForAI/aya_collection", "aya_dataset", split="train")

    if limit:
        actual_limit = min(len(ds_aya), limit)
        ds_aya = ds_aya.select(range(actual_limit))
        print(f"   -> DRY RUN: Limited Aya to {actual_limit} samples.")

    print("   -> Formatting Aya columns...")
    ds_aya = ds_aya.map(
        format_aya,
        remove_columns=ds_aya.column_names
    )

    # ---------------------------------------------------------
    # 3. Merge and Shuffle
    # ---------------------------------------------------------
    print("\n3. Merging Datasets...")
    combined_dataset = concatenate_datasets([ds_ultra, ds_aya])
    
    print(f"   -> Total samples before shuffling: {len(combined_dataset)}")
    print("   -> Shuffling dataset...")
    combined_dataset = combined_dataset.shuffle(seed=42)

    # ---------------------------------------------------------
    # 4. Save to JSONL (Generic Format)
    # ---------------------------------------------------------
    print(f"\n4. Saving to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in combined_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSUCCESS! Dataset saved at: {output_file}")
    print(f"Format: JSONL (Chat/Messages standard)")
    print(f"Total Count: {len(combined_dataset)}")

if __name__ == "__main__":
    main()