import argparse
import os
from datasets import load_dataset, concatenate_datasets
import time
from huggingface_hub import login

login("***REMOVED***ESWJGUGdFGbShXfXYdosaeFzzXpeeykKsX")

def main():
    parser = argparse.ArgumentParser(description="Pre-training Data Preparation Script")
    parser.add_argument("--output_dir", type=str, default="./data/pretraining", help="Directory to save the processed dataset")
    parser.add_argument("--dry_run", action="store_true", help="Run in streaming mode and only process a small subset for verification")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of samples to process in dry-run mode")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Starting Pre-training Data Preparation...")
    print(f"Mode: {'DRY RUN (Streaming)' if args.dry_run else 'PRODUCTION (Full Download)'}")
    print(f"Output Directory: {args.output_dir}")

    # Dataset Configs
    # CulturaX is huge, so for dry-run we stream. For production, we'd typically want specific languages.
    # Here we load a subset or the full thing depending on the flag.
    
    datasets_to_mix = []

    # 1. CulturaX (Multilingual)
    # Note: CulturaX requires specifying languages or "all". 
    # For this script, we'll pick a few major languages as a default example of "Multilingual".
    languages = ["en", "fr", "es", "zh", "hi", "ar"] 
    
    print("\nLoading CulturaX...")
    for lang in languages:
        print(f"  - Fetching language: {lang}")
        try:
            if args.dry_run:
                # Streaming mode
                ds = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
                ds = ds.take(args.sample_size)
                # Convert to list to materialize for concatenation in dry run
                # In a real streaming pipeline we might keep them as iterables, 
                # but for verification we want to see if we can save them.
                ds_list = list(ds) 
                print(f"    - Retrieved {len(ds_list)} samples (Dry Run)")
                # We need to convert back to a Dataset object to concatenate easily with same features if we were doing full processing
                # But for dry run, let's just keep it simple.
                datasets_to_mix.extend(ds_list)
            else:
                # Full download mode
                # WARNING: This is massive. In a real H100 run, you might want to shard this.
                ds = load_dataset("uonlp/CulturaX", lang, split="train")
                datasets_to_mix.append(ds)
        except Exception as e:
            print(f"    - Error loading {lang}: {e}")

    # 2. Falcon RefinedWeb (High Quality English)
    print("\nLoading Falcon RefinedWeb...")
    try:
        if args.dry_run:
            ds = load_dataset("tiiuae/falcon-refinedweb", split="train", streaming=True)
            ds = ds.take(args.sample_size)
            ds_list = list(ds)
            print(f"    - Retrieved {len(ds_list)} samples (Dry Run)")
            datasets_to_mix.extend(ds_list)
        else:
            ds = load_dataset("tiiuae/falcon-refinedweb", split="train")
            datasets_to_mix.append(ds)
    except Exception as e:
        print(f"    - Error loading Falcon RefinedWeb: {e}")

    print(f"\nProcessing complete. Total components: {len(datasets_to_mix)}")

    # Saving
    output_file = os.path.join(args.output_dir, "pretraining_data.jsonl")
    if args.dry_run:
        import json
        with open(output_file, "w") as f:
            for item in datasets_to_mix:
                f.write(json.dumps(item, default=str) + "\n")
        print(f"Dry run data saved to {output_file}")
    else:
        # In production, we concatenate and save to disk (e.g. arrow/parquet)
        if len(datasets_to_mix) > 0:
            final_ds = concatenate_datasets(datasets_to_mix)
            final_ds.save_to_disk(args.output_dir)
            print(f"Full dataset saved to {args.output_dir}")
        else:
            print("No data loaded.")
    return

if __name__ == "__main__":
    main()
