import argparse
import os
from datasets import load_dataset, concatenate_datasets
import time
from huggingface_hub import login

login("***REMOVED***xxx")

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

    
    datasets_to_mix = []

    languages = ["en"] 
    
    print("\nLoading CulturaX...")
    for lang in languages:
        print(f"  - Fetching language: {lang}")
        try:
            if args.dry_run:

                ds = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
                ds = ds.take(args.sample_size)

                ds_list = list(ds) 
                print(f"    - Retrieved {len(ds_list)} samples (Dry Run)")

                datasets_to_mix.extend(ds_list)
            else:

                ds = load_dataset("uonlp/CulturaX", lang, split="train")
                datasets_to_mix.append(ds)
        except Exception as e:
            print(f"    - Error loading {lang}: {e}")


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


    output_file = os.path.join(args.output_dir, "pretraining_data.jsonl")
    if args.dry_run:
        import json
        with open(output_file, "w") as f:
            for item in datasets_to_mix:
                f.write(json.dumps(item, default=str) + "\n")
        print(f"Dry run data saved to {output_file}")
    else:
        if len(datasets_to_mix) > 0:
            final_ds = concatenate_datasets(datasets_to_mix)
            final_ds.save_to_disk(args.output_dir)
            print(f"Full dataset saved to {args.output_dir}")
        else:
            print("No data loaded.")
    return

if __name__ == "__main__":
    main()
