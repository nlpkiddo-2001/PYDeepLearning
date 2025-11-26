import argparse
import os
import json
from datasets import load_dataset
from huggingface_hub import login

# Replace with your token or use 'huggingface-cli login' in terminal
login("***REMOVED***ESWJGUGdFGbShXfXYdosaeFzzXpeeykKsX")

def format_ultrachat(example):
    """
    Ensures UltraChat is in the correct Generic Messages format.
    """
    return {
        "messages": example['messages'],
        "source": "ultrachat_200k"
    }

def format_openhermes(example):
    """
    Converts OpenHermes (conversations) to Generic Messages format.
    """
    # OpenHermes uses 'conversations' with 'from' (human/gpt) and 'value'
    messages = []
    for turn in example['conversations']:
        role = "user" if turn['from'] == 'human' else "assistant"
        messages.append({"role": role, "content": turn['value']})
    
    return {
        "messages": messages,
        "source": "openhermes"
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Data Preparation Script (English Only)")
    parser.add_argument("--output_dir", type=str, default="./data/finetuning", help="Directory to save the processed dataset")
    parser.add_argument("--sample_limit", type=int, default=500_000, help="Total sample limit (default: 500k)")
    parser.add_argument("--dry_run", action="store_true", help="Run with a small limit for verification")
    args = parser.parse_args()

    if args.dry_run:
        args.sample_limit = 1_000  # 1k samples for dry run
        print("--- DRY RUN MODE ACTIVATED ---")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "finetuning_data.jsonl")

    print(f"Starting Fine-tuning Data Preparation...")
    print(f"Target Sample Limit: {args.sample_limit:,}")
    print(f"Output Directory: {args.output_dir}")

    current_samples = 0
    buffer = []

    # Datasets:
    # 1. UltraChat 200k (English)
    # 2. OpenHermes 2.5 (English) - Replaces Aya for English-only focus
    
    dataset_configs = [
        {"name": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "formatter": format_ultrachat},
        {"name": "teknium/OpenHermes-2.5", "split": "train", "formatter": format_openhermes}
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        for config in dataset_configs:
            if current_samples >= args.sample_limit:
                break
                
            ds_name = config["name"]
            print(f"\nProcessing {ds_name}...")
            
            try:
                ds = load_dataset(ds_name, split=config["split"], streaming=True)
                formatter = config["formatter"]
                
                for sample in ds:
                    if current_samples >= args.sample_limit:
                        break
                    
                    try:
                        formatted_sample = formatter(sample)
                        f.write(json.dumps(formatted_sample, ensure_ascii=False) + "\n")
                        current_samples += 1
                        
                        if current_samples % 10_000 == 0:
                            print(f"Progress: {current_samples:,} / {args.sample_limit:,} samples")
                            
                    except Exception as e:
                        # Skip malformed samples
                        continue

            except Exception as e:
                print(f"Error processing {ds_name}: {e}")

    print(f"\nProcessing Complete!")
    print(f"Total Samples Collected: {current_samples:,}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()