import argparse
import json
import os
import random
import re
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from tqdm import tqdm
import yaml

# Configuration for datasets
# Configuration for datasets
# Maps the generic names in config.midtrain.yaml to HF datasets
DATASET_MAPPINGS = {
    "midtrain": {
        "C4": {"path": "allenai/c4", "subset": "en", "split": "train", "streaming": True},
        "Starcoder": {"path": "bigcode/starcoderdata", "subset": None, "split": "train", "streaming": True},
        # Using OpenMathInstruct as a high-quality math equivalent
        "Math": {"path": "nvidia/OpenMathInstruct-1", "subset": None, "split": "train", "streaming": True},
        # Using Flan v2 (subsampled via streaming)
        # "FLAN": {"path": "WeiLiu/flan-v2", "subset": None, "split": "train", "streaming": True},
        # TriviaQA for knowledge
        "KnowledgeQA": {"path": "mandarjoshi/trivia_qa", "subset": "rc", "split": "train", "streaming": True},
        # DCLM baseline for high quality web text
        "DCLM": {"path": "mlfoundations/dclm-baseline-1.0", "subset": None, "split": "train", "streaming": True},
    },
    "sft": {
        # High quality mix
        "TuluV2": {"path": "allenai/tulu-v2-sft-mixture", "subset": None, "split": "train", "streaming": False},
        "GSM8K": {"path": "openai/gsm8k", "subset": "main", "split": "train", "streaming": False},
        # Added datasets
        "OASST": {"path": "OpenAssistant/oasst_top1_2023-08-25", "subset": None, "split": "train", "streaming": False},
        "UltraChat": {"path": "HuggingFaceH4/ultrachat_200k", "subset": None, "split": "train_sft", "streaming": False},
        "OrcaMath": {"path": "microsoft/orca-math-word-problems-200k", "subset": None, "split": "train", "streaming": False},
    }
}

def format_midtrain_example(example: Dict[str, Any], dataset_name: str) -> Dict[str, str]:
    """
    Format examples into raw text for mid-training.
    Returns: {"text": "..."}
    """
    text = ""
    
    if dataset_name == "C4":
        text = example.get("text", "")
        
    elif dataset_name == "Starcoder":
        text = example.get("content", "")
        
    elif dataset_name == "Math":
        # OpenMathInstruct: question + generated_solution
        q = example.get("question", "")
        sol = example.get("expected_answer", "")
        # Format: "Problem: ... \nSolution: ..."
        text = f"Problem:\n{q}\n\nSolution:\n{sol}"
        
    elif dataset_name == "FLAN":
        # Flan-v2 has inputs and targets
        inputs = example.get("inputs", "")
        targets = example.get("targets", "")
        text = f"{inputs}\n{targets}"
        
    elif dataset_name == "KnowledgeQA":
        # TriviaQA
        q = example.get("question", "")
        # Taking the first answer alias or value
        ans = ""
        if "answer" in example and "value" in example["answer"]:
             ans = example["answer"]["value"]
        text = f"Question: {q}\nAnswer: {ans}"
        
    elif dataset_name == "DCLM":
        text = example.get("text", "")
        
    return {"text": text}

def format_sft_example(example: Dict[str, Any], dataset_name: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Format examples into chat messages for SFT.
    Returns: {"messages": [{"role": "user", "content": ...}, ...]}
    """
    messages = []
    
    if dataset_name == "TuluV2":
        # Tulu structure: messages list
        if "messages" in example:
            # Already in format, just clean up
            raw_msgs = example["messages"]
            for m in raw_msgs:
                messages.append({"role": m["role"], "content": m["content"]})
        else:
            return None

    elif dataset_name == "GSM8K":
        q = example.get("question", "")
        org_ans = example.get("answer", "")
        # Create a standard user-assistant turn
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": org_ans}
        ]

    elif dataset_name == "OASST":
        # OASST Top1 usually has a text prompt and text response or similar structure, 
        # but the HF dataset usually preserves the tree or flattened conversation.
        # "oasst_top1_2023-08-25" typically has 'text' which is the full conversation or 'conversations'
        # Checking schema: it has 'text' which is the raw string, but we want structured.
        # Actually this dataset is often just text. Let's try to parse or if it has metadata.
        # Wait, oasst_top1 is often just the best response. 
        # If it's just text, we might skip or try to split.
        # HOWEVER: generic OASST has flexible schema. Let's assume standard "text" is user+assistant.
        # Better: usage of "timdettmers/openassistant-guanaco" is cleaner, but user asked for downloads.
        # Let's try to see if we can just take "text" and put it as content if we can't split?
        # No, dataloader expects messages.
        # Let's look for "messages" or "conversations" fields first.
        # If not, checking typical OASST structure: it might have "instruction" and "output" columns if processed.
        # If raw text: usually "### Human: ... ### Assistant: ..."
        # Let's try to split by special tokens if present, or just treat as one block if desperate? 
        # No, dataloader needs roles.
        # Let's assume the HF dataset version has 'messages' or we map 'text' to 'assistant' (bad).
        # HACK: For this specific dataset "OpenAssistant/oasst_top1_2023-08-25", it often comes formatted as text.
        # Let's try to extract if possible. If not, we might swap to a better formatted one like "timdettmers/openassistant-guanaco" which IS oasst top1.
        # But let's stick to the requested name if we can, or just use a robust regex.
        # Simple regex for "### Human:" style if present.
        text = example.get("text", "")
        # Try split
        parts = re.split(r"(### Human:|### Assistant:)", text)
        # minimal parsing
        if len(parts) > 1:
            current_role = None
            for p in parts:
                if "Human:" in p:
                    current_role = "user"
                elif "Assistant:" in p:
                    current_role = "assistant"
                else:
                    if current_role and p.strip():
                        messages.append({"role": current_role, "content": p.strip()})
        else:
            # Maybe it provides "instruction" / "output"?
            if "instruction" in example and "output" in example:
                 messages.append({"role": "user", "content": example["instruction"]})
                 messages.append({"role": "assistant", "content": example["output"]})
            else:
                 # Skip if we can't parse
                 return None

    elif dataset_name == "UltraChat":
        # Usually has 'messages' list
        if "messages" in example:
            for m in example["messages"]:
                messages.append({"role": m["role"], "content": m["content"]})
        else:
            return None

    elif dataset_name == "OrcaMath":
        # structured as 'question' and 'answer'
        q = example.get("question", "")
        a = example.get("answer", "")
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]
        
    return {"messages": messages}

def download_and_process(
    config_path: str,
    stage: str,
    max_samples_per_ds: int = 100000,
    output_base_dir: str = "./data",
    debug: bool = False
):

    """
    Download, format, and save datasets.
    """
    if stage == "midtrain":
        # Load midtrain config to get weights (optional, here we just fetch data)
        # We respect the paths in config if provided, otherwise default
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            datasets_conf = config.get("datasets", [])
        except Exception:
            print("Could not load config file, using default mappings only.")
            datasets_conf = []

        # Determine output paths from config or default
        # We will create a map of Name -> OutputPath
        ds_output_map = {}
        for d in datasets_conf:
            ds_output_map[d["name"]] = d["path"]

        sources = DATASET_MAPPINGS["midtrain"]
        
        for name, info in sources.items():
            print(f"Processing {name}...")
            
            # Determine target path
            if name in ds_output_map:
                save_dir = ds_output_map[name]
            else:
                save_dir = os.path.join(output_base_dir, "midtraining", name.lower())
                
            os.makedirs(save_dir, exist_ok=True)
            output_file_train = os.path.join(save_dir, "train.jsonl")
            output_file_val = os.path.join(save_dir, "val.jsonl")
            
            if os.path.exists(output_file_train):
                print(f"  {output_file_train} exists. Skipping.")
                continue

            # Load dataset (streaming for large ones)
            ds = load_dataset(
                info["path"], 
                info["subset"], 
                split=info["split"], 
                streaming=info["streaming"],
            )
            
            if info["streaming"]:
                ds = ds.shuffle(buffer_size=10000, seed=42)
                # Take max_samples + some for val
                if debug:
                    print("MIDTRAIN DEBUG MODE ENABLED → Downloading only 50 samples")
                    total_needed = 50   # only 50 samples total
                else:
                    total_needed = int(max_samples_per_ds * 1.1)

                data_iter = iter(ds)
                
                rows = []
                pbar = tqdm(total=total_needed, desc=f"Downloading {name}")
                for _ in range(total_needed):
                    try:
                        row = next(data_iter)
                        formatted = format_midtrain_example(row, name)
                        if formatted["text"].strip():
                            rows.append(formatted)
                        pbar.update(1)
                    except StopIteration:
                        break
                pbar.close()
            else:
                # Non-streaming
                ds = ds.shuffle(seed=42).select(range(min(len(ds), max_samples_per_ds)))
                rows = []
                for row in tqdm(ds, desc=f"Processing {name}"):
                    formatted = format_midtrain_example(row, name)
                    if formatted["text"].strip():
                        rows.append(formatted)

            # Split Train/Val (95/5)
            split_idx = int(len(rows) * 0.95)
            train_rows = rows[:split_idx]
            val_rows = rows[split_idx:]
            
            # Save
            print(f"  Saving {len(train_rows)} train rows to {output_file_train}")
            with open(output_file_train, "w", encoding="utf-8") as f:
                for r in train_rows:
                    f.write(json.dumps(r) + "\n")
                    
            print(f"  Saving {len(val_rows)} val rows to {output_file_val}")
            with open(output_file_val, "w", encoding="utf-8") as f:
                for r in val_rows:
                    f.write(json.dumps(r) + "\n")

    elif stage == "sft":
        # SFT: Consolidated dataset often preferred
        sources = DATASET_MAPPINGS["sft"]
        all_sft_rows = []
        
        for name, info in sources.items():
            print(f"Processing {name}...")
            # Special handling for gated/special datasets if needed
            try:
                ds = load_dataset(
                    info["path"], 
                    info["subset"], 
                    split=info["split"],
                    streaming=info["streaming"],
                )
            except Exception as e:
                print(f"Error loading {name} ({info['path']}): {e}")
                continue
            
            # For SFT, we might take all or a subset. 
            if debug:
                print("SFT DEBUG MODE ENABLED → Downloading only 50 samples")
                limit = 50
            else:
                limit = 50000 
            if name == "TuluV2": limit = 50000
            elif name == "UltraChat": limit = 30000 
            elif name == "OrcaMath": limit = 20000
            elif name == "GSM8K": limit = 10000
            
            if info["streaming"]:
                 pass 
            else:
                if len(ds) > limit:
                     ds = ds.shuffle(seed=42).select(range(limit))
                
                for row in tqdm(ds, desc=f"Formatting {name}"):
                    formatted = format_sft_example(row, name)
                    if formatted and formatted["messages"]:
                        all_sft_rows.append(formatted)

        # Shuffle consolidated data
        random.shuffle(all_sft_rows)
        
        # Split
        split_idx = int(len(all_sft_rows) * 0.95)
        train_rows = all_sft_rows[:split_idx]
        val_rows = all_sft_rows[split_idx:]
        
        # Save to single SFT directory
        save_dir = os.path.join(output_base_dir, "sft")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving {len(train_rows)} SFT train rows to {save_dir}/train.jsonl")
        with open(os.path.join(save_dir, "train.jsonl"), "w", encoding="utf-8") as f:
            for r in train_rows:
                f.write(json.dumps(r) + "\n")
                
        print(f"Saving {len(val_rows)} SFT val rows to {save_dir}/val.jsonl")
        with open(os.path.join(save_dir, "val.jsonl"), "w", encoding="utf-8") as f:
            for r in val_rows:
                f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=["midtrain", "sft"], required=True, help="dataset stage to download")
    parser.add_argument("--config", type=str, default="config.midtrain.yaml", help="Path to midtrain config (for paths)")
    parser.add_argument("--max_samples", type=int, default=200000, help="Max samples per midtrain dataset")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output base directory")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with very small samples")

    
    args = parser.parse_args()
    
    download_and_process(
        args.config, 
        args.stage, 
        args.max_samples, 
        args.output_dir,
        debug=args.debug
    )

