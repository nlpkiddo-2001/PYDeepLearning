import argparse
import json
import os
import random
import re
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from tqdm import tqdm
import yaml


DATASET_MAPPINGS = {
    "midtrain": {
        "C4": {"path": "allenai/c4", "subset": "en", "split": "train", "streaming": True},
        "Starcoder": {"path": "bigcode/starcoderdata", "subset": None, "split": "train", "streaming": True},
        "Math": {"path": "nvidia/OpenMathInstruct-1", "subset": None, "split": "train", "streaming": True},
        # "FLAN": {"path": "WeiLiu/flan-v2", "subset": None, "split": "train", "streaming": True},
        "KnowledgeQA": {"path": "mandarjoshi/trivia_qa", "subset": "rc", "split": "train", "streaming": True},
        "DCLM": {"path": "mlfoundations/dclm-baseline-1.0", "subset": None, "split": "train", "streaming": True},
    },
    "sft": {
        "TuluV2": {"path": "allenai/tulu-v2-sft-mixture", "subset": None, "split": "train", "streaming": False},
        "GSM8K": {"path": "openai/gsm8k", "subset": "main", "split": "train", "streaming": False},
        "OASST": {"path": "OpenAssistant/oasst_top1_2023-08-25", "subset": None, "split": "train", "streaming": False},
        "UltraChat": {"path": "HuggingFaceH4/ultrachat_200k", "subset": None, "split": "train_sft", "streaming": False},
        "OrcaMath": {"path": "microsoft/orca-math-word-problems-200k", "subset": None, "split": "train", "streaming": False},
    }
}

def format_midtrain_example(example: Dict[str, Any], dataset_name: str) -> Dict[str, str]:
    text = ""
    
    if dataset_name == "C4":
        text = example.get("text", "")
        
    elif dataset_name == "Starcoder":
        text = example.get("content", "")
        
    elif dataset_name == "Math":
        q = example.get("question", "")
        sol = example.get("expected_answer", "")
        text = f"Problem:\n{q}\n\nSolution:\n{sol}"
        
    elif dataset_name == "FLAN":
        inputs = example.get("inputs", "")
        targets = example.get("targets", "")
        text = f"{inputs}\n{targets}"
        
    elif dataset_name == "KnowledgeQA":
        q = example.get("question", "")

        ans = ""
        if "answer" in example and "value" in example["answer"]:
             ans = example["answer"]["value"]
        text = f"Question: {q}\nAnswer: {ans}"
        
    elif dataset_name == "DCLM":
        text = example.get("text", "")
        
    return {"text": text}

def format_sft_example(example: Dict[str, Any], dataset_name: str) -> Dict[str, List[Dict[str, str]]]:

    messages = []
    
    if dataset_name == "TuluV2":

        if "messages" in example:

            raw_msgs = example["messages"]
            for m in raw_msgs:
                messages.append({"role": m["role"], "content": m["content"]})
        else:
            return None

    elif dataset_name == "GSM8K":
        q = example.get("question", "")
        org_ans = example.get("answer", "")

        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": org_ans}
        ]

    elif dataset_name == "OASST":
        text = example.get("text", "")

        parts = re.split(r"(### Human:|### Assistant:)", text)

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
            if "instruction" in example and "output" in example:
                 messages.append({"role": "user", "content": example["instruction"]})
                 messages.append({"role": "assistant", "content": example["output"]})
            else:

                 return None

    elif dataset_name == "UltraChat":

        if "messages" in example:
            for m in example["messages"]:
                messages.append({"role": m["role"], "content": m["content"]})
        else:
            return None

    elif dataset_name == "OrcaMath":
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

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            datasets_conf = config.get("datasets", [])
        except Exception:
            print("Could not load config file, using default mappings only.")
            datasets_conf = []

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
                    total_needed = 50
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
            print(f"  Saving {len(val_rows)} val rows to {output_file_val}")
            with open(output_file_train, "w", encoding="utf-8") as f:
                for r in train_rows:
                    f.write(json.dumps(r) + "\n")
                    
            print(f"  Saving {len(val_rows)} val rows to {output_file_val}")
            with open(output_file_val, "w", encoding="utf-8") as f:
                for r in val_rows:
                    f.write(json.dumps(r) + "\n")

    elif stage == "sft":

        sources = DATASET_MAPPINGS["sft"]
        all_sft_rows = []
        
        for name, info in sources.items():
            print(f"Processing {name}...")

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

