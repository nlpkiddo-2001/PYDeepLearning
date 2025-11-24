import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect, DetectorFactory
from tqdm import tqdm
from collections import Counter

# Ensure consistent results
DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def extract_text_from_item(item):
    """
    Extracts text content based on known schemas.
    """
    # 1. 'text' field (Pre-training / Mid-training standardized)
    if 'text' in item:
        return item['text']
    
    # 2. 'messages' field (Fine-tuning / Chat)
    if 'messages' in item:
        # Concatenate user and assistant messages for detection
        text = ""
        for msg in item['messages']:
            if msg.get('content'):
                text += msg['content'] + " "
        return text.strip()
    
    # 3. Fallback: dump everything
    return str(item)

def analyze_file(file_path, sample_limit=None):
    print(f"Analyzing {file_path}...")
    languages = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    total_lines = len(lines)
    if sample_limit:
        lines = lines[:sample_limit]
        print(f"  -> Sampling {len(lines)} out of {total_lines} lines.")
        
    for line in tqdm(lines, desc="Detecting Languages"):
        try:
            item = json.loads(line)
            text = extract_text_from_item(item)
            if text and len(text.strip()) > 0:
                # Truncate text for speed if it's huge
                lang = detect_language(text[:1000]) 
                languages.append(lang)
            else:
                languages.append("empty")
        except json.JSONDecodeError:
            continue

    return languages

def main():
    parser = argparse.ArgumentParser(description="Analyze Dataset Language Distribution")
    parser.add_argument("--input_dir", type=str, default="./data", help="Root directory containing dataset folders")
    parser.add_argument("--sample_limit", type=int, default=None, help="Limit number of samples per file for speed")
    args = parser.parse_args()

    all_stats = []

    # Walk through the directory
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                full_path = os.path.join(root, file)
                stage = os.path.basename(root) # e.g., 'pretraining', 'midtraining'
                
                langs = analyze_file(full_path, args.sample_limit)
                counts = Counter(langs)
                total = len(langs)
                
                print(f"\nResults for {file} ({stage}):")
                df = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])
                df['percentage'] = (df['count'] / total) * 100
                df = df.sort_values('count', ascending=False)
                print(df.head(10))
                
                # Save stats
                stats_file = os.path.join(root, f"{file}_stats.csv")
                df.to_csv(stats_file)
                print(f"  -> Stats saved to {stats_file}")

                # Plot
                plt.figure(figsize=(10, 6))
                # Top 10 languages
                top_10 = df.head(10)
                plt.bar(top_10.index, top_10['count'])
                plt.title(f"Language Distribution - {stage}/{file}")
                plt.xlabel("Language")
                plt.ylabel("Count")
                plt.tight_layout()
                plot_file = os.path.join(root, f"{file}_dist.png")
                plt.savefig(plot_file)
                print(f"  -> Plot saved to {plot_file}")
                plt.close()

if __name__ == "__main__":
    main()
