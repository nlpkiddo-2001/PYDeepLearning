import os
import argparse
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Train BPE Tokenizer (English Only)")
    parser.add_argument("--vocab_size", type=int, default=56000, help="Vocabulary size")
    parser.add_argument("--sample_size_gb", type=float, default=1.0, help="Target sample size in GB")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--min_frequency", type=int, default=2, help="Minimum frequency for BPE merges")
    parser.add_argument("--skip_sampling", action="store_true", help="Skip sampling if file exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sample_file = os.path.join(args.output_dir, "tokenizer_sample.txt")
    tokenizer_file = os.path.join(args.output_dir, "tokenizer.json")

    if args.skip_sampling and os.path.exists(sample_file):
        print(f"Skipping sampling. Using existing file: {sample_file}")
    else:
        print(f"Sampling {args.sample_size_gb} GB of English data...")
        
        target_bytes = int(args.sample_size_gb * 1024 * 1024 * 1024)
        current_bytes = 0

        sources = [
            ("uonlp/CulturaX", "en"),
            ("tiiuae/falcon-refinedweb", None)
        ]

        iterators = []
        for dataset_name, lang in sources:
            try:
                print(f"Initializing {dataset_name}...")
                if lang:
                    ds = load_dataset(dataset_name, lang, split="train", streaming=True)
                else:
                    ds = load_dataset(dataset_name, split="train", streaming=True)
                iterators.append(iter(ds))
            except Exception as e:
                print(f"Error initializing {dataset_name}: {e}")

        if not iterators:
            raise RuntimeError("No data sources were successfully initialized!")

        with open(sample_file, "w", encoding="utf-8") as f:
            pbar = tqdm(total=target_bytes, unit="B", unit_scale=True, desc="Sampling Data")
            
            while current_bytes < target_bytes:
                active_sources = 0
                for iterator in iterators:
                    if current_bytes >= target_bytes:
                        break

                    try:
                        for _ in range(5000):
                            if current_bytes >= target_bytes:
                                break
                            try:
                                item = next(iterator)
                                text = item.get("text", "") or item.get("content", "")
                                if not text or len(text.strip()) < 10:
                                    continue
                                    
                                f.write(text + "\n")
                                text_bytes = len(text.encode("utf-8")) + 1
                                current_bytes += text_bytes
                                pbar.update(text_bytes)
                            except StopIteration:
                                break
                            except Exception as e:
                                continue 
                        active_sources += 1
                    except Exception as e:
                        print(f"Error processing source: {e}")
                        continue
                
                if active_sources == 0:
                    print("All sources exhausted.")
                    break
            pbar.close()
        
        print(f"Data sampling complete. Saved to {sample_file}")
        print(f"Final size: {current_bytes / (1024**3):.2f} GB")

    print("\nTraining BPE Tokenizer...")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Minimum frequency: {args.min_frequency}")
    
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    special_tokens = [
        "<|endoftext|>",
        "<|padding|>",
    ]
    
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    print("Training in progress...")
    tokenizer.train([sample_file], trainer)

    tokenizer.save(tokenizer_file)
    print(f"\nTokenizer saved to {tokenizer_file}")

    print("\n" + "="*60)
    print("TOKENIZER VERIFICATION")
    print("="*60)

    tokenizer_loaded = Tokenizer.from_file(tokenizer_file)

    vocab_size = tokenizer_loaded.get_vocab_size()
    print(f"\nVocabulary Size: {vocab_size}")

    eos_id = tokenizer_loaded.token_to_id("<|endoftext|>")
    pad_id = tokenizer_loaded.token_to_id("<|padding|>")
    print(f"EOS Token '<|endoftext|>' ID: {eos_id}")
    print(f"PAD Token '<|padding|>' ID: {pad_id}")

    test_cases = [
        "Hello, world! This is a test of the BPE tokenizer.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and artificial intelligence are transforming technology.",
        "She sells seashells by the seashore. 🌊",
        "print('Hello, World!')\nfor i in range(10):\n    print(i)",
    ]
    
    print("\n" + "-"*60)
    print("TEST CASES")
    print("-"*60)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Original: {test_text[:80]}...")
        
        encoded = tokenizer_loaded.encode(test_text)
        decoded = tokenizer_loaded.decode(encoded.ids)
        
        print(f"Tokens: {len(encoded.ids)}")
        print(f"Token IDs (first 20): {encoded.ids[:20]}")

        if decoded == test_text:
            print("✓ Lossless encoding/decoding")
        else:
            print("✗ WARNING: Decoded text differs from original")
            print(f"Decoded: {decoded[:80]}...")

    long_text = "hello " * 100
    encoded = tokenizer_loaded.encode(long_text)
    compression_ratio = len(long_text) / len(encoded.ids)
    print(f"\n" + "-"*60)
    print(f"Compression Ratio: {compression_ratio:.2f} chars/token")
    print(f"(Higher is better - typically 3-4 for English)")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

    print("\nTo use this tokenizer in your dataloader:")
    print(f"  tokenizer = Tokenizer.from_file('{tokenizer_file}')")
    print(f"  eos_token_id = tokenizer.token_to_id('<|endoftext|>')  # ID: {eos_id}")

if __name__ == "__main__":
    main()