import os
import json
import random
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

def create_dummy_tokenizer(output_path):
    # Create a simple BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Add special tokens
    special_tokens = ["<|endoftext|>", "<|padding|>"]
    tokenizer.add_special_tokens(special_tokens)
    
    # Train on some dummy data (in memory)
    from tokenizers import trainers
    trainer = trainers.BpeTrainer(vocab_size=1000, special_tokens=special_tokens)
    tokenizer.train_from_iterator(["This is a test sentence." for _ in range(100)], trainer)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"Created dummy tokenizer at {output_path}")

def create_dummy_data(output_dir, num_lines=100):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data.jsonl")
    
    with open(output_path, 'w') as f:
        for i in range(num_lines):
            text = f"This is a dummy sentence number {i}. " * 10
            f.write(json.dumps({"text": text}) + "\n")
            
    print(f"Created dummy data at {output_path}")

def main():
    create_dummy_tokenizer("./data/tokenizer.json")
    create_dummy_data("./data/pretraining")
    create_dummy_data("./data/midtraining")
    create_dummy_data("./data/finetuning")

if __name__ == "__main__":
    main()
