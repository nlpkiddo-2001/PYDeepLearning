import torch
import os
import sys
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast


current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from model import GPT, ModelConfig


CHECKPOINT_PATH = "/workspace/pretrain_slm/checkpoints/finetune/step_5000.pt"
TOKENIZER_PATH = os.path.join("/workspace/pretrain_slm/data/tokenizer.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.serialization.add_safe_globals([ModelConfig])

def load_model(checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    

    config_obj = checkpoint.get("config", None)
    if config_obj is None:
        raise ValueError("No config found in checkpoint! Cannot initialize model.")
    
    print("Model Config loaded from checkpoint:")
    print(config_obj)
    
    # Initialize model
    model = GPT(config_obj)
    
    # Load state dict
    state_dict = checkpoint.get("model", checkpoint)
    
    # Clean state dict (remove module. prefix from DDP)
    clean_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_sd[k[7:]] = v
        else:
            clean_sd[k] = v
            
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    
    return model

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering """
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9, device=DEVICE):
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) # (1, seq_len)
    
    print(f"Generating {max_new_tokens} tokens...", end="", flush=True)
    
    generated_tokens = []
    
    for _ in range(max_new_tokens):
        # Crop context if needed
        if input_tensor.shape[1] > model.args.max_seq_len:
            input_tensor = input_tensor[:, -model.args.max_seq_len:]
            
        with torch.no_grad():
            logits, _ = model(input_tensor)
            
        # Get last token logits
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Apply top-p
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)
        
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append
        input_tensor = torch.cat([input_tensor, next_token], dim=1)
        generated_token_id = next_token.item()
        generated_tokens.append(generated_token_id)
        
        # Simple progress indicator
        if len(generated_tokens) % 10 == 0:
            print(".", end="", flush=True)
            
        # Stop if EOS (check tokenizer for eos_token_id)
        if generated_token_id == tokenizer.eos_token_id:
            break
            
    print(" Done!")
    
    # Decode only the generated part
    full_text = tokenizer.decode(input_tensor[0].cpu().tolist())
    return full_text

def format_prompt(messages):
    """
    Format messages similar to dataloader.py training format.
    """
    formatted_parts = []
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'user':
            formatted_parts.append(f"<|user|>\n{content}")
        elif role == 'assistant':
            formatted_parts.append(f"<|assistant|>\n{content}")
        elif role == 'system':
            formatted_parts.append(f"<|system|>\n{content}")
    
    prompt = "\n".join(formatted_parts) + "\n<|assistant|>\n"
    return prompt

def main():
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Error: Tokenizer not found at {TOKENIZER_PATH}")
        return

    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|endoftext|>"
            tokenizer.eos_token = "<|endoftext|>"
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    try:
        model = load_model(CHECKPOINT_PATH, DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize chat history
    messages = [
        {"role": "system", "content": "You are a helpful assistant developed by Zoho CRM DI Text"}
    ]

    print("\nChat started (type 'exit' or press Ctrl+C to quit)\n")

    try:
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() == "exit":
                print("Exiting chat.")
                break

            messages.append({"role": "user", "content": user_input})

            prompt = format_prompt(messages)

            output_text = generate(
                model,
                tokenizer,
                prompt,
                temperature=0.3,
                top_p=0.9,
                device=DEVICE,
            )

            # Extract only the latest assistant response
            assistant_response = output_text.split("<|assistant|>")[-1].strip()

            print(f"Assistant: {assistant_response}\n")

            messages.append({"role": "assistant", "content": assistant_response})

    except KeyboardInterrupt:
        print("\nChat interrupted by user. Goodbye!")


if __name__ == "__main__":
    main()
