import torch
import os
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# Configuration
MODEL_PATH = "/workspace/pretrain_slm/checkpoints/hf_model"
TOKENIZER_PATH = "data/tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_path, tokenizer_path):
    print(f"Loading tokenizer from {tokenizer_path}...")
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        # Ensure special tokens are set if needed, though they might be in the file
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|endoftext|>"
            tokenizer.eos_token = "<|endoftext|>"
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

    print(f"Loading model from {model_path}...")
    try:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        )
        if DEVICE == "cpu":
            model = model.to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check if the path is correct. If you are running locally, adjust MODEL_PATH.")
        raise

    return model, tokenizer

def format_prompt(messages):
    """
    Format messages similar to dataloader.py training format.
    Training format: <|user|>user_msg<|assistant|>assistant_msg<|endoftext|>
    Inference format should end with <|assistant|> so the model generates the response.
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
    
    # Join parts and append the start of the assistant response
    prompt = "\n".join(formatted_parts) + "\n<|assistant|>\n"
    return prompt

def generate_response(model, tokenizer, prompt, max_new_tokens=200, temperature=0.01, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Remove token_type_ids if present, as LlamaForCausalLM doesn't use them
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the newly generated part (after the prompt)
    # Be careful with decoding, sometimes slight variations happen at boundaries
    # Simple approach: return the whole thing or split by prompt
    
    return generated_text

def main():
    # Check if paths exist (optional, for better error msgs)
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Warning: Tokenizer not found at {TOKENIZER_PATH}")

    # Initialize
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)
    
    # Example conversation
    messages = [
        {"role":"system", "content": "You are a helpful assistant developed by Vipin"},
        {"role": "user", "content": "Who are you?"}
    ]
    
    print("-" * 50)
    print("Input Messages:")
    for msg in messages:
        print(f"{msg['role']}: {msg['content']}")
        
    prompt = format_prompt(messages)
    print("-" * 50)
    print(f"Formatted Prompt:\n{repr(prompt)}")
    print("-" * 50)
    
    print("Generating response...")
    response = generate_response(model, tokenizer, prompt)
    
    print("Full Output:")
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    main()
