import os
import argparse
import torch
import json
from transformers import GPT2Config, GPT2LMHeadModel

def convert_to_hf(checkpoint_path, output_dir, model_config_path=None):
    """
    Convert custom GPT checkpoint to Hugging Face GPT2LMHeadModel.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract state dict
    state_dict = checkpoint.get("model", checkpoint)
    
    # Extract config
    # If config is in checkpoint, use it. Otherwise use default or provided path.
    config_obj = checkpoint.get("config", None)
    
    # Map custom config to HF GPT2Config
    # Assuming our GPT is similar to GPT-2/Llama
    # We need to know the architecture details.
    # Based on src/model.py (which I haven't seen fully but assumed standard GPT)
    
    # Let's assume standard GPT-2 config for now, user can adjust.
    # We try to infer from config_obj if available.
    
    vocab_size = 50304 # Default nanoGPT
    n_layer = 12
    n_head = 12
    n_embd = 768
    block_size = 1024
    
    if config_obj:
        # Try to extract attributes
        vocab_size = getattr(config_obj, 'vocab_size', vocab_size)
        n_layer = getattr(config_obj, 'n_layer', n_layer)
        n_head = getattr(config_obj, 'n_head', n_head)
        n_embd = getattr(config_obj, 'n_embd', n_embd)
        block_size = getattr(config_obj, 'block_size', getattr(config_obj, 'max_seq_len', block_size))
    
    print(f"Inferred Config: vocab_size={vocab_size}, n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}, ctx={block_size}")
    
    ***REMOVED***config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        activation_function="gelu_new", # or swiglu if llama
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=True
    )
    
    ***REMOVED***model = GPT2LMHeadModel(***REMOVED***config)
    
    # Map weights
    # Custom GPT usually has:
    # transformer.wte.weight -> wte.weight
    # transformer.wpe.weight -> wpe.weight
    # transformer.h.0.ln_1.weight -> h.0.ln_1.weight
    # ...
    
    # We need to inspect keys to map correctly.
    # Let's do a naive mapping and print mismatches.
    
    ***REMOVED***sd = ***REMOVED***model.state_dict()
    new_sd = {}
    
    # Remove "module." prefix if DDP
    clean_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_sd[k[7:]] = v
        else:
            clean_sd[k] = v
            
    # Mapping rules (adjust based on src/model.py)
    # Assuming nanoGPT style:
    # transformer.wte.weight -> transformer.wte.weight
    # transformer.wpe.weight -> transformer.wpe.weight
    # transformer.h.N... -> transformer.h.N...
    # lm_head.weight -> lm_head.weight
    
    # If our model uses "transformer" prefix, it might match HF GPT2.
    
    keys_matched = 0
    for k, v in clean_sd.items():
        if k in ***REMOVED***sd:
            if ***REMOVED***sd[k].shape == v.shape:
                new_sd[k] = v
                keys_matched += 1
            else:
                print(f"Shape mismatch for {k}: HF {***REMOVED***sd[k].shape} vs Ckpt {v.shape}")
                # Handle Transpose for Conv1D if needed (GPT2 uses Conv1D)
                if len(***REMOVED***sd[k].shape) == 2 and len(v.shape) == 2 and ***REMOVED***sd[k].shape == v.T.shape:
                    new_sd[k] = v.T
                    print(f"  -> Transposed {k}")
                    keys_matched += 1
        else:
            print(f"Key not found in HF model: {k}")
            
    print(f"Matched {keys_matched} / {len(***REMOVED***sd)} keys.")
    
    # Load
    missing, unexpected = ***REMOVED***model.load_state_dict(new_sd, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    ***REMOVED***model.save_pretrained(output_dir)
    print(f"Saved HF model to {output_dir}")
    
    # Save tokenizer if path provided?
    # User can just copy tokenizer.json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output HF directory")
    args = parser.parse_args()
    
    convert_to_hf(args.checkpoint, args.output_dir)
