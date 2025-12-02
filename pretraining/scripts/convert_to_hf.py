import os
import sys
import argparse
import torch
import json
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import ModelConfig
torch.serialization.add_safe_globals([ModelConfig])

def convert_to_hf(checkpoint_path, output_dir):
    """
    Convert custom Llama-style GPT checkpoint to Hugging Face LlamaForCausalLM.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    config_obj = checkpoint.get("config", None)

    if config_obj is None:
        raise ValueError("No config found in checkpoint! Cannot infer model architecture.")
    
    vocab_size = getattr(config_obj, 'vocab_size', 50257)
    n_layer = getattr(config_obj, 'n_layers', 32)
    n_head = getattr(config_obj, 'n_heads', 32)
    n_embd = getattr(config_obj, 'dim', 4096)
    n_kv_heads = getattr(config_obj, 'n_kv_heads', None)
    if n_kv_heads is None:
        n_kv_heads = n_head
    block_size = getattr(config_obj, 'max_seq_len', 4096)
    norm_eps = getattr(config_obj, 'norm_eps', 1e-5)
    
    clean_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_sd[k[7:]] = v
        else:
            clean_sd[k] = v
    
    ffn_hidden_dim = None
    if "layers.0.feed_forward.w1.weight" in clean_sd:
        ffn_hidden_dim = clean_sd["layers.0.feed_forward.w1.weight"].shape[0]
    
    if ffn_hidden_dim is None:
        hidden_dim = 4 * n_embd
        hidden_dim = int(2 * hidden_dim / 3)
        ffn_dim_multiplier = getattr(config_obj, 'ffn_dim_multiplier', None)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        multiple_of = getattr(config_obj, 'multiple_of', 256)
        ffn_hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    
    print(f"\nModel Configuration:")
    print(f"  vocab_size:     {vocab_size}")
    print(f"  n_layers:       {n_layer}")
    print(f"  n_heads:        {n_head}")
    print(f"  n_kv_heads:     {n_kv_heads}")
    print(f"  dim:            {n_embd}")
    print(f"  ffn_hidden_dim: {ffn_hidden_dim}")
    print(f"  max_seq_len:    {block_size}")
    print(f"  norm_eps:       {norm_eps}")
    
    # Create HuggingFace Llama config
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=n_embd,
        intermediate_size=ffn_hidden_dim,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=block_size,
        rms_norm_eps=norm_eps,
        tie_word_embeddings=True,
        rope_theta=10000.0,
    )
    
    print(f"\nCreating HuggingFace LlamaForCausalLM model...")
    hf_model = LlamaForCausalLM(config)
    hf_state_dict = hf_model.state_dict()
    
    # Remove "module." prefix if present (DDP)
    clean_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_sd[k[7:]] = v
        else:
            clean_sd[k] = v
    
    new_sd = {}
    
    print(f"\nMapping weights...")
    
    if "tok_embeddings.weight" in clean_sd:
        new_sd["model.embed_tokens.weight"] = clean_sd["tok_embeddings.weight"]
        new_sd["lm_head.weight"] = clean_sd["tok_embeddings.weight"]  # Tied weights
        print("  ✓ Mapped embeddings")
    
    if "norm.weight" in clean_sd:
        new_sd["model.norm.weight"] = clean_sd["norm.weight"]
        print("  ✓ Mapped final norm")
    
    for i in range(n_layer):
        layer_prefix_custom = f"layers.{i}"
        layer_prefix_hf = f"model.layers.{i}"
        
        mappings = {
            f"{layer_prefix_custom}.attention.wq.weight": f"{layer_prefix_hf}.self_attn.q_proj.weight",
            f"{layer_prefix_custom}.attention.wk.weight": f"{layer_prefix_hf}.self_attn.k_proj.weight",
            f"{layer_prefix_custom}.attention.wv.weight": f"{layer_prefix_hf}.self_attn.v_proj.weight",
            f"{layer_prefix_custom}.attention.wo.weight": f"{layer_prefix_hf}.self_attn.o_proj.weight",
            
            f"{layer_prefix_custom}.feed_forward.w1.weight": f"{layer_prefix_hf}.mlp.gate_proj.weight",
            f"{layer_prefix_custom}.feed_forward.w3.weight": f"{layer_prefix_hf}.mlp.up_proj.weight",
            f"{layer_prefix_custom}.feed_forward.w2.weight": f"{layer_prefix_hf}.mlp.down_proj.weight",

            f"{layer_prefix_custom}.attention_norm.weight": f"{layer_prefix_hf}.input_layernorm.weight",
            f"{layer_prefix_custom}.ffn_norm.weight": f"{layer_prefix_hf}.post_attention_layernorm.weight",
        }
        
        for custom_key, hf_key in mappings.items():
            if custom_key in clean_sd:
                new_sd[hf_key] = clean_sd[custom_key]
        
        if i == 0:
            print(f"  ✓ Mapped layer 0 (showing first layer only)")
    
    print(f"  ✓ Mapped all {n_layer} layers")
    
    missing, unexpected = hf_model.load_state_dict(new_sd, strict=False)
    
    print(f"\nWeight loading summary:")
    print(f"  Total keys mapped: {len(new_sd)}")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    
    if missing:
        print(f"\n  Missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"\n  Unexpected keys (first 5): {unexpected[:5]}")
    
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir)
    
    print(f"\n✅ Successfully saved HuggingFace model to {output_dir}")
    print(f"\nTo use this model:")
    print(f"  from transformers import LlamaForCausalLM")
    print(f"  model = LlamaForCausalLM.from_pretrained('{output_dir}')")
    
    return hf_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert custom Llama-style checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output HF directory")
    args = parser.parse_args()
    
    convert_to_hf(args.checkpoint, args.output_dir)