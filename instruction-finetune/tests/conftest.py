import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


@pytest.fixture(scope="session")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")


def make_tiny_model(vocab_size, attn="sdpa", hidden=64, seed=0):
    """random 2-layer Qwen3 with the real vocab, so the real tokenizer and chat template work"""
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=vocab_size, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=hidden // 4, tie_word_embeddings=True,
    )
    cfg._attn_implementation = attn
    model = Qwen3ForCausalLM(cfg).float()
    model.config.use_cache = False
    return model
