import torch.nn as nn

from residual_layer_norm import ResidualLayerNorm
from multi_head_attention import MultiHeadAttention
from positional_feed_forward import Positional_Feed_Forward_Network


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3, efficient_mha=False):
        super().__init__()

        self.norm1 = ResidualLayerNorm(d_model, dropout)
        self.norm2 = ResidualLayerNorm(d_model, dropout)

        if efficient_mha:
            pass
        else:
            self.mha = MultiHeadAttention(d_model, num_heads, dropout)

        self.feed_forward = Positional_Feed_Forward_Network(d_model, d_ff, dropout)

    def forward(self, X, mask):
        multi_head_attention, encoder_attention_weights = self.mha(X, X, X, mask)

        norm1 = self.norm1(multi_head_attention, X)
        feed_forward = self.feed_forward(norm1)

        norm2 = self.norm2(feed_forward, norm1)

        return norm2, encoder_attention_weights

