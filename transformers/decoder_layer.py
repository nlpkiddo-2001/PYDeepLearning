import torch.nn as nn
import torch

from residual_layer_norm import ResidualLayerNorm
from multi_head_attention import MultiHeadAttention
from positional_feed_forward import Positional_Feed_Forward_Network


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3, efficient_mha=False):
        super().__init__()
        self.norm1 = ResidualLayerNorm(d_model)
        self.norm2 = ResidualLayerNorm(d_model)
        self.norm3 = ResidualLayerNorm(d_model)

        if efficient_mha:
            pass
        else:
            self.masked_mha = MultiHeadAttention(d_model, num_heads, dropout)
            self.enc_dec_mha = MultiHeadAttention(d_model, num_heads, dropout)

        self.feed_forward = Positional_Feed_Forward_Network(d_model, d_ff)

    def forward(self, x , encoded_output, trgt_mask, src_mask):
        masked_mha, masked_mha_weights = self.masked_mha(x,x,x, mask=trgt_mask)

        norm1 = self.norm1(masked_mha, x)

        enc_dec_mha, enc_dec_mha_attn_weights = self.enc_dec_mha(norm1, encoded_output, encoded_output, mask=src_mask)

        norm2 = self.norm2(enc_dec_mha, norm1)

        feed_forward = self.feed_forward(norm2)

        norm3 = self.norm3(feed_forward, norm2)

        return norm3, masked_mha_weights, enc_dec_mha_attn_weights


