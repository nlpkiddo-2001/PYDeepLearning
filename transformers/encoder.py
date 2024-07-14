import torch.nn as nn
from encoder_layer import EncoderLayer
from embed import Embeddings
from positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, Embedding: Embeddings, d_model, num_heads, num_layer, d_ff, device="cpu", dropout=0.3,
                 efficient_mha=False):
        super().__init__()

        self.embedding = Embedding
        self.positional_encoding = PositionalEncoding(d_model, device=device)

        self.encoders = nn.ModuleList([EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
            efficient_mha
        ) for layer in range(num_layer)])

    def forward(self, x, mask=None):
        embeddings = self.embedding(x)
        encoding = self.PE(embeddings)

        for encoder in self.encoders:
            encoding, encoder_attention_weights = encoder(encoding, mask)

        return encoding, encoder_attention_weights
