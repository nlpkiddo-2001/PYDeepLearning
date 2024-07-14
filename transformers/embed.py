import torch.nn as nn
import math as m


class Embeddings(nn.Module):
    def __init__(self, vocab_size, padding_idx, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x):
        embeddings = self.embed(x)
        return embeddings * m.sqrt(self.d_model)