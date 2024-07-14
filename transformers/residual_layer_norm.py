import torch.nn as nn


class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.layerNorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, residual):
        layerNorm = self.layerNorm(self.dropout(X) + residual)
        return layerNorm
