import torch.nn as nn


class Positional_Feed_Forward_Network(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.3):
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, X):
        feed_forward = self.feed_forward(X)
        return feed_forward


