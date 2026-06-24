"""
Three VAD Model architectures
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ------------------
# Module 1
# ------------------

class FrameMLP(nn.Module):
    "This is the simplest VAD possible: classify each frames independently from its 80 dim mel vector"

    def __init__(self, n_mels: int = 80, hidden: int = 128, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_mels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (Batch, n_mels, Time) 
        B, M, T = x.shape
        x = x.transpose(1, 2).reshape(B*T, M)
        out = self.net(x)
        return out.view(B, T)


# ------------------
# Module 2
# ------------------

class ConvBlock(nn.Module):
    """conv1d → batchnorm → ReLU, with 'same' padding so length is preserved."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class FrameCNN(nn.Module):
    """
    1D CNN over time. The mel dimension acts as input channels.
    Each output frame sees a ~150 ms window of input frames (3 layers of kernel-5).
    """

    def __init__(self, n_mels: int = 80, channels: int = 64, n_layers: int = 3,
                 kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        layers = [ConvBlock(n_mels, channels, kernel=kernel, dropout=dropout)]
        for _ in range(n_layers - 1):
            layers.append(ConvBlock(channels, channels, kernel=kernel, dropout=dropout))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels, 1, kernel_size=1)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T)
        h = self.backbone(x)        
        logits = self.head(h)      
        return logits.squeeze(1)    
    
# ------------------
# Module 3
# ------------------

class FrameCRNN(nn.Module):
    """
    CNN extracts local features , GRU integrates them across the whole clip
    """
    def __init__(self, n_mels: int = 80, channels: int = 64, rnn_hidden: int = 64,
                 n_conv: int = 2, kernel: int = 5, dropout: float = 0.1,
                 bidirectional: bool = True):
        super().__init__()
        layers = [ConvBlock(n_mels, channels, kernel=kernel, dropout=dropout)]
        for _ in range(n_conv - 1):
            layers.append(ConvBlock(channels, channels, kernel=kernel, dropout=dropout))
        self.backbone = nn.Sequential(*layers)

        self.rnn = nn.GRU(
            input_size=channels,
            hidden_size=rnn_hidden,
            batch_first=True,
            bidirectional=bidirectional,
        )
        rnn_out_dim = rnn_hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(rnn_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
        h = self.backbone(x)            
        h = h.transpose(1, 2)           
        h, _ = self.rnn(h)            
        logits = self.head(h).squeeze(-1) 
        return logits
    

def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, n_mels: int = 80) -> nn.Module:
    name = name.lower()

    if name == "mlp":
        return FrameMLP(n_mels=n_mels)
    elif name == "cnn":
        return FrameCNN(n_mels=n_mels)
    elif name == "crnn":
        return FrameCRNN(n_mels=n_mels)
