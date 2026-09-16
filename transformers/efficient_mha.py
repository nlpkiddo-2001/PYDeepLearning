import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=4, num_heads=2, dropout=0.3):
        super().__init__()
        self.d = d_model / num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)

        self.mha_linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Query, Key, Value, mask=None):
        Q_K_matmul = torch.matmul(Query, Key.permute(0, 1, 3, 2))
        scores = Q_K_matmul / m.sqrt(self.d)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, Value)
        return output, attention_weights

    def forward(self, pre_q, pre_k, pre_v, mask=None):
        Q = self.linear_Q(pre_q)
        K = self.linear_K(pre_k)
        V = self.linear_V(pre_v)

        batch_size = pre_q.shape[0]

        Q = Q.reshape(batch_size, self.num_heads, -1, self.d)
        K = K.reshape(batch_size, self.num_heads, -1, self.d)
        V = V.reshape(batch_size, self.num_heads, -1, self.d)

        output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.reshape(batch_size, -1, self.d_model)
        projection = self.dropout(self.mha_linear(output))

        return projection, attn_weights
