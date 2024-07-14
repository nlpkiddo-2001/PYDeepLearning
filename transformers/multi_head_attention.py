import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=4, num_heads=2, num_dropout=0.3):
        super().__init__()
        self.d = d_model // num_heads
        self.num_heads = num_heads

        self.dropout = nn.Dropout(num_dropout)

        self.linear_Qs = nn.ModuleList([nn.Linear(d_model, self.d)
                                        for _ in range(num_heads)])
        self.linear_Ks = nn.ModuleList([nn.Linear(d_model, self.d)
                                        for _ in range(num_heads)])
        self.linear_Vs = nn.ModuleList([nn.Linear(d_model, self.d)
                                        for _ in range(num_heads)])

        self.multi_head_attention_linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # shape(Q) = [B x seq_len x D/num_heads]
        # shape(K, V) = [B x seq_len x D/num_heads]
        query_Key_matmul = torch.matmul(query, key.permute(0,2,1))
        scores = query_Key_matmul/math.sqrt(self.d)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, value)
        # scaling
        return output, attention_weights

    def forward(self, pre_q, pre_k, pre_v,mask=None):
        Query = [linear_q(pre_q) for linear_q in self.linear_Qs]
        Key = [linear_k(pre_k) for linear_k in self.linear_Ks]
        Value = [linear_v(pre_v) for linear_v in self.linear_Vs]

        output_per_head = []
        attention_weights_per_head = []

        for Query_, Key_, Value_ in zip(Query, Key, Value):
            output, attention_weight = self.scaled_dot_product_attention(Query_, Key_, Value_, mask)

            output_per_head.append(output)
            attention_weights_per_head.append(attention_weight)

        output = torch.cat(output_per_head, -1)
        attn_weights = torch.stack(attention_weights_per_head).permute(1, 0, 2, 3)

        projection = self.dropout(self.mha_linear(output))

        return projection, attn_weights
