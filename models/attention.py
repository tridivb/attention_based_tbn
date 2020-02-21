import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        dim_size,
        dropout=0.0,
        max_len=25,
        encoding_type="concat",
        device=torch.device("cuda"),
    ):
        super(PositionalEncoding, self).__init__()
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.encoding_type = encoding_type
        self.dim_size = dim_size
        self.max_len = max_len
        self.dropout = dropout

        pe = torch.zeros(max_len, dim_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        pe[:, 0::2] = torch.sin(position)
        pe[:, 1::2] = torch.cos(position)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x.squeeze(2)
        b, c, s = x.shape
        if self.encoding_type == "add":
            x = x + self.pe[: x.size(0), :]
        elif self.encoding_type == "concat":
            x = torch.cat((x, self.pe.expand(b, self.dim_size, self.max_len)), dim=1)
        if self.dropout > 0:
            return self.dropout(x)
        else:
            return x


class AttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(AttentionLayer, self).__init__()
        self.attention_layer = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, bias=True
        )

    def forward(self, query, key, value):
        attention_out, attention_wts = self.attention_layer(query, key, value)
        return attention_out, attention_wts
