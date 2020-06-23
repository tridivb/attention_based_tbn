import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).expand(
            -1, dim_size // 2
        ) * torch.arange(1, dim_size // 2 + 1)
        pe[:, 0::2] = torch.sin(position)
        pe[:, 1::2] = torch.cos(position)
        # pe = torch.flip(pe, dims=(0,1))
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


class SoftAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(SoftAttention, self).__init__()
        self.attention_layer = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, bias=True
        )

    def forward(self, query, key, value):
        attention_out, attention_wts = self.attention_layer(query, key, value)
        return attention_out, attention_wts


class UniModalAttention(torch.nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size=256,
        use_gumbel=True,
        temperature=1,
        one_hot=True,
    ):
        super(UniModalAttention, self).__init__()
        # in size is number of channels in input feature
        # out_size is the dimension of the distribution or size of audio feature along temporal axis
        self.seq = nn.Sequential(
            nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_size)
        )
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        self.one_hot = one_hot

    def forward(self, input1, input2):
        logits = self.seq(input1)
        if self.training and self.use_gumbel:
            mul_matrix = F.gumbel_softmax(
                logits, tau=self.temperature, hard=self.one_hot
            )
        else:
            mul_matrix = F.softmax(logits, dim=1)
            # calculate weighted sum over the feature matrix
        out = input2 * mul_matrix.unsqueeze(dim=1)
        out = out.sum(dim=2)
        return out, mul_matrix


class PrototypeAttention(torch.nn.Module):
    def __init__(
        self,
        in_size,
        win_size,
        hidden_size=256,
        use_gumbel=True,
        temperature=1,
        device=torch.device("cuda"),
    ):
        super(PrototypeAttention, self).__init__()
        # in size is number of channels in input feature
        self.in_size = in_size
        self.win_size = win_size
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        self.device = device

        self._create_prototype_weights()
        no_prototypes = self.prototype_wts.shape[0]

        self.seq = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, no_prototypes),
        )

    def _create_prototype_weights(self):
        proto_wt1 = cv2.getGaussianKernel(self.win_size, sigma=1)
        shift = self.win_size // 2 - 2
        proto_wt2 = np.roll(proto_wt1, -shift)
        proto_wt3 = np.roll(proto_wt1, shift)
        prototype_wts = np.concatenate((proto_wt1, proto_wt2, proto_wt3), axis=1).T
        prototype_wts = nn.Parameter(
            torch.from_numpy(prototype_wts).float().to(self.device)
        )
        prototype_wts.requires_grad = True

        self.register_buffer("prototype_wts", prototype_wts)

    def forward(self, input1, input2):
        logits = self.seq(input1)
        if self.training and self.use_gumbel:
            mul_matrix = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        else:
            mul_matrix = F.softmax(logits, dim=1)
        # calculate weighted multiplier over the prototype weights
        attention_wts = torch.matmul(mul_matrix, self.prototype_wts)
        # calculate weighted sum over the feature matrix
        out = input2 * attention_wts.unsqueeze(dim=1)
        out = out.sum(dim=2)
        return out, attention_wts
