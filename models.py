import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np


class DeepSetLayer(nn.Module):
    def __init__(self,
        dim: int,
        num_layers: int = 2,
        activation = nn.ReLU
    ):
        super(DeepSetLayer, self).__init__()

        def mlp():
            layers = []
            for _ in range(num_layers):
                layers.append(nn.Linear(dim, dim))
                layers.append(activation())
            return nn.Sequential(*layers)

        self.encoder = mlp()
        self.decoder = mlp()

    def forward(self, X, mask):
        batch_size, length, dim = X.shape
        # X.shape = batch_size x length x dim
        # mask.shape = batch_size x length
    
        X = self.encoder(X.reshape(-1, dim))  # Shape: batch_size x length x dim_output
        X = X.reshape(batch_size, length, dim)
        
        # Nullify vectors with mask
        X = X * mask.unsqueeze(-1)  # Shape: batch_size x length x dim
        
        # Mean over -2 dimension (length)
        mean_X = X.mean(dim=-2)  # Shape: batch_size x dim
        return self.decoder(mean_X)  # Shape: batch_size x dim


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X).squeeze(1)


class SetTransformer(nn.Module):
    def __init__(
        self,
        # dim_input: int,
        num_layers: int = 4,
        # num_outputs: int = 1,
        num_inds: int = 32,
        dim_hidden: int = 128,
        num_heads: int = 4,
        ln: bool = False
    ):
        super(SetTransformer, self).__init__()
        self.layers = [
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
            for _ in range(num_layers)
        ]
        
        # self.dec = nn.Sequential(
        #         PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        #         SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #         SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #         nn.Linear(dim_hidden, dim_output))

    def forward(self, X, mask):
        for layer in self.layers:
            X = layer(X)
        return X.mean(dim=-2)
        # return self.dec(self.enc(X))


class NestedSetsModel(nn.Module):
    def __init__(
        self,
        encoders: List[nn.Module],
        dim_hidden: int = 128,
        dim_output: int = 128,
        vocab_size: int = 100,
        embedding: Optional[nn.Module] = None,
    ):
        super(NestedSetsModel, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output

        self.embedding = embedding or nn.Embedding(vocab_size, dim_hidden)
        self.encoders = encoders
        self.mlp = nn.Linear(dim_hidden, dim_output)


    def forward(self, inputs):
        batch_size, lengths = inputs.shape[0], inputs.shape[1:]
        mask = (inputs != 0)  # Shape: batch_size x lengths[0] x ... x lengths[-1]

        x = self.embedding(inputs) * mask.unsqueeze(-1) # Shape: batch_size x length x k x dim_hidden
        for i, encoder in enumerate(self.encoders):
            x = x.reshape(batch_size * np.prod(lengths[:-(i+1)]).astype(int), lengths[-(i+1)], -1)
            # x.shape = (batch_size * lengths[0] * ... * lengths[-(i + 2)]) x lengths[-(i + 1)] x dim_hidden
            x = encoder(x, mask.reshape(-1, lengths[-(i + 1)]))
            mask = mask.all(-1)

        return self.mlp(x)
        
