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
