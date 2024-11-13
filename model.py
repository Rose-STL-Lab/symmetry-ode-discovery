import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(self.shape)
    

class EncoderMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_norm, n_comps, activation, n_layers, latent_dim, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # Batch norm
            Reshape(-1, hidden_dim) if batch_norm and n_comps > 1 else nn.Identity(),
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            Reshape(-1, n_comps, hidden_dim) if batch_norm and n_comps > 1 else nn.Identity(),
            getattr(nn, activation)(*kwargs['activation_args']),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                # Batch norm
                Reshape(-1, hidden_dim) if batch_norm and n_comps > 1 else nn.Identity(),
                nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                Reshape(-1, n_comps, hidden_dim) if batch_norm and n_comps > 1 else nn.Identity(),
                getattr(nn, activation)(*kwargs['activation_args']),
            ) for _ in range(n_layers-1)],
            nn.Linear(hidden_dim, latent_dim) if not kwargs['ortho_ae'] else orthogonal(nn.Linear(hidden_dim, latent_dim)),
            Reshape(-1, latent_dim) if batch_norm and n_comps > 1 else nn.Identity(),
            nn.BatchNorm1d(latent_dim) if batch_norm else nn.Identity(),
            Reshape(-1, n_comps, latent_dim) if batch_norm and n_comps > 1 else nn.Identity(),
        )

    def forward(self, x):
        return self.layers(x)
    

class DecoderMLP(nn.Module):
    def __init__(self, latent_dim, hidden_dim, activation, n_layers, input_dim, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            getattr(nn, activation)(*kwargs['activation_args']),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                getattr(nn, activation)(*kwargs['activation_args']),
            ) for _ in range(n_layers-1)],
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.layers(x)
    

class SplitModel(nn.Module):
    def __init__(self, model_class, **kwargs):
        super().__init__()
        self.model1 = model_class(**kwargs)
        self.model2 = model_class(**kwargs)

    def forward(self, x):
        x1, x2 = torch.split(x, x.shape[-1]//2, dim=-1)
        return torch.cat([self.model1(x1), self.model2(x2)], dim=-1)
