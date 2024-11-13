import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
import numpy as np
from model import *


class AutoEncoder(nn.Module):
    '''
    Arguments:
        input_dim: dimension of input
        hidden_dim: dimension of hidden layer
        latent_dim: dimension of latent layer
        n_layers: number of hidden layers
        n_comps: number of components
        activation: activation function
        flatten: whether to flatten input
    Input:
        x: (batch_size, n_comps, input_dim)
    Output:
        z: (batch_size, n_comps, latent_dim)
        xhat: (batch_size, n_comps, input_dim)
    '''
    def __init__(self, **kwargs):
        super().__init__()
        ae_arch = kwargs['ae_arch']
        input_dim = kwargs['input_dim']
        hidden_dim = kwargs['hidden_dim']
        latent_dim = kwargs['latent_dim']
        n_layers = kwargs['n_layers']
        n_comps = kwargs['n_comps']
        activation = kwargs['activation']
        batch_norm = kwargs['batch_norm']

        if ae_arch == 'mlp':
            self.encoder = nn.Sequential(
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
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                getattr(nn, activation)(*kwargs['activation_args']),
                *[nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    getattr(nn, activation)(*kwargs['activation_args']),
                ) for _ in range(n_layers-1)],
                nn.Linear(hidden_dim, input_dim),
            )

        elif ae_arch == 'mlp_split':
            self.encoder = SplitModel(EncoderMLP, **kwargs)
            self.decoder = SplitModel(DecoderMLP, **kwargs)

        elif ae_arch == 'stick_cnn':
            # input: (bs, n_comps (2), 3, 128, 256)
            self.model = FullSwingStickModel(n_comps, 3)
            self.encoder = self.model.encode
            self.decoder = self.model.decode

        elif ae_arch == 'stick_cnn_pretrain':
            self.model = FullSwingStickModel(n_comps, 3, refine=False)
            self.encoder = self.model.encode
            self.decoder = self.model.decode

        elif ae_arch == 'pendulum_cnn':
            self.model = FullPendulumModel(n_comps, 3)
            self.encoder = self.model.encode
            self.decoder = self.model.decode

        elif ae_arch == 'none':
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return z, xhat

    def decode(self, z):
        return self.decoder(z)
    
    def encode(self, x):
        return self.encoder(x)
    
    def compute_dz(self, x, dx):
        dz = jvp(self.encode, x, v=dx)[1]
        return dz
    
    def compute_dx(self, z, dz):
        dx = jvp(self.decode, z, v=dz)[1]
        return dx
    
    def iga(self, g, x, normalize_z=True):
        '''
        Compute the infinitesimal action of the Lie algebra element g on x.
        '''
        z = self.encode(x)
        if normalize_z:  # zero mean
            z = z - z.mean(dim=0, keepdim=True)
        reshape_flag = len(z.shape) > 2
        if reshape_flag:
            v_z = torch.einsum('jk, ...k->...j', g, z.reshape(z.shape[0], -1))
        else:
            v_z = torch.einsum('jk, ...k->...j', g, z)
        if reshape_flag:
            v_z = v_z.reshape(z.shape)
        # v_x = self.compute_dx(z, v_z)
        v_x = jvp(self.decode, z, v=v_z, create_graph=True, strict=True)[1]
        return v_x
    