import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from utils import *


class IntParameter(nn.Module):
    def __init__(self, k=2, noise=0.1):
        super(IntParameter, self).__init__()
        self.noise = noise
        self.k = k
    
    def forward(self, data):
        noise = torch.randn_like(data) * self.noise
        return torch.round(torch.clamp(self.k * (data + noise), -self.k - 0.49, self.k + 0.49))


class LieGenerator(nn.Module):
    def __init__(self, **kwargs):
        super(LieGenerator, self).__init__()
        self.repr = kwargs['repr']
        group_idx = kwargs['group_idx']
        self.uniform_max = kwargs['uniform_max']
        self.coef_dist = kwargs['coef_dist']
        self.g_init = kwargs['g_init']
        self.task = kwargs['task']
        self.sigma_init = kwargs['sigma_init']
        self.int_param = kwargs['int_param']
        self.int_param_noise = kwargs['int_param_noise']
        self.int_param_max = kwargs['int_param_max']
        self.threshold = kwargs['gan_st_thres']
        self.keep_center = kwargs['keep_center']
        self.activated_channel = None  # default to all channel
        self.construct_group_representation(self.repr, group_idx)
        self.masks = [mask.to(kwargs['device']) if mask is not None else None for mask in self.masks]
        self.int_param_approx = IntParameter(k=self.int_param_max, noise=self.int_param_noise)

    def construct_group_representation(self, repr_str, group_idx):
        # analyze the repr string
        repr = []
        tuple_list = repr_str.split('+')
        for t in tuple_list:
            t = t.strip()
            if t.startswith('(') and t.endswith(')'):
                elements = t[1:-1].split(',')
                elements = [e.strip() for e in elements]
                repr.append(tuple(elements))
        self.group_idx = group_idx.split(',')
        if len(self.group_idx) != len(repr):
            raise ValueError('Number of group indices does not match number of components in representation string.')
        group_idx_dict = {}
        for i, idx in enumerate(self.group_idx):
            if idx not in group_idx_dict:
                group_idx_dict[idx] = []
            group_idx_dict[idx].append(i)

        # repr is a tuple of 
        # either (N1, N2, N3) indicating N1 N3-dim vectors acted on by N2-dim Lie group,
        # or (N1, STR) specifying the group, or (N1,) indicating N1 scalars
        self.Li = nn.ParameterList()
        self.sigma = nn.ParameterList()
        self.struct_const = nn.ParameterList()
        self.masks = []  # mask for sequential thresholding
        self.n_comps = []
        self.n_channels = []
        self.learnable = []
        self.f_Li = []
        self.n_dims = 0
        for i, r in enumerate(repr):
            if len(r) >= 3:
                if len(r) == 3:
                    n_comps, n_channels, n_dims = r
                    self.f_Li.append(lambda Li: Li)
                else:  # len=4
                    n_comps, n_channels, n_dims, group_str = r
                    if group_str == 'o':
                        self.f_Li.append(lambda Li: Li - torch.transpose(Li, -1, -2))
                    else:
                        raise ValueError(f'Group {group_str} not implemented yet.')
                n_comps, n_channels, n_dims = int(n_comps), int(n_channels), int(n_dims)
                Li = nn.Parameter(torch.randn(n_channels, n_dims, n_dims))
                struct_const = nn.Parameter(torch.zeros(n_channels, n_channels, n_channels))
                mask = torch.ones_like(Li)
                self.Li.append(Li)
                self.struct_const.append(struct_const)
                self.masks.append(mask)
                self.n_comps.append(n_comps)
                self.n_channels.append(n_channels)
                self.learnable.append(True)
                self.n_dims += n_dims * n_comps
                self.sigma.append(nn.Parameter(torch.eye(n_channels, n_channels) * self.sigma_init, requires_grad=False))
            elif len(r) == 1:
                n_comps = int(r[0])
                self.Li.append(nn.Parameter(torch.zeros(1, n_comps, n_comps), requires_grad=False))
                self.struct_const.append(nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False))
                self.masks.append(None)
                self.n_comps.append(1)
                self.n_channels.append(1)
                self.learnable.append(False)
                self.n_dims += n_comps
                self.sigma.append(nn.Parameter(torch.eye(1, 1)))
                self.f_Li.append(lambda Li: Li)
            elif len(r) == 2:
                n_comps, group_str = r
                n_comps = int(n_comps)
                self.masks.append(None)
                self.f_Li.append(lambda Li: Li)
                if group_str == 'so2':
                    self.Li.append(nn.Parameter(torch.FloatTensor([[[0.0, 1.0], [-1.0, 0.0]]]), requires_grad=False))
                    self.struct_const.append(nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False))
                    self.n_comps.append(n_comps)
                    self.n_channels.append(1)
                    self.learnable.append(False)
                    self.n_dims += n_comps * 2
                    self.sigma.append(nn.Parameter(torch.eye(1, 1) * self.sigma_init, requires_grad=False))
                elif group_str == 'sim2':
                    self.Li.append(nn.Parameter(torch.FloatTensor([[[-0.2, 1.0], [-1.0, 0.0]]]), requires_grad=False))
                    self.struct_const.append(nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False))
                    self.n_comps.append(n_comps)
                    self.n_channels.append(1)
                    self.learnable.append(False)
                    self.n_dims += n_comps * 2
                    self.sigma.append(nn.Parameter(torch.eye(1, 1) * self.sigma_init, requires_grad=False))
                elif group_str == 'scaling2':
                    self.Li.append(nn.Parameter(torch.FloatTensor([[[2.0, 0.0], [0.0, 1.0]]]), requires_grad=False))
                    self.struct_const.append(nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False))
                    self.n_comps.append(n_comps)
                    self.n_channels.append(1)
                    self.learnable.append(False)
                    self.n_dims += n_comps * 2
                    self.sigma.append(nn.Parameter(torch.eye(1, 1) * self.sigma_init, requires_grad=False))
                elif group_str == 'so2*r':
                    Li = nn.Parameter(torch.FloatTensor([[[0.0, 1.0], [-1.0, 0.0]], [[0.1, 0.0], [0.0, 0.1]]]), requires_grad=False)
                    self.Li.append(Li)
                    self.struct_const.append(nn.Parameter(torch.zeros(2, 2, 2), requires_grad=False))
                    self.n_comps.append(n_comps)
                    self.n_channels.append(2)
                    self.learnable.append(False)
                    self.n_dims += n_comps * 2
                    self.sigma.append(nn.Parameter(torch.eye(2, 2) * self.sigma_init, requires_grad=False))
                elif group_str == 'so3':
                    self.Li.append(nn.Parameter(so(3), requires_grad=False))
                    self.struct_const.append(nn.Parameter(torch.zeros(3, 3, 3), requires_grad=False))
                    self.n_comps.append(n_comps)
                    self.n_channels.append(3)
                    self.learnable.append(False)
                    self.n_dims += n_comps * 3
                    self.sigma.append(nn.Parameter(torch.eye(3, 3) * self.sigma_init, requires_grad=False))
                elif group_str == 'so3+1':
                    L = torch.zeros(3, 4, 4)
                    L[:, :3, :3] = so(3)
                    self.Li.append(nn.Parameter(L, requires_grad=False))
                    self.struct_const.append(nn.Parameter(torch.zeros(3, 3, 3), requires_grad=False))
                    self.n_comps.append(n_comps)
                    self.n_channels.append(3)
                    self.learnable.append(False)
                    self.n_dims += n_comps * 4
                    self.sigma.append(nn.Parameter(torch.eye(3, 3) * self.sigma_init, requires_grad=False))
                elif group_str == 'so4':
                    self.Li.append(nn.Parameter(so(4), requires_grad=False))
                    self.struct_const.append(nn.Parameter(torch.zeros(6, 6, 6), requires_grad=False))
                    self.n_comps.append(n_comps)
                    self.n_channels.append(6)
                    self.learnable.append(False)
                    self.n_dims += n_comps * 4
                    self.sigma.append(nn.Parameter(torch.eye(6, 6) * self.sigma_init, requires_grad=False))
                else:
                    raise ValueError(f'Group {group_str} not implemented yet.')
            else:
                raise ValueError(f'Invalid representation string at position {i}: {r}')
        
        # Check if the group indices are valid
        for k, v in group_idx_dict.items():
            n_ch = self.n_channels[v[0]]
            for i in v:
                if self.n_channels[i] != n_ch:
                    raise ValueError(f'Group index {k} contains channels of different dimensions.')
                
        # Complete
        # print(f'Constructed Lie group representation with {self.n_dims} latent dimensions.')
        # print(self.getLi())

    def set_activated_channel(self, ch):
        self.activated_channel = ch

    def activate_all_channels(self):
        self.activated_channel = None

    # def channel_corr(self):
    #     s = 0.0
    #     for Li in self.Li:
    #         norm = torch.einsum('kdf,kdf->k', Li, Li)
    #         Li_N = Li / (torch.sqrt(norm).unsqueeze(-1).unsqueeze(-1) + 1e-6)
    #         s += torch.sum(torch.abs(torch.triu(torch.einsum('bij,cij->bc', Li_N, Li_N), diagonal=1)))
    #     return s
    
    def reg_norm(self):
        s = 0.0
        for Li, f, mask, learnable in zip(self.Li, self.f_Li, self.masks, self.learnable):
            if learnable:
                s += torch.sum(torch.clamp(0.5 - torch.einsum('kdf,kdf->k', f(Li) * mask, f(Li) * mask), min=0.0))
        return s
    
    def reg_ortho(self):
        s = 0.0
        for Li, f, mask, learnable in zip(self.Li, self.f_Li, self.masks, self.learnable):
            Li_m = f(Li) * mask
            Li_m_norm = torch.einsum('kdf,kdf->k', Li_m, Li_m)
            Li_m = Li_m / (torch.sqrt(Li_m_norm).unsqueeze(-1).unsqueeze(-1) + 1e-6)
            if learnable:
                s += torch.sum(torch.square(torch.triu(torch.einsum('bij,cij->bc', Li_m, Li_m), diagonal=1)))
        return s
    
    def reg_closure(self):
        s = 0.0
        for Li, f, c, mask, learnable in zip(self.Li, self.f_Li, self.struct_const, self.masks, self.learnable):
            if not learnable:
                continue
            n_dims = Li.shape[0]
            Li_m = f(Li) * mask
            Li_m_norm = torch.einsum('kdf,kdf->k', Li_m, Li_m)
            Li_m = Li_m / (torch.sqrt(Li_m_norm).unsqueeze(-1).unsqueeze(-1) + 1e-6)
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    commutator = Li_m[i] @ Li_m[j] - Li_m[j] @ Li_m[i]
                    s += torch.sum(torch.square(commutator - torch.einsum('k,kij->ij', c[i, j], Li_m)))
        return s

    def forward(self, x):  # random transformation on x
        # x: (batch_size, *, n_dims)
        # normalize x to have zero mean
        if not self.keep_center:
            x_mean = torch.mean(x, dim=list(range(len(x.shape)-1)), keepdim=True)
            x = x - x_mean
        batch_size = x.shape[0]
        output_shape = x.shape
        if len(x.shape) == 3:
            x = x.reshape(batch_size, -1)
        # z = self.sample_coefficient(batch_size, x.device)
        # g_z = torch.matrix_exp(torch.einsum('bj,jkl->bkl', z, self.getLi()))
        g_z = self.sample_group_element(batch_size, x.device)
        x_t = torch.einsum('bij,bj->bi', g_z, x)
        x_t = x_t.reshape(output_shape)
        if not self.keep_center:
            x_t = x_t + x_mean
        return x_t
    
    def infinitesimal_transform(self, x, L_idx):
        '''
        x: (batch_size, *, n_dims)
        L_idx: index of the Lie algebra basis to use
        Compute the infinitesimal change of x by Lie algebra element.
        '''
        # normalize x to have zero mean
        if not self.keep_center:
            x_mean = torch.mean(x, dim=list(range(len(x.shape)-1)), keepdim=True)
            x = x - x_mean
        batch_size = x.shape[0]
        output_shape = x.shape
        if len(x.shape) == 3:
            x = x.reshape(batch_size, -1)
        L = self.get_full_basis_list()[L_idx]
        L_x = torch.einsum('ij,bj->bi', L, x)
        L_x = L_x.reshape(output_shape)
        return L_x

    def set_threshold(self, threshold):
        # relative to max in each channel
        for Li, f, mask in zip(self.Li, self.f_Li, self.masks):
            if mask is None:
                continue
            max_chval = torch.amax(torch.abs(f(Li)), dim=(1, 2), keepdim=True)
            mask.data = torch.logical_and(torch.abs(f(Li)) > threshold * max_chval, mask).float()
            # mask.data = (torch.abs(f(Li)) > threshold * max_chval).float()
    
    def sample_group_element(self, batch_size, device):
        start_dim = 0
        g = []
        z_dict = {}

        # only sample one z for each group as specified in group_idx
        for i, idx in enumerate(self.group_idx):
            if idx not in z_dict:
                z_dict[idx] = self.sample_coefficient(batch_size, self.n_channels[i], self.sigma[i], device)

        # compute group element
        for Li, f, group_idx, mask, n_comps, learnable in zip(self.Li, self.f_Li, self.group_idx, self.masks, self.n_comps, self.learnable):
            if learnable and self.int_param:
                Li = self.int_param_approx(f(Li))
            if learnable and mask is not None:
                Li = f(Li) * mask
            # z = self.sample_coefficient(batch_size, n_channels, sigma, device)
            z = z_dict[group_idx]
            g_z = torch.matrix_exp(torch.einsum('bj,jkl->bkl', z, Li))
            for _ in range(n_comps):
                end_dim = start_dim + g_z.shape[1]
                g_z_padded = F.pad(g_z, (start_dim, self.n_dims - end_dim, start_dim, self.n_dims - end_dim))
                g.append(g_z_padded)
                start_dim = end_dim
        g = torch.stack(g, dim=1)
        g = torch.sum(g, dim=1)
        return g
    
    def get_full_basis_list(self, split_channel=True):
        start_dim = 0
        v = []
        group_idx_dict = {}
        for i, idx in enumerate(self.group_idx):
            if idx not in group_idx_dict:
                group_idx_dict[idx] = []
        for Li, f, group_idx, mask, n_comps, learnable in zip(self.Li, self.f_Li, self.group_idx, self.masks, self.n_comps, self.learnable):
            if learnable and mask is not None:
                Li = f(Li) * mask
            v_comp = []
            for _ in range(n_comps):
                end_dim = start_dim + Li.shape[1]
                v_padded = F.pad(Li, (start_dim, self.n_dims - end_dim, start_dim, self.n_dims - end_dim))
                v_comp.append(v_padded)
                start_dim = end_dim
            v_comp = torch.stack(v_comp, dim=1)
            v_comp = torch.sum(v_comp, dim=1)
            group_idx_dict[group_idx].append(v_comp)
        for idx in group_idx_dict.keys():
            if split_channel:
                v += [ch for ch in sum(group_idx_dict[idx])]
            else:
                v.append(sum(group_idx_dict[idx]))
        return v
    
    def get_deterministic_group_elems(self, split_channel=False, scale=1.0):
        '''
        Return a list of group elements with deterministic coefficients.
        Used for exporting the model as a finite symmetry regularizer.
        '''
        lie_basis_list = self.get_full_basis_list(split_channel=split_channel)
        g_list = []
        for sigma, L in zip(self.sigma, lie_basis_list):
            if len(L.shape) == 3:
                Li_split = [Li for Li in L]
                for Li in Li_split:
                    g_z = torch.matrix_exp(sigma * Li * scale)
                    g_list.append(g_z)
            else:
                g_z = torch.matrix_exp(sigma * L * scale)
                g_list.append(g_z)
        return g_list

    def sample_coefficient(self, batch_size, n_channels, params, device):
        if self.coef_dist == 'normal':
            sigma = params
            z = torch.randn(batch_size, n_channels, device=device) @ sigma
        elif self.coef_dist == 'uniform':
            uniform_max = params
            z = torch.rand(batch_size, n_channels, device=device) * 2 * uniform_max - uniform_max
        elif self.coef_dist == 'uniform_int_grid':
            uniform_max = params
            z = torch.randint(-int(uniform_max), int(uniform_max), (batch_size, n_channels), device=device, dtype=torch.float32)
        ch = self.activated_channel
        if ch is not None:  # leaving only specified columns
            mask = torch.zeros_like(z, device=z.device)
            mask[:, ch] = 1
            z = z * mask
        return z
    
    def transform(self, g_z, x, tp):
        return torch.einsum('bjk,bk->bj', g_z, x)
        # if tp == 'vector':
        #     return torch.einsum('bjk,btk->btj', g_z, x)
        # elif tp == 'scalar':
        #     return x
        # elif tp == 'grid':
        #     grid = F.affine_grid(g_z[:, :-1], x.shape)
        #     return F.grid_sample(x, grid)

    def getLi(self):
        return self.get_full_basis_list(split_channel=False)
        # convert ParameterList to list of tensors
        # return [self.int_param_approx(Li) if self.int_param and learnable
        #         else f(Li) * mask if learnable else f(Li)
        #         for Li, f, mask, learnable in zip(self.Li, self.f_Li, self.masks, self.learnable)]
    
    def getStructureConst(self):
        return [c.reshape(-1, c.shape[-1]) for c, learnable in zip(self.struct_const, self.learnable) if learnable]


class Discriminator(nn.Module):
    def __init__(self, latent_dim, n_comps, hidden_dim, n_layers, activation='ReLU', **kwargs):
        super(Discriminator, self).__init__()
        self.input_dim = latent_dim * n_comps
        if kwargs['use_original_x']:
            self.input_dim += kwargs['input_dim'] * n_comps
        if kwargs['use_invariant_y']:
            if kwargs['embed_y']:
                self.y_embedding = nn.Embedding(kwargs['y_classes'], kwargs['y_embed_dim'])
                self.input_dim += kwargs['y_embed_dim']
            else:
                self.input_dim += kwargs['y_dim']
            self.embed_y = kwargs['embed_y']
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            getattr(nn, activation)(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                getattr(nn, activation)(),
            ) for _ in range(n_layers-1)],
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )


    def forward(self, z, y=None, x=None):
        # z: latent representation; y: invariant label; x: original input
        z = z.reshape(z.shape[0], -1)
        if y is not None:
            if self.embed_y:
                y = self.y_embedding(y)
            z = torch.cat([z, y], dim=-1)
        if x is not None:
            x = x.reshape(x.shape[0], -1)
            z = torch.cat([z, x], dim=-1)
        validity = self.model(z)
        return validity
