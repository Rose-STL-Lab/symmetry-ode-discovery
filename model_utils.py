import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
from functools import partial
import numpy as np

def symmreg_i(x_fx, autoencoder, generator, f=None, dfdx=None, normalize='global', z_mean=None, relative=True, require_grad=False, numpy=False):
    '''
    Compute the infinitesimal symmetry regularization loss.
    x_fx: input and predicted output: (batch_size, 2, input_dim)
    autoencoder, generator: nn.Module representing the symmetry
    f: function represented by nn.Module to be symmetrized
    dfdx: alternatively, the Jacobian of f at x
    normalize: 'in_batch' or 'global', whether to normalize the latent vector
    z_mean: if normalize='global', the mean of the latent vectors; if None, use the final batch norm layer in encoder
    relative: whether to use relative loss (i.e. normalize by the scale of difference between x and fx)
    require_grad: whether to compute the gradient of the loss
    '''

    if numpy:
        x_fx = torch.from_numpy(x_fx).float().to(autoencoder.device)
        if z_mean is not None:
            z_mean = torch.from_numpy(z_mean).float().to(autoencoder.device)
        if require_grad:
            raise ValueError('Cannot require grad when numpy=True.')

    if f is None and dfdx is None:
        raise ValueError('Either f or dfdx must be specified.')
    if f is not None and dfdx is not None:
        raise ValueError('Only one of f and dfdx can be specified.')   
    jvp_fn = partial(jvp, create_graph=True, strict=True) if require_grad else jvp
    autoencoder.eval()
    generator.eval()

    with torch.set_grad_enabled(require_grad):
        loss = 0.0
        z = autoencoder.encode(x_fx)
        x_dim = x_fx.shape[-1]
        x, fx = x_fx[:, 0], x_fx[:, 1]

        if normalize == 'in_batch':
            z = z - z.mean(dim=0, keepdim=True)
        elif normalize == 'global':
            if z_mean is None:
                z_mean = autoencoder.encoder[-2].bias
            z = z - z_mean
        z_shape = z.shape

        for v in generator.get_full_basis_list():
            v_z = torch.einsum('jk,...k->...j', v, z.reshape(z_shape[0], -1))
            v_z = v_z.reshape(z_shape)
            v_x_fx = jvp_fn(autoencoder.decoder, z, v=v_z)[1]
            v_x, v_fx = v_x_fx[:, 0], v_x_fx[:, 1]
            if f is not None:
                input_variation = jvp_fn(f, x, v_x)[1]
            elif dfdx is not None:
                input_variation = torch.einsum('bjk,bk->bj', dfdx, v_x)
            if not relative:
                loss += torch.mean((input_variation - v_fx) ** 2)
            else:
                loss += torch.mean((input_variation - v_fx) ** 2) / torch.mean(input_variation ** 2)

    if numpy:
        loss = loss.cpu().numpy()

    return loss  # , input_variation, v_fx

def symmreg_f(x_fx, autoencoder, generator, f, normalize='global', z_mean=None, relative=True, require_grad=False, numpy=False):
    '''
    Compute the finite symmetry regularization loss.
    x_fx: input and predicted output: (batch_size, 2, input_dim)
    autoencoder, generator: nn.Module representing the symmetry
    f: function to be symmetrized
    normalize: 'in_batch' or 'global', whether to normalize the latent vector
    z_mean: if normalize='global', the mean of the latent vectors; if None, use the final batch norm layer in encoder
    relative: whether to use relative loss (i.e. normalize by the scale of difference between x and fx)
    require_grad: whether to compute the gradient of the loss
    '''

    autoencoder.eval()
    generator.eval()

    if numpy:
        x_fx = torch.from_numpy(x_fx).float().to(generator.Li[0].device)
        if z_mean is not None:
            z_mean = torch.from_numpy(z_mean).float().to(generator.Li[0].device)
        if require_grad:
            raise ValueError('Cannot require grad when numpy=True.')

    with torch.set_grad_enabled(require_grad):
        loss = 0.0
        z = autoencoder.encode(x_fx)
        x_dim = x_fx.shape[-1]
        x, fx = x_fx[:, 0], x_fx[:, 1]

        if normalize == 'in_batch':
            z = z - z.mean(dim=0, keepdim=True)
        elif normalize == 'global':
            if z_mean is None:
                z_mean = autoencoder.encoder[-2].bias
            z = z - z_mean
        z_shape = z.shape

        for g in generator.get_deterministic_group_elems():
            g_z = torch.einsum('jk,...k->...j', g, z.reshape(z_shape[0], -1))
            g_z = g_z.reshape(z_shape)
            g_z = g_z + z_mean
            g_x_fx = autoencoder.decode(g_z)
            g_x, g_fx = g_x_fx[:, 0], g_x_fx[:, 1]
            if numpy:
                g_x = g_x.cpu().numpy()
            f_g_x = f(g_x)
            if numpy:
                f_g_x = torch.from_numpy(f_g_x).float().to(generator.Li[0].device)
            if not relative:
                loss += torch.mean((f_g_x - g_fx) ** 2)
            else:
                loss += torch.mean((f_g_x - g_fx) ** 2) / torch.mean((f_g_x - fx) ** 2)

    if numpy:
        loss = loss.cpu().numpy()

    return loss

def symmreg_r(x, autoencoder, generator, h, normalize='global', z_mean=None, require_grad=False, scale=0.01):
    '''
    Compute the reversed symmetry regularization loss.
    x_fx: input and predicted output: (batch_size, 2, input_dim)
    autoencoder, generator: nn.Module representing the symmetry
    h: ODE to be symmetrized
    normalize: 'in_batch' or 'global', whether to normalize the latent vector
    z_mean: if normalize='global', the mean of the latent vectors; if None, use the final batch norm layer in encoder
    require_grad: whether to compute the gradient of the loss
    '''

    jvp_fn = partial(jvp, create_graph=True, strict=True) if require_grad else jvp
    autoencoder.eval()
    generator.eval()

    g_list = generator.get_deterministic_group_elems(scale=scale)
    n_group_elems = len(g_list)

    def group_transform(x, g_idx=0, normalize='global', z_mean=None):
        xx = torch.stack([x, x], dim=1)
        z = autoencoder.encode(xx)
        if normalize == 'in_batch':
            z = z - z.mean(dim=0, keepdim=True)
        elif normalize == 'global':
            if z_mean is None:
                z_mean = autoencoder.encoder[-2].bias
            z = z - z_mean
        z_shape = z.shape
        g_z = torch.einsum('jk,...k->...j', g_list[g_idx], z.reshape(z_shape[0], -1))
        g_z = g_z.reshape(z_shape)
        g_z = g_z + z_mean
        g_xx = autoencoder.decode(g_z)
        return g_xx[:, 0]

    with torch.set_grad_enabled(require_grad):
        loss = 0.0
        for i in range(n_group_elems):
            group_transform_ith = partial(group_transform, g_idx=i, normalize=normalize, z_mean=z_mean)
            gx = group_transform_ith(x)
            hx = h(x)
            variation1 = jvp_fn(group_transform_ith, x, v=hx)[1]
            variation2 = h(gx)
            loss += torch.mean((variation1 - variation2) ** 2)

    return loss

def precompute_symmreg_r(x, autoencoder, generator, z_mean=None, scale=0.01):
    '''
    Precompute the group transformation g(x) and its Jacobian J_g(x) for reversed symmetry regularization loss.
    This decouples the group transformation from the ODE, allowing for integration with PySR.
    '''
    from torch.func import jacrev, jacfwd, vmap  # pytorch beta feature

    autoencoder.eval()
    generator.eval()

    g_list = generator.get_deterministic_group_elems(scale=scale)
    n_group_elems = len(g_list)

    def group_transform(x, g_idx=0, normalize='global', z_mean=None):
        xx = torch.stack([x, x], dim=1)
        z = autoencoder.encode(xx)
        if normalize == 'in_batch':
            z = z - z.mean(dim=0, keepdim=True)
        elif normalize == 'global':
            if z_mean is None:
                z_mean = autoencoder.encoder[-2].bias
            z = z - z_mean
        z_shape = z.shape
        g_z = torch.einsum('jk,...k->...j', g_list[g_idx], z.reshape(z_shape[0], -1))
        g_z = g_z.reshape(z_shape)
        g_z = g_z + z_mean
        g_xx = autoencoder.decode(g_z)
        return g_xx[:, 0]
    
    with torch.no_grad():
        gx_list = []
        Jgx_list = []
        for i in range(n_group_elems):
            group_transform_ith = partial(group_transform, g_idx=i, normalize='global', z_mean=z_mean)
            gx = group_transform_ith(x)
            gx_list.append(gx)
            Jgx = vmap(jacfwd(group_transform_ith))(x)
            Jgx_list.append(Jgx)

    return gx_list, Jgx_list


make_symmreg = lambda autoencoder, generator: partial(symmreg_i, autoencoder=autoencoder, generator=generator)
make_symmreg_pttrain = lambda autoencoder, generator: partial(symmreg_i, autoencoder=autoencoder, generator=generator, require_grad=True)
make_symmreg_np = lambda autoencoder, generator: partial(symmreg_i, autoencoder=autoencoder, generator=generator, numpy=True)
make_fsymmreg = lambda autoencoder, generator: partial(symmreg_f, autoencoder=autoencoder, generator=generator)
make_fsymmreg_pttrain = lambda autoencoder, generator: partial(symmreg_f, autoencoder=autoencoder, generator=generator, require_grad=True)
make_fsymmreg_np = lambda autoencoder, generator: partial(symmreg_f, autoencoder=autoencoder, generator=generator, numpy=True)
make_rsymmreg = lambda autoencoder, generator: partial(symmreg_r, autoencoder=autoencoder, generator=generator)
make_rsymmreg_pttrain = lambda autoencoder, generator: partial(symmreg_r, autoencoder=autoencoder, generator=generator, require_grad=True)

def odeint(f, x0, t, dt, method='euler', full_traj=False):
    '''
    Integrate an ODE f over a time interval differentiably.
    f: a PyTorch nn.Module representing the ODE
    x0: initial state
    t: time
    dt: timestep
    method: 'euler' or 'rk4'
    full_traj: whether to return the full trajectory
    '''
    n_steps = int(t / dt)
    if full_traj:
        traj = []
    if method == 'euler':
        for i in range(n_steps):
            x0 = x0 + dt * f(x0)
            if full_traj:
                traj.append(x0)
    elif method == 'rk4':
        for i in range(n_steps):
            k1 = f(x0)
            k2 = f(x0 + dt / 2 * k1)
            k3 = f(x0 + dt / 2 * k2)
            k4 = f(x0 + dt * k3)
            x0 = x0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if full_traj:
                traj.append(x0)
    else:
        raise ValueError('Unrecognized ODEInt method.')
    if full_traj:
        return torch.stack(traj, dim=0)
    else:
        return x0
