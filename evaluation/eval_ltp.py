import numpy as np
import sympy as sp
import torch
import os
from model_utils import odeint
from dataset import ode_dt_dict


@torch.no_grad()
def eval_ltp_accuracy(regressor, autoencoder, x, dt=None, **kwargs):
    '''
    Long-term prediction using the learned dynamics.
    x: (n_ics, n_steps, n_dim)
    '''

    x0 = x[:, 0]
    x_shape = x.shape
    if len(x.shape) == 3:
        n_ics, n_steps, n_dim = x.shape
    else:
        n_ics, n_steps, _, n_dim = x.shape
    n_steps -= 1
    task = kwargs['task'].split('_')[-1]
    if dt is None:
        dt = ode_dt_dict[task]
    t_max = n_steps * dt
    
    if autoencoder is not None:
        z0 = autoencoder.encode(x0)
        if len(z0.shape) == 3:
            z0 = z0.flatten(0, 1)
        z_pred = odeint(regressor, z0, t_max, dt, method='rk4', full_traj=True)
        z_pred = z_pred.transpose(0, 1)  # (n_ics, n_steps, n_dim)
        x_pred = autoencoder.decode(z_pred.flatten(0, 1)).reshape(n_ics, n_steps, n_dim)
    else:
        x_pred = odeint(regressor, x0, t_max, dt, method='rk4', full_traj=True)
        x_pred = x_pred.transpose(0, 1)

    error = torch.mean((x[:, 1:] - x_pred) ** 2, dim=-1)
    res = {
        'x_pred': x_pred,
        't': torch.arange(1, n_steps+1) * dt,
        'error': error,
    }
    return { k: v.cpu().numpy() for k, v in res.items() }
