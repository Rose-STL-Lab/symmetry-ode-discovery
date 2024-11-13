import numpy as np
from tqdm import tqdm, trange
from functools import partial
from data_utils.smoothing import num_diff_gp


def solve_ode_batch(ode, x0, dt=0.002, num_steps=2000, solver='rk4', **kwargs):
    x = np.zeros((num_steps, *x0.shape))
    dx = np.zeros_like(x)
    x[0] = x0
    update_fn = partial(ode, **kwargs)
    if solver == 'rk4':
        for i in trange(num_steps):
            dx1 = update_fn(x[i])
            dx[i] = dx1
            if i == num_steps - 1:
                break
            k1 = dt * dx1
            dx2 = update_fn(x[i] + 0.5 * k1)
            k2 = dt * dx2
            dx3 = update_fn(x[i] + 0.5 * k2)
            k3 = dt * dx3
            dx4 = update_fn(x[i] + k3)
            k4 = dt * dx4
            x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    else:
        raise NotImplementedError
    return x, dx

def gen_data(ode, init_fn, n_ics=1000, dt=0.002, num_steps=2000, subsample_rate=1,  noise=0.0, multiplicative_noise=False, smoothing=None, **kwargs):
    x0 = init_fn(n_ics)
    x, dx = solve_ode_batch(ode, x0, dt=dt, num_steps=num_steps, **kwargs)
    if noise > 0:
        # compute std for each dimension
        x_std = np.std(x, axis=(0, 1))
        if not multiplicative_noise:
            x += np.random.randn(*x.shape) * noise * x_std
        else:
            x *= (1 + np.random.randn(*x.shape) * noise)
        if smoothing is None:
            dx[:-1, :] = np.diff(x, axis=0) / dt
        elif smoothing == 'gp':
            print('Smoothing with Gaussian process...')
            dx, x = num_diff_gp(x, dt, noise_level=noise, std_base=x_std, sigma_in=kwargs['gp_sigma_in'])
    x = x[::subsample_rate]
    dx = dx[::subsample_rate]
    x = np.transpose(x, (1, 0, 2))  # (n_ics, num_steps, dim)
    dx = np.transpose(dx, (1, 0, 2))  # (n_ics, num_steps, dim)
    return x, dx
