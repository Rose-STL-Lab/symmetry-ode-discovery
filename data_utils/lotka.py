import numpy as np
import torch
from tqdm import tqdm, trange
from functools import partial
from data_utils.ode import *
import argparse


# random initial conditions for Lotka-Volterra equation
def generate_random_ics(n_ics=10000, h_min=3, h_max=4.5, canonical=True):
    initial_conditions = []
    for _ in range(n_ics):
        x0 = np.random.uniform(0, 1, size=2)
        if canonical:
            x0 = np.log(x0)
        h = H_lv(x0, canonical=canonical)
        while h < h_min or h > h_max:
            x0 = np.random.uniform(0, 1, size=2)
            if canonical:
                x0 = np.log(x0)
            h = H_lv(x0, canonical=canonical)
        initial_conditions.append(x0)
    return np.array(initial_conditions)

# Hamiltonian for Lotka-Volterra equation
def H_lv(x, a=2/3, b=4/3, c=1.0, d=1.0, canonical=True):
    if canonical:
        return c * np.exp(x[..., 0]) - d * x[..., 0] + b * np.exp(x[..., 1]) - a * x[..., 1]
    else:
        return c * x[..., 0] - d * np.log(x[..., 0]) + b * x[..., 1] - a * np.log(x[..., 1])

# derivative for Lotka-Volterra equation
def lotka_volterra(x, a=2/3, b=4/3, c=1.0, d=1.0, canonical=True, **kwargs):
    dx = np.zeros_like(x)
    if not canonical:
        dx[..., 0] = a * x[..., 0] - b * x[..., 0] * x[..., 1]
        dx[..., 1] = c * x[..., 0] * x[..., 1] - d * x[..., 1]
    else:
        dx[..., 0] = a - b * np.exp(x[..., 1])
        dx[..., 1] = c * np.exp(x[..., 0]) - d
    return dx

get_lv_data = partial(gen_data, ode=lotka_volterra, init_fn=generate_random_ics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_ics', type=int, default=200)
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--smoothing', type=str, default=None)
    parser.add_argument('--gp_sigma_in', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--save_name', type=str, default='train')
    args = parser.parse_args()

    x, dx = get_lv_data(n_ics=args.n_ics, num_steps=args.num_steps, noise=args.noise, smoothing=args.smoothing, gp_sigma_in=args.gp_sigma_in)

    # np.save(f'{args.save_dir}/lv-{args.save_name}-x.npy', x)
    # np.save(f'{args.save_dir}/lv-{args.save_name}-dx.npy', dx)
    x = torch.from_numpy(x).to(torch.float32)
    dx = torch.from_numpy(dx).to(torch.float32)
    smoothing_str = f'-{args.smoothing}' if args.smoothing is not None else ''
    torch.save(x, f'{args.save_dir}/lv-{args.save_name}-noise{int(100*args.noise):02d}{smoothing_str}-x.pt')
    torch.save(dx, f'{args.save_dir}/lv-{args.save_name}-noise{int(100*args.noise):02d}{smoothing_str}-dx.pt')
