import numpy as np
import torch
from tqdm import tqdm, trange
from functools import partial
import argparse
from data_utils.ode import *


# random initial conditions for growth system
def generate_random_ics(n_ics=1000, **kwargs):
    initial_conditions = []
    for _ in range(n_ics):
        x0 = np.random.uniform(0.2, 1, size=2)
        initial_conditions.append(x0)
    return np.array(initial_conditions)

# derivative for growth system
def growth(x, a=0.1, b=0.3, **kwargs):
    dx = np.zeros_like(x)
    dx[..., 0] = a * x[..., 1] ** 2 - b * x[..., 0]
    dx[..., 1] = x[..., 1]
    return dx

get_growth_data = partial(gen_data, ode=growth, init_fn=generate_random_ics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_ics', type=int, default=100)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--subsample_rate', type=int, default=10)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--noise', type=float, default=0.2)
    # parser.add_argument('--multiplicative_noise', action='store_true')
    parser.add_argument('--smoothing', type=str, default=None)
    parser.add_argument('--gp_sigma_in', type=float, default=0.05)
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--save_name', type=str, default='train')
    args = parser.parse_args()

    x, dx = get_growth_data(
        n_ics=args.n_ics, num_steps=args.num_steps, subsample_rate=args.subsample_rate,
        dt=args.dt, noise=args.noise, multiplicative_noise=True, smoothing=args.smoothing, gp_sigma_in=args.gp_sigma_in
    )

    # np.save(f'{args.save_dir}/selkov-{args.save_name}-x.npy', x)
    # np.save(f'{args.save_dir}/selkov-{args.save_name}-dx.npy', dx)
    x = torch.from_numpy(x).to(torch.float32)
    dx = torch.from_numpy(dx).to(torch.float32)
    smoothing_str = f'-{args.smoothing}' if args.smoothing is not None else ''
    torch.save(x, f'{args.save_dir}/growth-{args.save_name}-noise{int(100*args.noise):02d}{smoothing_str}-x.pt')
    torch.save(dx, f'{args.save_dir}/growth-{args.save_name}-noise{int(100*args.noise):02d}{smoothing_str}-dx.pt')
