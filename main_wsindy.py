import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import argparse
from functools import partial
from torch.utils.data import DataLoader
from gan import *
from autoencoder import *
from sindy import SINDyRegression, WSINDyWrapper
from train import *
from evaluation.eval_eq import eval_sindy_regressor, sindy_truth
from dataset import get_dataset
from parser_utils import get_args


if __name__ == '__main__':
    args = get_args()
    
    # Initialize wandb
    wandb.init(project='anonym', entity='anonym', name=args.wandb_name, config=args)

    # Set random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # args to dict
    args = vars(args)

    # Load dataset
    train_dataset, val_dataset, args = get_dataset(args)
    n_ics, n_steps = train_dataset.n_ics, train_dataset.n_steps
    train_x = train_dataset.x.reshape(n_ics, n_steps, -1)
    # randomly sample a sub-trajectory w/ 80% length
    rnd_start_step = np.random.randint(0, n_steps - int(0.8 * n_steps))
    rnd_traj_idx = np.random.randint(0, n_ics)
    train_x = train_x[rnd_traj_idx, rnd_start_step:rnd_start_step + int(0.8 * n_steps)]
    n_steps = int(0.8 * n_steps)
    dt = ode_dt_dict[args['task']]
    t = torch.arange(n_steps) * dt
    t_max = n_steps * dt

    regressor = SINDyRegression(**args).to(args['device'])
    wsindy_wrapper = WSINDyWrapper(regressor, t, t_max, **args)

    # Solve weak SINDy
    train_WSINDy(
        wrapper=wsindy_wrapper,
        train_x=train_x,
        **args
    )
    
    # Save final model
    if not os.path.exists(f'saved_models/{args["save_dir"]}'):
        os.makedirs(f'saved_models/{args["save_dir"]}')
    torch.save(regressor.state_dict(), f'saved_models/{args["save_dir"]}/regressor.pt')

    # evaluation for equation discovery
    if not args['mt_data']:
        print('\n=== Evaluation ===\n')
        true_eq = sindy_truth[args['task']]
        coef, cf, mse, cf_all, mse_all = eval_sindy_regressor(regressor, true_eq)
        print(f'Correct form: {cf}')
        print(f'MSE: {np.where(cf, mse, 0.0)}')
        print(f'MSE (any): {mse}')
        eval_results = {
            'coefficients': coef,
            'correct_form': cf,
            'mse': mse,
            'correct_form_all': cf_all,
            'mse_all': mse_all,
        }
        eval_save_dir = f'eval_results/{args["save_dir"]}'
        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir)
        np.savez(f'{eval_save_dir}/seed{seed}.npz', **eval_results)
    
    wandb.finish()
