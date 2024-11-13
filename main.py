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
from sindy import *
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
    if args['sindy_optimizer'] != 'lbfgs':
        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    else:
        data_size = int(len(train_dataset) * args['lbfgs_subsample'])
        train_loader = DataLoader(train_dataset, batch_size=data_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    # Initialize model
    autoencoder = AutoEncoder(**args).to(args['device'])
    discriminator = Discriminator(**args).to(args['device'])
    generator = LieGenerator(**args).to(args['device'])

    # Load model if needed
    laligan_path = args['load_laligan']
    if laligan_path is not None:
        autoencoder.load_state_dict(torch.load(f'saved_models/{laligan_path}/autoencoder.pt'))
        # discriminator.load_state_dict(torch.load(f'saved_models/{laligan_path}/discriminator.pt'))
        # generator.load_state_dict(torch.load(f'saved_models/{laligan_path}/generator.pt'))
        try:
            saved_state_dict = torch.load(f'saved_models/{laligan_path}/generator.pt')
            generator.load_state_dict(saved_state_dict)
        except:  # fix compatbility issue in loading generator
            new_state_dict = generator.state_dict()
            for name, param in new_state_dict.items():
                if name not in saved_state_dict:
                    saved_state_dict[name] = param
            generator.load_state_dict(saved_state_dict)
        # generator.set_threshold(args['gan_st_thres'])
        generator.masks = torch.load(f'saved_models/{laligan_path}/generator_mask.pt')
        generator.masks = [mask.to(args['device']) if mask is not None else None for mask in generator.masks]
        # print(generator.getLi())

    if args['fix_laligan']:
        for module in [autoencoder, generator, discriminator]:
            for param in module.parameters():
                param.requires_grad = False

    # Initialize regressor
    if args['eq_constraint']:
        L_list = generator.get_full_basis_list()
        repr_dim = L_list[0].shape[-1] // args['n_comps']
        L_trunc = [L[:repr_dim, :repr_dim].detach().cpu() for L in L_list]
        args['L_list'] = L_trunc
    regressor = SINDyRegression(**args).to(args['device'])
    # Distilled regressor
    if args['distill_latent']:
        # set eq_constraint and use_latent to False in new args
        args_distill = args.copy()
        args_distill['eq_constraint'] = False
        args_distill['use_latent'] = False
        args_distill['L_list'] = []
        regressor_dst = SINDyRegression(**args_distill).to(args['device'])
    else:
        regressor_dst = None

    # Train model
    if args['mt_data']:  # symmetry discovery on multi-timestep data
        train_fn = train_lassi
    elif args['sindy_optimizer'] == 'lbfgs':
        train_fn = train_SIGED_lbfgs
    else:
        train_fn = train_SIGED
    train_fn(
        autoencoder=autoencoder,
        discriminator=discriminator,
        generator=generator,
        regressor=regressor,
        regressor_dst=regressor_dst,
        train_loader=train_loader,
        test_loader=val_loader,
        **args
    )
    
    # Save final model
    if not os.path.exists(f'saved_models/{args["save_dir"]}'):
        os.makedirs(f'saved_models/{args["save_dir"]}')
    torch.save(autoencoder.state_dict(), f'saved_models/{args["save_dir"]}/autoencoder.pt')
    torch.save(discriminator.state_dict(), f'saved_models/{args["save_dir"]}/discriminator.pt')
    torch.save(generator.state_dict(), f'saved_models/{args["save_dir"]}/generator.pt')
    torch.save(generator.masks, f'saved_models/{args["save_dir"]}/generator_mask.pt')
    torch.save(regressor.state_dict(), f'saved_models/{args["save_dir"]}/regressor.pt')
    torch.save(regressor.L_list, f'saved_models/{args["save_dir"]}/regressor_lie_list.pt')
    if regressor_dst is not None:
        torch.save(regressor_dst.state_dict(), f'saved_models/{args["save_dir"]}/regressor.pt')

    # evaluation for equation discovery
    if not args['mt_data']:
        print('\n=== Evaluation ===\n')
        true_eq = sindy_truth[args['task']]
        regressor_eval = regressor_dst if args['distill_latent'] else regressor
        coef, cf, mse, cf_all, mse_all = eval_sindy_regressor(regressor_eval, true_eq)
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
