import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import argparse
from torch.utils.data import DataLoader
from gan import *
from autoencoder import *
from train import *
from dataset import *
from sindy import *
from parser_utils import get_sindy_args
from utils import get_dataset

if __name__ == '__main__':
    args = get_sindy_args()

    # Initialize wandb
    wandb.init(project='anonym', name=args.wandb_name, config=args)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # args to dict
    args = vars(args)

    # Load dataset
    train_dataset, val_dataset, args = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
    
    # Initialize model
    AEType = AutoEncoder
    if args['load_ae']:
        autoencoder = AEType(**args).to(args['device'])
        autoencoder.load_state_dict(torch.load(f'saved_models/{args["load_dir"]}/autoencoder.pt'))
    elif args['learn_ae']:
        autoencoder = AEType(**args).to(args['device'])
    else:
        args['ae_arch'] = 'none'
        autoencoder = AEType(**args).to(args['device'])
    if args['load_Lie']:
        L_list = torch.load(f'saved_models/{args["load_dir"]}/Lie_list.pt')
        # Consider only the first set of Lie generators
        L_list = L_list[0].detach().cpu()
        args['L_list'] = [L_list[i] for i in range(L_list.shape[0])]
    else:
        args['L_list'] = []
    regressor = SINDyRegression(**args).to(args['device'])
    
    # Train regressor
    train_fn = train_SINDy
    train_fn(autoencoder, regressor, train_loader, val_loader, **args)

    wandb.finish()
