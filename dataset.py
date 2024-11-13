import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import scipy.io as sio
import pickle
import re
import os
from tqdm import trange, tqdm
from glob import glob
from PIL import Image
from data_utils.lotka import get_lv_data

data_path = './data'

def get_dataset(args):
    if args['task'] == 'rd':
        train_dataset = ReactionDiffusionDataset(mode='train')
        val_dataset = ReactionDiffusionDataset(mode='val')
        args['input_dim'] = train_dataset[0][0].shape[0]
        args['flatten'] = False
    elif args['task'] == 'mt_rd':
        train_dataset = MultiTimestepReactionDiffusionDataset(mode='train')
        val_dataset = MultiTimestepReactionDiffusionDataset(mode='val')
        args['input_dim'] = train_dataset[0][0].shape[1]
        args['mt_data'] = True
    elif args['task'] == 'lv':
        train_dataset = ODEDataset(ode_name='lv', mode='train', noise=args['noise'], smoothing=args['smoothing'])
        val_dataset = ODEDataset(ode_name='lv', mode='val', noise=args['noise'], smoothing=args['smoothing'])
        args['input_dim'] = train_dataset[0][0].shape[-1]
    elif args['task'] == 'mt_lv':
        train_dataset = MTODEDataset(ode_name='lv', mode='train', noise=args['noise'], smoothing=args['smoothing'])
        val_dataset = MTODEDataset(ode_name='lv', mode='val', noise=args['noise'], smoothing=args['smoothing'])
        args['input_dim'] = train_dataset[0][0].shape[-1]
        args['mt_data'] = True
    elif args['task'] == 'selkov':
        train_dataset = ODEDataset(ode_name='selkov', mode='train', noise=args['noise'], smoothing=args['smoothing'])
        val_dataset = ODEDataset(ode_name='selkov', mode='val', noise=args['noise'], smoothing=args['smoothing'])
        args['input_dim'] = train_dataset[0][0].shape[-1]
    elif args['task'] == 'mt_selkov':
        n_timesteps = 2
        interval = 50
        train_dataset = MTODEDataset(ode_name='selkov', mode='train', noise=args['noise'], smoothing=args['smoothing'], n_timesteps=n_timesteps, interval=interval)
        val_dataset = MTODEDataset(ode_name='selkov', mode='val', noise=args['noise'], smoothing=args['smoothing'], n_timesteps=n_timesteps, interval=interval)
        args['input_dim'] = train_dataset[0][0].shape[-1]
        args['mt_data'] = True
    elif args['task'] == 'dosc':
        train_dataset = ODEDataset(ode_name='dosc', mode='train', noise=args['noise'], smoothing=args['smoothing'])
        val_dataset = ODEDataset(ode_name='dosc', mode='val', noise=args['noise'], smoothing=args['smoothing'])
        args['input_dim'] = train_dataset[0][0].shape[-1]
    elif args['task'] == 'growth':
        train_dataset = ODEDataset(ode_name='growth', mode='train', noise=args['noise'], smoothing=args['smoothing'])
        val_dataset = ODEDataset(ode_name='growth', mode='val', noise=args['noise'], smoothing=args['smoothing'])
        args['input_dim'] = train_dataset[0][0].shape[-1]
    else:
        raise NotImplementedError
    return train_dataset, val_dataset, args
    
# Modified from SINDy AE: https://github.com/kpchamp/SindyAutoencoders/blob/master/examples/rd/example_reactiondiffusion.py
class ReactionDiffusionDataset(Dataset):
    def __init__(self, path=f'{data_path}/reaction_diffusion.mat', random=False, mode='train', downsample=False):
        data = sio.loadmat(path)
        n_samples = data['t'].size
        n = data['x'].size
        N = n*n

        data['uf'] += 1e-6*np.random.randn(data['uf'].shape[0], data['uf'].shape[1], data['uf'].shape[2])
        data['duf'] += 1e-6*np.random.randn(data['duf'].shape[0], data['duf'].shape[1], data['duf'].shape[2])

        if not random:
            # consecutive samples
            training_samples = np.arange(int(.8*n_samples))
            val_samples = np.arange(int(.8*n_samples), int(.9*n_samples))
            test_samples = np.arange(int(.9*n_samples), n_samples)
        else:
            # random samples
            perm = np.random.permutation(int(.9*n_samples))
            training_samples = perm[:int(.8*n_samples)]
            val_samples = perm[int(.8*n_samples):]
            test_samples = np.arange(int(.9*n_samples), n_samples)

        if mode == 'train':
            samples = training_samples
        elif mode == 'val':
            samples = val_samples
        elif mode == 'test':
            samples = test_samples

        self.data = {'t': data['t'][samples],
                        'y1': data['x'].T,
                        'y2': data['y'].T,
                        'x': data['uf'][:,:,samples].reshape((N,-1)).T,
                        'dx': data['duf'][:,:,samples].reshape((N,-1)).T}
        if downsample:
            # reshape each x (10000) to (100, 100) and downsample to (28, 28)
            from scipy.ndimage import zoom
            x = self.data['x'].reshape((-1, 100, 100))
            dx = self.data['dx'].reshape((-1, 100, 100))
            downsampled_x = np.zeros((self.data['x'].shape[0], 28, 28))
            downsampled_dx = np.zeros((self.data['x'].shape[0], 28, 28))
            for i in range(self.data['x'].shape[0]):
                downsampled_x[i] = zoom(x[i], 0.28)
                downsampled_dx[i] = zoom(dx[i], 0.28)
            self.data['x'] = downsampled_x.reshape((-1, 28*28))
            self.data['dx'] = downsampled_dx.reshape((-1, 28*28))

        self.data = {k: torch.FloatTensor(v) for k, v in self.data.items()}
        
    def __len__(self):
        return self.data['x'].shape[0]

    def __getitem__(self, idx):
        return self.data['x'][idx], self.data['dx'][idx], self.data['dx'][idx]
    

class MultiTimestepReactionDiffusionDataset(Dataset):
    def __init__(self, path=f'{data_path}/reaction_diffusion.mat', n_timesteps=2, mode='train', downsample=False):
        data = sio.loadmat(path)
        n_samples = data['t'].size
        n = data['x'].size
        N = n*n

        data['uf'] += 1e-6*np.random.randn(data['uf'].shape[0], data['uf'].shape[1], data['uf'].shape[2])
        data['duf'] += 1e-6*np.random.randn(data['duf'].shape[0], data['duf'].shape[1], data['duf'].shape[2])

        training_samples = np.arange(int(.8*n_samples))
        val_samples = np.arange(int(.8*n_samples), int(.9*n_samples))
        test_samples = np.arange(int(.9*n_samples), n_samples)
        if mode == 'train':
            samples = training_samples
        elif mode == 'val':
            samples = val_samples
        elif mode == 'test':
            samples = test_samples

        if downsample:
            # reshape each x (10000) to (100, 100) and downsample to (28, 28)
            from scipy.ndimage import zoom
            x = data['uf'].reshape(100, 100, -1)
            dx = data['duf'].reshape(100, 100, -1)
            downsampled_x = np.zeros((28, 28, x.shape[-1]))
            downsampled_dx = np.zeros((28, 28, x.shape[-1]))
            for i in range(x.shape[-1]):
                downsampled_x[..., i] = zoom(x[..., i], 0.28)
                downsampled_dx[..., i] = zoom(dx[..., i], 0.28)
            data['uf'] = downsampled_x
            data['duf'] = downsampled_dx

        self.data = []
        for i in range(n_timesteps, len(samples)):
            self.data.append({'x': torch.FloatTensor(np.transpose(data['uf'][:,:,samples[i-n_timesteps:i]], axes=(2,0,1)).reshape((n_timesteps,-1))),
                              'dx': torch.FloatTensor(np.transpose(data['duf'][:,:,samples[i-n_timesteps:i]], axes=(2,0,1)).reshape((n_timesteps,-1)))})
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['x'], self.data[idx]['dx']
    

ode_dt_dict = {
    'lv': 0.002,
    'selkov': 0.002,
    'dosc': 0.2,
    'growth': 0.02,
    'rd': 0.05,
}
    

class ODEDataset(Dataset):
    def __init__(self, path=f'{data_path}', ode_name='lv', mode='train', noise=0.0, smoothing=None):
        super().__init__()
        smoothing_str = f'-{smoothing}' if smoothing is not None else ''
        try:
            print(f'Loading existing {ode_name} {mode} data...')
            x = torch.load(f'{path}/{ode_name}-{mode}-noise{int(100*noise):02d}{smoothing_str}-x.pt')
            dx = torch.load(f'{path}/{ode_name}-{mode}-noise{int(100*noise):02d}{smoothing_str}-dx.pt')
        except FileNotFoundError:
            print(f'Load data failed. Generating {ode_name} {mode} data...')
            n_ics = 200 if 'train' in mode else 20
            num_steps = 10000
            x, dx = get_lv_data(n_ics=n_ics, noise=noise, num_steps=num_steps, smoothing=smoothing, gp_sigma_in=0.1)
            x = torch.FloatTensor(x)
            dx = torch.FloatTensor(dx)
            torch.save(x, f'{path}/{ode_name}-{mode}-noise{int(100*noise):02d}{smoothing_str}-x.pt')
            torch.save(dx, f'{path}/{ode_name}-{mode}-noise{int(100*noise):02d}{smoothing_str}-dx.pt')

        x = x.to(torch.float32)
        dx = dx.to(torch.float32)

        n_ics, n_steps, input_dim = x.shape
        self.n_ics, self.n_steps, self.input_dim = n_ics, n_steps, input_dim
        self.x = x.reshape((n_ics*n_steps, input_dim))
        self.dx = dx.reshape((n_ics*n_steps, input_dim))

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx]
    

class MTODEDataset(Dataset):
    def __init__(self, path=f'{data_path}', ode_name='lv', mode='train', n_timesteps=2, interval=10, noise=0.0, smoothing=None):
        super().__init__()
        smoothing_str = f'-{smoothing}' if smoothing is not None else ''
        try:
            print(f'Loading existing {ode_name} {mode} data...')
            x = torch.load(f'{path}/{ode_name}-{mode}-noise{int(100*noise):02d}{smoothing_str}-x.pt')
            dx = torch.load(f'{path}/{ode_name}-{mode}-noise{int(100*noise):02d}{smoothing_str}-dx.pt')
        except FileNotFoundError:
            print(f'Load data failed. Generating {ode_name} {mode} data...')
            n_ics = 200 if 'train' in mode else 20
            num_steps = 10000
            x, dx = get_lv_data(n_ics=n_ics, noise=noise, num_steps=num_steps, smoothing=smoothing, gp_sigma_in=0.1)
            x = torch.FloatTensor(x)
            dx = torch.FloatTensor(dx)
            torch.save(x, f'{path}/{ode_name}-{mode}-noise{int(100*noise):02d}{smoothing_str}-x.pt')
            torch.save(dx, f'{path}/{ode_name}-{mode}-noise{int(100*noise):02d}{smoothing_str}-dx.pt')

        x = x.to(torch.float32)
        dx = dx.to(torch.float32)

        self.n_timesteps = n_timesteps
        if n_timesteps < 2:
            raise ValueError('n_timesteps must be greater than 1 for multi-timestep dataset')
        n_ics, n_steps, input_dim = x.shape
        self.n_ics, self.n_steps, self.input_dim = n_ics, n_steps, input_dim
        self.x = []
        self.dx = []
        for i in range(n_ics):
            for j in range(n_steps-n_timesteps*interval):
                self.x.append(x[i, j:j+n_timesteps*interval:interval, :].reshape((n_timesteps, input_dim)))
                self.dx.append(dx[i, j:j+n_timesteps*interval:interval, :].reshape((n_timesteps, input_dim)))

        self.x = torch.stack(self.x)
        self.dx = torch.stack(self.dx)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx]


class LotkaVolterraDataset(Dataset):
    def __init__(self, path=f'{data_path}', mode='train', noise=0.0, smoothing=None):
        super().__init__()
        smoothing_str = f'-{smoothing}' if smoothing is not None else ''
        try:
            print(f'Loading existing Lotka-Volterra {mode} data...')
            x = torch.load(f'{path}/lv-{mode}-noise{int(100*noise):02d}{smoothing_str}-x.pt')
            dx = torch.load(f'{path}/lv-{mode}-noise{int(100*noise):02d}{smoothing_str}-dx.pt')
        except FileNotFoundError:
            print(f'Load data failed. Generating Lotka-Volterra {mode} data...')
            n_ics = 200 if 'train' in mode else 20
            num_steps = 10000
            x, dx = get_lv_data(n_ics=n_ics, noise=noise, num_steps=num_steps, smoothing=smoothing, gp_sigma_in=0.1)
            x = torch.FloatTensor(x)
            dx = torch.FloatTensor(dx)
            torch.save(x, f'{path}/lv-{mode}-noise{int(100*noise):02d}{smoothing_str}-x.pt')
            torch.save(dx, f'{path}/lv-{mode}-noise{int(100*noise):02d}{smoothing_str}-dx.pt')

        n_ics, n_steps, input_dim = x.shape
        self.n_ics, self.n_steps, self.input_dim = n_ics, n_steps, input_dim
        self.x = x.reshape((n_ics*n_steps, input_dim))
        self.dx = dx.reshape((n_ics*n_steps, input_dim))

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx]
    

class MTLotkaVolterraDataset(Dataset):
    def __init__(self, path=f'{data_path}', n_timesteps=2, interval=10, mode='train', noise=0.0, smoothing=None):
        super().__init__()
        smoothing_str = f'-{smoothing}' if smoothing is not None else ''
        try:
            print(f'Loading existing Lotka-Volterra {mode} data...')
            x = torch.load(f'{path}/lv-{mode}-noise{int(100*noise):02d}{smoothing_str}-x.pt')
            dx = torch.load(f'{path}/lv-{mode}-noise{int(100*noise):02d}{smoothing_str}-dx.pt')
        except FileNotFoundError:
            print(f'Load data failed. Generating Lotka-Volterra {mode} data...')
            n_ics = 200 if 'train' in mode else 20
            num_steps = 10000
            x, dx = get_lv_data(n_ics=n_ics, noise=noise, num_steps=num_steps, smoothing=smoothing, gp_sigma_in=0.1)
            x = torch.FloatTensor(x)
            dx = torch.FloatTensor(dx)
            torch.save(x, f'{path}/lv-{mode}-noise{int(100*noise):02d}{smoothing_str}-x.pt')
            torch.save(dx, f'{path}/lv-{mode}-noise{int(100*noise):02d}{smoothing_str}-dx.pt')

        self.n_timesteps = n_timesteps
        if n_timesteps < 2:
            raise ValueError('n_timesteps must be greater than 1 for multi-timestep dataset')
        n_ics, n_steps, input_dim = x.shape
        self.n_ics, self.n_steps, self.input_dim = n_ics, n_steps, input_dim
        self.x = []
        self.dx = []
        for i in range(n_ics):
            for j in range(n_steps-n_timesteps*interval):
                self.x.append(x[i, j:j+n_timesteps*interval:interval, :].reshape((n_timesteps, input_dim)))
                self.dx.append(dx[i, j:j+n_timesteps*interval:interval, :].reshape((n_timesteps, input_dim)))

        self.x = torch.stack(self.x)
        self.dx = torch.stack(self.dx)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx]
    
    
class SimpleLinear(Dataset):
    """
    Simple Linear Example
    n_samples: number of samples
    coeff: coefficients of the linear equation

    e.g. coeff = [a b
                  c d]
         dx = ax+by
         dy = cx+dy
    """
    def __init__(self, n_samples, coeff=torch.tensor([[1.0, 0.0], [0.0, 1.0]])):
        self.x = torch.randn(n_samples, 2) * 10
        self.dx = self.x @ coeff.T

        '''
        # evolve with time step 0.01
        for i in range(1, n_samples):
            self.x[i, :] = self.x[i-1, :] + 0.01 * self.x[i-1, :] @ coeff.T
        self.dx = self.x @ coeff.T
        '''

        # add noise
        self.x += 1e-3 * torch.randn_like(self.x)
        self.dx += 1e-3 * torch.randn_like(self.dx)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx], self.dx[idx]
    