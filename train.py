import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from autoencoder import AutoEncoder
from gan import Discriminator, LieGenerator
from tqdm import tqdm, trange
import wandb
import os
from torch.autograd.functional import jvp
from model_utils import *
from sindy import *


def train_lassi(
    autoencoder, discriminator, generator, train_loader, test_loader,
    num_epochs, lr_ae, lr_d, lr_g, w_recon, w_gan, w_reg_norm, w_reg_sim, w_reg_ortho, w_reg_closure,
    use_original_x, gan_st_freq, gan_st_thres, ae_arch,
    include_sindy, regressor, lr_sindy, w_sindy_z, w_sindy_x, sindy_reg_type, w_sindy_reg, st_freq, threshold,
    device, log_interval, save_interval, save_dir, **kwargs
):
    no_ae_flag = (ae_arch == 'none')
    if no_ae_flag:
        optimizer_ae = None
    else:
        optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=lr_ae)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g)
    if include_sindy:
        if w_sindy_x > 0.0:
            optimizer_sindy = torch.optim.Adam(regressor.parameters(), lr=lr_sindy)
            scheduler_sindy = torch.optim.lr_scheduler.MultiStepLR(optimizer_sindy, milestones=[1, 2, 3], gamma=10)
            sindy_loss = torch.nn.MSELoss()
        else:  # optimize regressor by lstsq in latent space
            optimizer_sindy = None
            scheduler_sindy = None
            sindy_loss = torch.nn.MSELoss()
            for p in regressor.parameters():
                p.requires_grad = False
    else:
        optimizer_sindy = None
        scheduler_sindy = None
        w_sindy_z = w_sindy_x = w_sindy_reg = 0.0
    adversarial_loss = torch.nn.BCELoss()
    recon_loss = torch.nn.MSELoss()

    log_items = [
        'loss_ae', 'loss_g', 'loss_reg_norm', 'loss_reg_ortho', 'loss_reg_closure', 'loss_d_real', 'loss_d_fake', 'loss_ae_rel',
        'loss_sindy_x', 'loss_sindy_z', 'loss_sindy_reg',
    ]
    log_items_test = [
        'test_loss_ae', 'test_loss_g', 'test_loss_d_real', 'test_loss_d_fake',
        'test_loss_sindy_x', 'test_loss_sindy_z',
    ]
    log_flag = [w > 0 for w in [
        w_recon, w_gan, w_reg_norm, w_reg_ortho, w_reg_closure, w_gan, w_gan, w_recon,
        w_sindy_x, w_sindy_z, w_sindy_reg,
    ]]
    log_flag = {k: v for k, v in zip(log_items, log_flag)}
    log_flag_test = [w > 0 for w in [
        w_recon, w_gan, w_gan, w_gan, w_sindy_x, w_sindy_z,
    ]]
    log_flag_test = {k: v for k, v in zip(log_items_test, log_flag_test)}

    for epoch in range(num_epochs):
        # torch.autograd.set_detect_anomaly(True)
        running_losses = { k: [] for k in log_items }
        autoencoder.train()
        discriminator.train()
        generator.train()
        if include_sindy:
            regressor.train()
        for i, (x, dx) in enumerate(train_loader):
            x = x.to(device)
            if include_sindy:
                dx = dx.to(device)
            bs = x.shape[0]

            # Adversarial ground truths
            valid = torch.ones((bs, 1)).to(device)
            fake = torch.zeros((bs, 1)).to(device)

            # Reconstruction loss
            z, xhat = autoencoder(x)
            loss_ae = recon_loss(xhat, x)
            running_losses['loss_ae'].append(loss_ae.item())
            loss_ae_rel = loss_ae / recon_loss(x, torch.zeros_like(x))
            running_losses['loss_ae_rel'].append(loss_ae_rel.item())
            loss = w_recon * loss_ae

            # Generator loss
            zt = generator(z)  # transformed latent space representation
            xt = autoencoder.decode(zt) if use_original_x else None
            d_fake = discriminator(zt, None, xt)
            loss_g = adversarial_loss(d_fake, valid)
            running_losses['loss_g'].append(loss_g.item())
            loss = loss + w_gan * loss_g

            if not np.isclose(w_reg_norm, 0.0):
                loss_reg_norm = generator.reg_norm()
                running_losses['loss_reg_norm'].append(loss_reg_norm.item())
                loss = loss + w_reg_norm * loss_reg_norm
                # loss = loss + w_reg_norm * torch.abs(nn.CosineSimilarity(dim=-1)(zt, z).mean())
            elif not np.isclose(w_reg_sim, 0.0):  # alternatively, use data similarity for regularization
                loss_reg_norm = torch.abs(nn.CosineSimilarity(dim=-1)(zt, z).mean())
                running_losses['loss_reg_norm'].append(loss_reg_norm.item())
                loss = loss + w_reg_sim * loss_reg_norm
            else:
                running_losses['loss_reg_norm'].append(0.0)

            if not np.isclose(w_reg_ortho, 0.0):
                loss_reg_ortho = generator.reg_ortho()
                running_losses['loss_reg_ortho'].append(loss_reg_ortho.item())
                loss = loss + w_reg_ortho * loss_reg_ortho
            else:
                running_losses['loss_reg_ortho'].append(0.0)

            if not np.isclose(w_reg_closure, 0.0):
                loss_reg_closure = generator.reg_closure()
                running_losses['loss_reg_closure'].append(loss_reg_closure.item())
                loss = loss + w_reg_closure * loss_reg_closure
            else:
                running_losses['loss_reg_closure'].append(0.0)

            # Discriminator loss
            z_detached = z.detach()
            zt_detached = zt.detach()
            x_detached = xhat.detach() if use_original_x else None
            xt_detached = xt.detach() if use_original_x else None
            loss_d_real = adversarial_loss(discriminator(z_detached, x_detached), valid)
            loss_d_fake = adversarial_loss(discriminator(zt_detached, xt_detached), fake)
            running_losses['loss_d_real'].append(loss_d_real.item())
            running_losses['loss_d_fake'].append(loss_d_fake.item())
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss = loss + loss_d

            # SINDy loss
            if include_sindy:
                if w_sindy_x > 0.0:
                    dz = autoencoder.compute_dz(x, dx)
                    dz_pred = regressor(z)
                    dx_pred = autoencoder.compute_dx(z, dz_pred)
                    loss_sindy_z = sindy_loss(dz_pred, dz)
                    loss_sindy_x = w_sindy_x * sindy_loss(dx_pred, dx)
                    running_losses['loss_sindy_z'].append(loss_sindy_z.item())
                    running_losses['loss_sindy_x'].append(loss_sindy_x.item())
                    loss = loss + w_sindy_z * loss_sindy_z + w_sindy_x * loss_sindy_x
                    if sindy_reg_type == 'l1':
                        loss_sindy_reg = sum([torch.norm(p, 1) for p in regressor.parameters()])
                        running_losses['loss_sindy_reg'].append(loss_sindy_reg.item())
                        loss = loss + w_sindy_reg * loss_sindy_reg
                    else:
                        raise ValueError(f'Unknown regularization type: {sindy_reg_type}')
                else:  # solve lstsq in latent space
                    dz = autoencoder.compute_dz(x, dx)
                    if regressor.constraint:
                        with torch.no_grad():
                            # check for difference between new and old Li
                            L_list = generator.get_full_basis_list()
                            repr_dim = L_list[0].shape[-1] // kwargs['n_comps']
                            L_trunc = [L[:repr_dim, :repr_dim].detach().cpu() for L in L_list]
                            diff = sum([torch.norm(L_trunc[i] - regressor.L_list[i]) for i in range(len(L_trunc))])
                            if diff > 0.1 or i == len(train_loader) - 1:  # significant change or last batch
                                regressor.update_Q(L_trunc)
                    loss_sindy_z = solve_SINDy(regressor, z[:, 0], dz[:, 0], w_sindy_reg, threshold)
                    running_losses['loss_sindy_z'].append(loss_sindy_z.item())
                    loss = loss + w_sindy_z * loss_sindy_z
                    running_losses['loss_sindy_x'].append(0.0)
                    running_losses['loss_sindy_reg'].append(0.0)
            else:
                running_losses['loss_sindy_z'].append(0.0)
                running_losses['loss_sindy_x'].append(0.0)
                running_losses['loss_sindy_reg'].append(0.0)
                
            # Backprop
            if not no_ae_flag:
                optimizer_ae.zero_grad()
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            if optimizer_sindy is not None:
                optimizer_sindy.zero_grad()
            loss.backward()
            if not no_ae_flag:
                optimizer_ae.step()
            optimizer_d.step()
            optimizer_g.step()
            if optimizer_sindy is not None:
                optimizer_sindy.step()

        if scheduler_sindy is not None:
            scheduler_sindy.step()

        # sequential thresholding
        if gan_st_freq > 0 and (epoch + 1) % gan_st_freq == 0:
            generator.set_threshold(gan_st_thres)
            # w_reg *= 0.5
        if include_sindy and st_freq > 0 and (epoch + 1) % st_freq == 0 and w_sindy_x > 0.0:
            regressor.set_threshold(threshold)

        # Log progress
        wandb_log = {
            k: np.mean(running_losses[k]) for k in log_items
        }
        if (epoch + 1) % log_interval == 0:
            print(', '.join([ f'Epoch {epoch}' ] + 
                [ f'{k}: {np.mean(running_losses[k]):.4f}' for k in log_items if log_flag[k] ]
            ))
            autoencoder.eval()
            discriminator.eval()
            generator.eval()
            with torch.no_grad():
                running_losses = { k: [] for k in log_items_test }
                for i, (x, dx) in enumerate(test_loader):
                    x = x.to(device)
                    dx = dx.to(device)
                    bs = x.shape[0]
                    valid = torch.ones((bs, 1)).to(device)
                    fake = torch.zeros((bs, 1)).to(device)
                    z, xhat = autoencoder(x)
                    zt = generator(z)
                    xt = autoencoder.decode(zt)
                    d_fake = discriminator(zt, None, xt if use_original_x else None)
                    d_real = discriminator(z, None, x if use_original_x else None)
                    loss_ae = recon_loss(xhat, x)
                    loss_ae_rel = loss_ae / recon_loss(x, torch.zeros_like(x))
                    loss_g = adversarial_loss(d_fake, valid)
                    loss_d_real = adversarial_loss(d_real, valid)
                    loss_d_fake = adversarial_loss(d_fake, fake)
                    running_losses['test_loss_ae'].append(loss_ae.item())
                    running_losses['test_loss_g'].append(loss_g.item())
                    running_losses['test_loss_d_real'].append(loss_d_real.item())
                    running_losses['test_loss_d_fake'].append(loss_d_fake.item())
                    if include_sindy:
                        dz = autoencoder.compute_dz(x, dx)
                        dz_pred = regressor(z)
                        dx_pred = autoencoder.compute_dx(z, dz_pred)
                        loss_sindy_z = sindy_loss(dz_pred, dz)
                        loss_sindy_x = sindy_loss(dx_pred, dx)
                        running_losses['test_loss_sindy_z'].append(loss_sindy_z.item())
                        running_losses['test_loss_sindy_x'].append(loss_sindy_x.item())
                    else:
                        running_losses['test_loss_sindy_z'].append(0.0)
                        running_losses['test_loss_sindy_x'].append(0.0)
                    
                wandb_log.update({
                    k: np.mean(running_losses[k]) for k in log_items_test
                })
                print(', '.join([ f'Epoch {epoch}' ] + 
                    [ f'{k}: {np.mean(running_losses[k]):.4f}' for k in log_items_test if log_flag_test[k] ]
                ))
                if kwargs['print_li']:
                    print(generator.getLi())
                    # print(generator.getStructureConst())
                if include_sindy:
                    regressor.print()

        wandb.log(wandb_log)
        
        if (epoch + 1) % save_interval == 0:
            if not os.path.exists(f'saved_models/{save_dir}'):
                os.makedirs(f'saved_models/{save_dir}')
            torch.save(autoencoder.state_dict(), f'saved_models/{save_dir}/autoencoder_{epoch}.pt')
            torch.save(discriminator.state_dict(), f'saved_models/{save_dir}/discriminator_{epoch}.pt')
            torch.save(generator.state_dict(), f'saved_models/{save_dir}/generator_{epoch}.pt')
            torch.save(generator.masks, f'saved_models/{save_dir}/generator_mask_{epoch}.pt')
            if include_sindy:
                torch.save(regressor.state_dict(), f'saved_models/{save_dir}/regressor_{epoch}.pt')
                torch.save(regressor.L_list, f'saved_models/{save_dir}/regressor_lie_list_{epoch}.pt')


def train_SINDy(
        autoencoder, regressor, train_loader, test_loader,
        num_epochs, lr, reg_type, w_reg, seq_thres_freq, threshold, rel_loss, 
        device, log_interval, save_interval, save_dir, **kwargs
):
    # Initialize optimizers
    optimizer_sindy = torch.optim.Adam(regressor.parameters(), lr=lr)

    # Loss functions
    sindy_loss = torch.nn.MSELoss()
    recon_loss = torch.nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        regressor.train()
        running_losses = [[], [], [], []]
        for i, (x, dx, _) in enumerate(train_loader):
            x = x.to(device)
            dx = dx.to(device)
            # Regularization loss
            if reg_type == 'l1':
                loss_reg = sum([torch.norm(p, 1) for p in regressor.parameters()])
                running_losses[0].append(loss_reg.item())
            else:
                raise ValueError(f'Unknown regularization type: {reg_type}')
            loss = w_reg * loss_reg       
            
            # Reconstruction loss
            z, xhat = autoencoder(x)
            loss_recon = recon_loss(xhat, x)
            running_losses[1].append(loss_recon.item())
            # dz loss & dx loss
            dz = autoencoder.compute_dz(x, dx)
            dz_pred = regressor(z)
            dx_pred = autoencoder.compute_dx(z, dz_pred)
            if rel_loss:
                # Denominator at least 0.1
                denom = torch.max(sindy_loss(dz, torch.zeros_like(dz, device=device)),
                                    torch.ones_like(loss, device=device) * 0.1)
                loss_sindy_z = sindy_loss(dz_pred, dz) / denom
            else:
                loss_sindy_z = sindy_loss(dz_pred, dz)
            loss_sindy_x = sindy_loss(dx_pred, dx)
            running_losses[2].append(loss_sindy_z.item())
            running_losses[3].append(loss_sindy_x.item())
            loss += loss_sindy_z

            # Optimization
            optimizer_sindy.zero_grad()
            loss.backward()
            optimizer_sindy.step()


        # Sequential thresholding
        if seq_thres_freq > 0 and (epoch + 1) % seq_thres_freq == 0:
            regressor.set_threshold(threshold)
            w_reg *= 0.5

        # Log progress
        wandb.log({'loss_reg': np.mean(running_losses[0]),
                   'loss_recon': np.mean(running_losses[1]),
                    'loss_sindy_z': np.mean(running_losses[2]),
                    'loss_sindy_x': np.mean(running_losses[3])})

        if (epoch + 1) % log_interval == 0:
            print(f'Epoch {epoch}, loss_reg: {np.mean(running_losses[0]):.4f}, '
                  f'loss_recon: {np.mean(running_losses[1]):.4f}, '
                  f'loss_sindy_z: {np.mean(running_losses[2]):.4f}, '
                  f'loss_sindy_x: {np.mean(running_losses[3]):.4f}')
            regressor.eval()
            autoencoder.eval()
            with torch.no_grad():
                running_losses = [[], [], [], []]
                for i, (x, dx, _) in enumerate(test_loader):
                    x = x.to(device)
                    dx = dx.to(device)
                    z, xhat = autoencoder(x)
                    loss_recon = recon_loss(xhat, x)
                    dz = autoencoder.compute_dz(x, dx)
                    dz_pred = regressor(z)
                    dx_pred = autoencoder.compute_dx(z, dz_pred)
                    loss_sindy_z = sindy_loss(dz_pred, dz)
                    loss_sindy_x = sindy_loss(dx_pred, dx)
                    running_losses[1].append(loss_recon.item())
                    running_losses[2].append(loss_sindy_z.item())
                    running_losses[3].append(loss_sindy_x.item())

                    # Regularization
                    if reg_type == 'l1':
                        loss_reg = sum([torch.norm(p, 1) for p in regressor.parameters()])
                        running_losses[0].append(loss_reg.item())
                    else:
                        raise ValueError(f'Unknown regularization type: {reg_type}')
                    
                wandb.log({'test_loss_reg': np.mean(running_losses[0]),
                           'test_loss_recon': np.mean(running_losses[1]),
                           'test_loss_sindy_z': np.mean(running_losses[2]),
                           'test_loss_sindy_x': np.mean(running_losses[3]),})
                print(f'Epoch {epoch} test, loss_reg: {np.mean(running_losses[0]):.4f}, '
                      f'loss_recon: {np.mean(running_losses[1]):.4f}, '
                      f'loss_sindy_z: {np.mean(running_losses[2]):.4f}, '
                      f'loss_sindy_x: {np.mean(running_losses[3]):.4f}')
                regressor.print()

        if (epoch + 1) % save_interval == 0:
            if not os.path.exists(f'saved_models/{save_dir}'):
                os.makedirs(f'saved_models/{save_dir}')
            torch.save(regressor.state_dict(), f'saved_models/{save_dir}/regressor_{epoch}.pt')


def train_SIGED(
    train_loader, test_loader, num_epochs, device, log_interval, save_interval, save_dir,  ## global
    autoencoder, discriminator, generator,  # symmetry discovery model
    lr_ae, lr_d, lr_g, w_recon, w_gan, w_reg_norm, w_reg_ortho, w_reg_closure,  # symmetry discovery parameters
    use_original_x, gan_st_freq, gan_st_thres, ae_arch,  # symmetry discovery parameters
    regressor, use_latent, lr_sindy, w_sindy_z, w_sindy_x, sindy_reg_type, w_sindy_reg, w_sym_reg, st_freq, threshold, int_t, int_dt,  # SINDy
    **kwargs
):
    # no_ae_flag = (ae_arch == 'none')
    # if no_ae_flag:
    #     optimizer_ae = None
    # else:
    #     optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=lr_ae)
    # optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    # optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g)
    optimizer_sindy = torch.optim.Adam(regressor.parameters(), lr=lr_sindy)
    sindy_loss = torch.nn.MSELoss()
    adversarial_loss = torch.nn.BCELoss()
    recon_loss = torch.nn.MSELoss()
    symm_loss = make_symmreg_pttrain(autoencoder, generator)

    log_items = [
        'loss_sindy_x', 'loss_sindy_z', 'loss_sindy_reg', 'loss_sym_reg',
        'loss_ae', 'loss_g', 'loss_reg_norm', 'loss_reg_ortho', 'loss_reg_closure', 'loss_d_real', 'loss_d_fake', 'loss_ae_rel',
    ]
    log_items_test = [
        'test_loss_sindy_x', 'test_loss_sindy_z',
        'test_loss_ae', 'test_loss_g', 'test_loss_d_real', 'test_loss_d_fake',
    ]
    log_flag = [w > 0 for w in [
        w_sindy_x, w_sindy_z, w_sindy_reg, w_sym_reg,
        w_recon, w_gan, w_reg_norm, w_reg_ortho, w_reg_closure, w_gan, w_gan, w_recon,
    ]]
    log_flag = {k: v for k, v in zip(log_items, log_flag)}
    log_flag_test = [w > 0 for w in [
        w_sindy_x, w_sindy_z, w_recon, w_gan, w_gan, w_gan,
    ]]
    log_flag_test = {k: v for k, v in zip(log_items_test, log_flag_test)}

    # regressor.Xi.data = torch.FloatTensor([
    #             [0.667, 0, 0, 0, 0, 0, 0, -1.333],
    #             [-1, 0, 0, 0, 0, 0, 1, 0],
    #         ]).to(regressor.Xi.device)

    for epoch in range(num_epochs):
        running_losses = { k: [] for k in log_items }
        # autoencoder.train()
        # discriminator.train()
        # generator.train()
        regressor.train()
        for i, (x, dx) in enumerate(train_loader):
            x = x.to(device)
            dx = dx.to(device)
            bs, xdim = x.shape[0], x.shape[-1]

            # # Adversarial ground truths
            # valid = torch.ones((bs, 1)).to(device)
            # fake = torch.zeros((bs, 1)).to(device)

            # # Reconstruction loss
            # z, xhat = autoencoder(x_gan)
            # loss_ae = w_recon * recon_loss(xhat, x_gan)
            # running_losses['loss_ae'].append(loss_ae.item() / max(w_recon, 1e-6))
            # loss_ae_rel = loss_ae / recon_loss(x_gan, torch.zeros_like(x_gan))
            # running_losses['loss_ae_rel'].append(loss_ae_rel.item() / max(w_recon, 1e-6))
            # loss = loss_ae

            # # Generator loss
            # zt = generator(z)  # transformed latent space representation
            # xt = autoencoder.decode(zt) if use_original_x else None
            # d_fake = discriminator(zt, None, xt)
            # loss_g = w_gan * adversarial_loss(d_fake, valid)
            # running_losses['loss_g'].append(loss_g.item() / max(w_gan, 1e-6))
            # loss = loss + loss_g

            # if not np.isclose(w_reg_norm, 0.0):
            #     loss_reg_norm = generator.reg_norm()
            #     running_losses['loss_reg_norm'].append(loss_reg_norm.item())
            #     loss = loss + w_reg_norm * loss_reg_norm
            #     # loss = loss + w_reg_norm * torch.abs(nn.CosineSimilarity(dim=-1)(zt, z).mean())
            # else:
            #     running_losses['loss_reg_norm'].append(0.0)

            # if not np.isclose(w_reg_ortho, 0.0):
            #     loss_reg_ortho = generator.reg_ortho()
            #     running_losses['loss_reg_ortho'].append(loss_reg_ortho.item())
            #     loss = loss + w_reg_ortho * loss_reg_ortho
            # else:
            #     running_losses['loss_reg_ortho'].append(0.0)

            # if not np.isclose(w_reg_closure, 0.0):
            #     loss_reg_closure = generator.reg_closure()
            #     running_losses['loss_reg_closure'].append(loss_reg_closure.item())
            #     loss = loss + w_reg_closure * loss_reg_closure
            # else:
            #     running_losses['loss_reg_closure'].append(0.0)

            # # Discriminator loss
            # z_detached = z.detach()
            # zt_detached = zt.detach()
            # x_detached = xhat.detach() if use_original_x else None
            # xt_detached = xt.detach() if use_original_x else None
            # loss_d_real = adversarial_loss(discriminator(z_detached, None, x_detached), valid)
            # loss_d_fake = adversarial_loss(discriminator(zt_detached, None, xt_detached), fake)
            # running_losses['loss_d_real'].append(loss_d_real.item())
            # running_losses['loss_d_fake'].append(loss_d_fake.item())
            # loss_d = (loss_d_real + loss_d_fake) / 2
            # loss = loss + loss_d

            # Equation discovery
            loss = 0.0
            if use_latent:
                z, xhat = autoencoder(x)
                dz = autoencoder.compute_dz(x, dx)
                dz_pred = regressor(z)
                dx_pred = autoencoder.compute_dx(z, dz_pred)
                loss_sindy_z = w_sindy_z * sindy_loss(dz_pred, dz)
                loss_sindy_x = w_sindy_x * sindy_loss(dx_pred, dx)
                running_losses['loss_sindy_z'].append(loss_sindy_z.item() / max(w_sindy_z, 1e-6))
                running_losses['loss_sindy_x'].append(loss_sindy_x.item() / max(w_sindy_x, 1e-6))
                # symmetry regularization
                loss_sym_reg = 0.0
                for v in generator.get_full_basis_list():
                    loss_sym_reg += torch.norm(jvp(regressor, z, torch.einsum('ij, bj->bi', v, z)) - torch.einsum('ij, bj->bi', v, dz_pred)) ** 2
                running_losses['loss_sym_reg'].append(loss_sym_reg.item())
                loss_sym_reg = w_sym_reg * loss_sym_reg
                loss = loss + loss_sindy_z + loss_sindy_x + loss_sym_reg
            else:
                dx_pred = regressor(x)
                loss_sindy_x = sindy_loss(dx_pred, dx)
                running_losses['loss_sindy_x'].append(loss_sindy_x.item())
                running_losses['loss_sindy_z'].append(0.0)
                # symmetry regularization
                def forward_step(x):
                    return odeint(regressor, x, int_t, int_dt)
                fx_pred = forward_step(x)
                x_fx = torch.stack([x, fx_pred], dim=1)
                loss_sym_reg = symm_loss(x_fx, f=forward_step)
                running_losses['loss_sym_reg'].append(loss_sym_reg.item())
                loss = loss + w_sindy_x * loss_sindy_x + w_sym_reg * loss_sym_reg
            if sindy_reg_type == 'l1':
                loss_sindy_reg = sum([torch.norm(p, 1) for p in regressor.parameters()])
                running_losses['loss_sindy_reg'].append(loss_sindy_reg.item())
                loss = loss + w_sindy_reg * loss_sindy_reg
            else:
                raise ValueError(f'Unknown regularization type: {sindy_reg_type}')
                
            # Backprop
            # if not no_ae_flag:
            #     optimizer_ae.zero_grad()
            # optimizer_d.zero_grad()
            # optimizer_g.zero_grad()
            optimizer_sindy.zero_grad()
            loss.backward()
            # if not no_ae_flag:
            #     optimizer_ae.step()
            # optimizer_d.step()
            # optimizer_g.step()
            optimizer_sindy.step()

        # sequential thresholding
        # if gan_st_freq > 0 and (epoch + 1) % gan_st_freq == 0:
        #     generator.set_threshold(gan_st_thres)
        if st_freq > 0 and (epoch + 1) % st_freq == 0:
            regressor.set_threshold(threshold)
            # w_sindy_reg *= 0.5

        wandb_log = {
            k: np.mean(running_losses[k]) for k in log_items
        }
        if (epoch + 1) % log_interval == 0:
            print(', '.join([ f'Epoch {epoch}' ] + 
                [ f'{k}: {np.mean(running_losses[k]):.4f}' for k in log_items if log_flag[k] ]
            ))
            autoencoder.eval()
            discriminator.eval()
            generator.eval()
            with torch.no_grad():
                running_losses = { k: [] for k in log_items_test }
                for i, (x, dx) in enumerate(test_loader):
                    x = x.to(device)
                    dx = dx.to(device)
                    bs = x.shape[0]
                    # valid = torch.ones((bs, 1)).to(device)
                    # fake = torch.zeros((bs, 1)).to(device)
                    # z, xhat = autoencoder(x)
                    # zt = generator(z)
                    # xt = autoencoder.decode(zt)
                    # d_fake = discriminator(zt, None, xt if use_original_x else None)
                    # d_real = discriminator(z, None, x_gan if use_original_x else None)
                    # loss_ae = recon_loss(xhat, x_gan)
                    # loss_g = adversarial_loss(d_fake, valid)
                    # loss_d_real = adversarial_loss(d_real, valid)
                    # loss_d_fake = adversarial_loss(d_fake, fake)
                    # running_losses['test_loss_ae'].append(loss_ae.item())
                    # running_losses['test_loss_g'].append(loss_g.item())
                    # running_losses['test_loss_d_real'].append(loss_d_real.item())
                    # running_losses['test_loss_d_fake'].append(loss_d_fake.item())
                    if use_latent:
                        z, xhat = autoencoder(x)
                        dz = autoencoder.compute_dz(x, dx)
                        dz_pred = regressor(z)
                        dx_pred = autoencoder.compute_dx(z, dz_pred)
                        loss_sindy_z = sindy_loss(dz_pred, dz)
                        loss_sindy_x = sindy_loss(dx_pred, dx)
                        running_losses['test_loss_sindy_z'].append(loss_sindy_z.item())
                        running_losses['test_loss_sindy_x'].append(loss_sindy_x.item())
                    else:
                        dx_pred = regressor(x)
                        loss_sindy_x = sindy_loss(dx_pred, dx)
                        running_losses['test_loss_sindy_x'].append(loss_sindy_x.item())
                        running_losses['test_loss_sindy_z'].append(0.0)
                    
                wandb_log.update({
                    k: np.mean(running_losses[k]) for k in log_items_test
                })
                print(', '.join([ f'Epoch {epoch}' ] + 
                    [ f'{k}: {np.mean(running_losses[k]):.4f}' for k in log_items_test if log_flag_test[k] ]
                ))
                # if kwargs['print_li']:
                #     print(generator.getLi())
                if kwargs['print_eq']:
                    regressor.print()

        wandb.log(wandb_log)
        
        if (epoch + 1) % save_interval == 0:
            if not os.path.exists(f'saved_models/{save_dir}'):
                os.makedirs(f'saved_models/{save_dir}')
            # torch.save(autoencoder.state_dict(), f'saved_models/{save_dir}/autoencoder_{epoch}.pt')
            # torch.save(discriminator.state_dict(), f'saved_models/{save_dir}/discriminator_{epoch}.pt')
            # torch.save(generator.state_dict(), f'saved_models/{save_dir}/generator_{epoch}.pt')
            torch.save(regressor.state_dict(), f'saved_models/{save_dir}/regressor_{epoch}.pt')


def train_SIGED_lbfgs(
    train_loader, test_loader, num_epochs, device, log_interval, save_interval, save_dir,  # global
    autoencoder, generator,  # symmetry discovery model
    regressor, regressor_dst, use_latent, distill_latent, lr_sindy, w_sindy_z, w_sindy_x,  # SINDy
    sindy_reg_type, w_sindy_reg, sym_reg_type, w_sym_reg, st_freq, threshold, int_t, int_dt,  # SINDy
    **kwargs
):
    if distill_latent and not use_latent:
        raise ValueError('Cannot distill without first learning latent space equation. Set use_latent=True.')
    train_data = next(iter(train_loader))
    x, dx = train_data
    x = x.to(device)
    dx = dx.to(device)
    optimizer = torch.optim.LBFGS(regressor.parameters(), lr=lr_sindy)
    sindy_loss = torch.nn.MSELoss()
    if sym_reg_type == 'i':
        symm_loss = make_symmreg_pttrain(autoencoder, generator)
    elif sym_reg_type == 'f':
        symm_loss = make_fsymmreg_pttrain(autoencoder, generator)
    elif sym_reg_type == 'r':
        symm_loss = make_rsymmreg_pttrain(autoencoder, generator)
    autoencoder.eval()
    generator.eval()
    losses = {}
    prev_params = [p.detach().clone() for p in regressor.parameters()]
    pprev_params = [p.detach().clone() for p in regressor.parameters()]
    tol = 1e-3  # tolerance for LBFGS convergence

    def closure():
        optimizer.zero_grad()
        if use_latent:
            z, xhat = autoencoder(x)
            dz = autoencoder.compute_dz(x, dx)
            dz_pred = regressor(z)
            dx_pred = autoencoder.compute_dx(z, dz_pred)
            loss_sindy_z = sindy_loss(dz_pred, dz)
            loss_sindy_x = sindy_loss(dx_pred, dx)
            losses['loss_sindy_z'] = loss_sindy_z.item()
            losses['loss_sindy_x'] = loss_sindy_x.item()
            # symmetry regularization
            # loss_sym_reg = 0.0
            # for v in generator.get_full_basis_list():
            #     loss_sym_reg += torch.norm(jvp(regressor, z, torch.einsum('ij, bj->bi', v, z)) - torch.einsum('ij, bj->bi', v, dz_pred)) ** 2
            # losses['loss_sym_reg'] = loss_sym_reg.item()
            loss = w_sindy_z * loss_sindy_z + w_sindy_x * loss_sindy_x  # + w_sym_reg * loss_sym_reg
        else:
            dx_pred = regressor(x)
            loss_sindy_x = sindy_loss(dx_pred, dx)
            losses['loss_sindy_x'] = loss_sindy_x.item()
            # symmetry regularization
            if w_sym_reg > 0.0:
                if sym_reg_type in ['i', 'f']:
                    def forward_step(x):
                        return odeint(regressor, x, int_t, int_dt)
                    fx_pred = forward_step(x)
                    x_fx = torch.stack([x, fx_pred], dim=1)
                    loss_sym_reg = symm_loss(x_fx, f=forward_step)
                elif sym_reg_type == 'r':  # reversed symmetry loss
                    loss_sym_reg = symm_loss(x, h=regressor)
                losses['loss_sym_reg'] = loss_sym_reg.item()
            else:
                loss_sym_reg = 0.0
            loss = w_sindy_x * loss_sindy_x + w_sym_reg * loss_sym_reg
        if sindy_reg_type == 'l1':
            loss_sindy_reg = sum([torch.norm(p, 1) for p in regressor.parameters()])
            losses['loss_sindy_reg'] = loss_sindy_reg.item()
            loss = loss + w_sindy_reg * loss_sindy_reg
        elif sindy_reg_type == 'none':
            pass
        else:
            raise ValueError(f'Unknown regularization type: {sindy_reg_type}')
        
        loss.backward()
        return loss
    
    n_iters = 0
    for epoch in range(num_epochs):
        n_iters += 1
        optimizer.step(closure)
        # check for nan; occasionally happens with LBFGS
        if any(torch.isnan(p).any() for p in regressor.parameters()):
            print(f'NaN encountered at iteration {epoch}; exit training.')
            break
        wandb_log = deepcopy(losses)
        with torch.no_grad():
            param_update_norm = sum(
                torch.norm(p - p_prev) for p, p_prev in zip(regressor.parameters(), prev_params)
            )
        if param_update_norm < tol:
            param_update_norm_2 = sum(
                torch.norm(p - p_prev) for p, p_prev in zip(regressor.parameters(), pprev_params)
            )  # param update since last thresholding
            if param_update_norm_2 < tol:
                print(f'Final convergence reached at iteration {epoch}; exit training.')
                if not os.path.exists(f'saved_models/{save_dir}'):
                    os.makedirs(f'saved_models/{save_dir}')
                torch.save(regressor.state_dict(), f'saved_models/{save_dir}/regressor_{epoch}.pt')
                break
            n_iters = 0
            regressor.set_threshold(threshold)  # sequential thresholding
            optimizer = torch.optim.LBFGS(regressor.parameters(), lr=lr_sindy)  # reset optimizer
            pprev_params = [p.detach().clone() for p in regressor.parameters()]  # reset previous parameters
            print(f'Convergence reached at iteration {epoch}; apply parameter thresholding and reset optimizer.')
        elif st_freq > 0 and n_iters % st_freq == 0:
            n_iters = 0
            regressor.set_threshold(threshold)
            optimizer = torch.optim.LBFGS(regressor.parameters(), lr=lr_sindy)  # reset optimizer
            print(f'Max number of LBFGS iterations reached; apply parameter thresholding and reset optimizer.')
        prev_params = [p.detach().clone() for p in regressor.parameters()]
        
        if (epoch + 1) % log_interval == 0:
            print(', '.join([ f'Epoch {epoch}' ] + 
                [ f'{k}: {losses[k]:.4f}' for k in losses ]
            ))
            autoencoder.eval()
            generator.eval()
            running_losses = { k: [] for k in ['test_loss_sindy_z', 'test_loss_sindy_x'] }
            for i, (x_test, dx_test) in enumerate(test_loader):
                x_test = x_test.to(device)
                dx_test = dx_test.to(device)
                bs = x.shape[0]
                if use_latent:
                    z, xhat = autoencoder(x)
                    dz = autoencoder.compute_dz(x, dx)
                    dz_pred = regressor(z)
                    dx_pred = autoencoder.compute_dx(z, dz_pred)
                    loss_sindy_z = sindy_loss(dz_pred, dz)
                    loss_sindy_x = sindy_loss(dx_pred, dx)
                    running_losses['test_loss_sindy_z'].append(loss_sindy_z.item())
                    running_losses['test_loss_sindy_x'].append(loss_sindy_x.item())
                else:
                    dx_pred = regressor(x)
                    loss_sindy_x = sindy_loss(dx_pred, dx)
                    running_losses['test_loss_sindy_x'].append(loss_sindy_x.item())
                    running_losses['test_loss_sindy_z'].append(0.0)
            wandb_log.update({
                k: np.mean(running_losses[k]) for k in running_losses
            })
            print(', '.join([ f'Epoch {epoch}' ] + 
                [ f'{k}: {np.mean(running_losses[k]):.4f}' for k in running_losses ]
            ))
            if kwargs['print_eq']:
                regressor.print()

        wandb.log(wandb_log)

        if (epoch + 1) % save_interval == 0:
            if not os.path.exists(f'saved_models/{save_dir}'):
                os.makedirs(f'saved_models/{save_dir}')
            torch.save(regressor.state_dict(), f'saved_models/{save_dir}/regressor_{epoch}.pt')

    # (Optional) Phase 2: distill equation from latent to data space
    if not distill_latent:
        return
    print('\n=== Phase 2: distill equation from latent to data space ===\n')
    
    # Get dataset from latent equation
    x, _ = train_data
    x = x.to(device)
    with torch.no_grad():
        z, _ = autoencoder(x)
        dz_pred = regressor(z)
        dx = autoencoder.compute_dx(z, dz_pred)

    # Train a new regressor on data space
    optimizer = torch.optim.LBFGS(regressor_dst.parameters(), lr=lr_sindy)
    losses = {}
    prev_params = [p.detach().clone() for p in regressor_dst.parameters()]
    pprev_params = [p.detach().clone() for p in regressor_dst.parameters()]
    
    def closure_dst():
        optimizer.zero_grad()
        dx_pred = regressor_dst(x)
        loss_sindy_x = sindy_loss(dx_pred, dx)
        losses['loss_sindy_x'] = loss_sindy_x.item()
        loss = w_sindy_x * loss_sindy_x
        if sindy_reg_type == 'l1':
            loss_sindy_reg = sum([torch.norm(p, 1) for p in regressor_dst.parameters()])
            losses['loss_sindy_reg'] = loss_sindy_reg.item()
            loss = loss + w_sindy_reg * loss_sindy_reg
        elif sindy_reg_type == 'none':
            pass
        else:
            raise ValueError(f'Unknown regularization type: {sindy_reg_type}')
        
        loss.backward()
        return loss
    
    n_iters = 0
    for epoch in range(num_epochs):
        n_iters += 1
        optimizer.step(closure_dst)
        # check for nan; occasionally happens with LBFGS
        if any(torch.isnan(p).any() for p in regressor_dst.parameters()):
            print(f'NaN encountered at iteration {epoch}; exit training.')
            break
        wandb_log = deepcopy(losses)
        with torch.no_grad():
            param_update_norm = sum(
                torch.norm(p - p_prev) for p, p_prev in zip(regressor_dst.parameters(), prev_params)
            )
        if param_update_norm < tol:
            param_update_norm_2 = sum(
                torch.norm(p - p_prev) for p, p_prev in zip(regressor_dst.parameters(), pprev_params)
            )  # param update since last thresholding
            if param_update_norm_2 < tol:
                print(f'Final convergence reached at iteration {epoch}; exit training.')
                if not os.path.exists(f'saved_models/{save_dir}'):
                    os.makedirs(f'saved_models/{save_dir}')
                torch.save(regressor_dst.state_dict(), f'saved_models/{save_dir}/regressor_{epoch}.pt')
                break
            n_iters = 0
            regressor_dst.set_threshold(threshold)  # sequential thresholding
            optimizer = torch.optim.LBFGS(regressor_dst.parameters(), lr=lr_sindy)  # reset optimizer
            pprev_params = [p.detach().clone() for p in regressor_dst.parameters()]  # reset previous parameters
            print(f'Convergence reached at iteration {epoch}; apply parameter thresholding and reset optimizer.')
        elif st_freq > 0 and n_iters % st_freq == 0:
            n_iters = 0
            regressor_dst.set_threshold(threshold)
            optimizer = torch.optim.LBFGS(regressor_dst.parameters(), lr=lr_sindy)  # reset optimizer
            print(f'Max number of LBFGS iterations reached; apply parameter thresholding and reset optimizer.')
        prev_params = [p.detach().clone() for p in regressor_dst.parameters()]
        
        if (epoch + 1) % log_interval == 0:
            print(', '.join([ f'Epoch {epoch}' ] + 
                [ f'{k}: {losses[k]:.4f}' for k in losses ]
            ))
            if kwargs['print_eq']:
                regressor_dst.print()

        wandb.log(wandb_log)

        if (epoch + 1) % save_interval == 0:
            if not os.path.exists(f'saved_models/{save_dir}'):
                os.makedirs(f'saved_models/{save_dir}')
            torch.save(regressor_dst.state_dict(), f'saved_models/{save_dir}/regressor_{epoch}.pt')


def train_WSINDy(
    wrapper, train_x, num_epochs, device, log_interval, save_interval, save_dir,
    w_sindy_reg, threshold, **kwargs
):
    train_x = train_x.to(device)

    for epoch in range(num_epochs):
        residual, completed = wrapper.solve(train_x, w_sindy_reg, threshold)
        
        if (epoch + 1) % log_interval == 0:
            print(f'Iteration {epoch}, loss: {residual:.4f}')
            wrapper.regressor.print()
        if completed:
            print(f'Final convergence reached at iteration {epoch}; exit training.')
            break


def train_SINDy(
    regressor, x, dx, num_epochs, device, log_interval, save_interval, save_dir,
    w_sindy_reg, threshold, **kwargs
):
    x = x.to(device)
    dx = dx.to(device)

    for epoch in range(num_epochs):
        residual, completed = solve_SINDy_one_step(regressor, x, dx, w_sindy_reg, threshold)
        
        if (epoch + 1) % log_interval == 0:
            print(f'Iteration {epoch}, loss: {residual:.4f}')
            regressor.print()
        if completed:
            print(f'Final convergence reached at iteration {epoch}; exit training.')
            break
