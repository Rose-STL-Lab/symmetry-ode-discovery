import numpy as np
import pysr
pysr.julia_helpers.init_julia()
import torch
from pysr import PySRRegressor
import sympy as sp
from parser_utils import get_args
from utils import get_dataset
from gan import *
from autoencoder import *
from model_utils import precompute_symmreg_r


if __name__ == '__main__':
    args = get_args()

    # args to dict
    args = vars(args)

    # Load dataset
    train_dataset, val_dataset, args = get_dataset(args)
    x = train_dataset.x
    dx = train_dataset.dx
    subsample_size = int(len(x) * args['pysr_subsample'])

    # Load model if needed
    if args['pysr_symmreg']:
        # load model
        laligan_path = args['load_laligan']
        autoencoder = AutoEncoder(**args).to(args['device'])
        generator = LieGenerator(**args).to(args['device'])
        autoencoder.load_state_dict(torch.load(f'saved_models/{laligan_path}/autoencoder.pt'))
        try:
            saved_state_dict = torch.load(f'saved_models/{laligan_path}/generator.pt')
            generator.load_state_dict(saved_state_dict)
        except:  # fix compatbility issue in loading generator
            new_state_dict = generator.state_dict()
            for name, param in new_state_dict.items():
                if name not in saved_state_dict:
                    saved_state_dict[name] = param
            generator.load_state_dict(saved_state_dict)
        generator.masks = torch.load(f'saved_models/{laligan_path}/generator_mask.pt')
        generator.masks = [mask.to(args['device']) if mask is not None else None for mask in generator.masks]

    # repeat experiments
    x_sup = x.clone()
    dx_sup = dx.clone()
    for seed in range(100):
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        subsample_idx = np.random.choice(len(x), subsample_size, replace=False)
        x = x_sup[subsample_idx]
        dx = dx_sup[subsample_idx]

        xdim = x.shape[1]
        y = torch.zeros(x.shape[0])

        # Compute transformed data for reversed symmetry regularization
        if args['pysr_symmreg']:

            # compute g(x) and J_g(x)
            gx_list, Jgx_list = precompute_symmreg_r(x.to(args['device']), autoencoder, generator)
            gx_list = [gx.cpu() for gx in gx_list]
            Jgx_list = [Jgx.transpose(1, 2).flatten(1).cpu() for Jgx in Jgx_list]
            xdim = x.shape[1]
            groupdim = len(gx_list)
            x = torch.cat([x] + gx_list + Jgx_list, dim=1)  # (n_samples, n + d * n + d * n * n)
            x = torch.cat([x, dx], dim=1)
            eq_input_dim = x.shape[1]
            dx = torch.zeros(x.shape[0])
            w_sym_reg = args['w_sym_reg']

            # def symmreg objective in Julia code
            # this only works for 2D systems
            # as PySR doesn't naturally support considering multiple components simultaneously in the objective
            # here we manually split the tree into two components, representing both equations
            # the discovery result needs manual inspection for interpretation
            objective = f"""
            function symmreg(tree, dataset::Dataset{{T,L}}, options) where {{T,L}}
                # two components
                tree.degree != 2 && return L(Inf)

                h1 = tree.l
                h2 = tree.r

                # Split X into components
                X = dataset.X
                bs = size(X, 2)
                X1 = X[1 : {xdim}, :]  # State x
                X2 = X[{xdim + 1} : {xdim + groupdim * xdim}, :]  # Transformed state g(x)
                X3 = X[{xdim + groupdim * xdim + 1} : end - {xdim}, :]  # Jacobian J_g(x)
                X3 = reshape(X3, {xdim}, {xdim}, bs, {groupdim})  # Reshape to (n, n, bs, d)
                X3 = permutedims(X3, (1, 2, 4, 3))  # Permute to (n, n, d, bs)
                y = X[end - {xdim} + 1 : end, :]  # Derivative dx
                y = permutedims(y, (2, 1))  # Permute to (bs, n)

                # the tree must accept the inputs of the original dimension
                # genetic programming will trim out the terms for zero dimensions
                padded_x = zeros(Float32, size(X, 1), bs)
                padded_x[1 : {xdim}, :] = X1

                # prediction
                h1x, flag = eval_tree_array(h1, padded_x, options)
                h2x, flag = eval_tree_array(h2, padded_x, options)
                prediction = cat(h1x, h2x, dims=2)
                diffs = prediction .- y
                loss = sum(diffs .^ 2) / length(diffs)

                # symmreg
                for i in 1:{groupdim}
                    gx = X2[(i - 1) * {xdim} + 1 : i * {xdim}, :]
                    padded_gx = zeros(Float32, size(X, 1), bs)
                    padded_gx[1 : {xdim}, :] = gx

                    Jgx = X3[:, :, i, :]
                    h1gx, flag = eval_tree_array(h1, padded_gx, options)
                    h2gx, flag = eval_tree_array(h2, padded_gx, options)
                    hgx = cat(h1gx, h2gx, dims=2)
                    hgx = permutedims(hgx, (2, 1))

                    Jgxhx = zeros({xdim}, bs)
                    for j in 1:bs
                        Jgxhx[:, j] = Jgx[:, :, j] * hgx[:, j]
                    end
                    symm_diff = Jgxhx .- hgx
                    loss += {w_sym_reg} * sum(symm_diff .^ 2) / size(hgx, 2)
                end

                return loss
            end
            """

            config = {
                'lv': {
                    'model_selection': 'accuracy',
                    'niterations': 40,
                    'binary_operators': ["+", "*", "-"],
                    'unary_operators': ["exp"],
                    'loss': None,
                    'full_objective': objective,
                    'batching': False,
                    # batch_size=args['pysr_bs'],
                    'parsimony': 0.0016,
                    'maxsize': 25,
                    'temp_equation_file': True,
                    'tempdir': './pysr_temp'
                },
                'selkov': {
                    'model_selection': 'accuracy',
                    'niterations': 40,
                    'binary_operators': ["+", "*", "-"],
                    'unary_operators': [],
                    'loss': None,
                    'full_objective': objective,
                    'batching': False,
                    # batch_size=args['pysr_bs'],
                    'parsimony': 0.0016,
                    'maxsize': 40,
                    'maxdepth': 6,
                    'temp_equation_file': True,
                    'tempdir': './pysr_temp'
                },
            }


            model = PySRRegressor(**config[args['task']])

        else:
            loss = "loss(prediction, target) = (prediction - target)^2"

            config = {
                'lv': {
                    'niterations': 40,
                    'binary_operators': ["+", "*", "-"],
                    'unary_operators': ["exp"],
                    'loss': loss,
                    'full_objective': None,
                    'temp_equation_file': True,
                    'tempdir': './pysr_temp'
                },
                'selkov': {
                    'niterations': 40,
                    'binary_operators': ["+", "*", "-"],
                    'unary_operators': [],
                    'loss': loss,
                    'full_objective': None,
                    'temp_equation_file': True,
                    'tempdir': './pysr_temp'
                },
                'dosc': {
                    'niterations': 40,
                    'binary_operators': ["+", "*", "-"],
                    'unary_operators': [],
                    'loss': loss,
                    'full_objective': None,
                    'temp_equation_file': True,
                    'tempdir': './pysr_temp'
                },
                'growth': {
                    'niterations': 40,
                    'binary_operators': ["+", "*", "-"],
                    'unary_operators': [],
                    'loss': loss,
                    'full_objective': None,
                    'temp_equation_file': True,
                    'tempdir': './pysr_temp'
                }
            }

            model = PySRRegressor(**config[args['task']])

        model.fit(x, dx)

        if not os.path.exists(f'saved_models/{args["save_dir"]}'):
            os.makedirs(f'saved_models/{args["save_dir"]}')

        # print(model)
        equation = model.get_best()  # pandas.Series

        # should be only one equation
        if args['pysr_symmreg']:
            eq = equation['sympy_format']
            # to sympy
            eq = sp.sympify(eq)
            print(eq)
            # evaluate at x = 0
            eq = eq.subs({sp.Symbol(f'x{i}'): 0 for i in range(xdim, eq_input_dim)})
            # eq = sp.simplify(eq)
            # print(eq)

            with open(f'saved_models/{args["save_dir"]}/equation_seed{seed}.txt', 'w') as f:
                f.write(str(eq))

        else:
            # get each equation in the series
            equation_str = [eq['sympy_format'] for eq in equation]
            for i in range(len(equation)):
                print(equation_str[i])

            with open(f'saved_models/{args["save_dir"]}/equations_seed{seed}.txt', 'w') as f:
                f.write('\n'.join([str(eq) for eq in equation_str]))
