import numpy as np
import sympy as sp
import torch
import os


def eval_sindy_regressor(regressor, truth, threshold=0.05):
    """
    Evaluate SINDy regressor
    :param regressor: SINDy regressor
    :param truth: ground truth
    :return: evaluation metrics
    """
    # Get the equation
    with torch.no_grad():
        coef = regressor.get_Xi() if regressor.constraint else regressor.Xi
        mask = regressor.mask
        coef = coef.cpu().numpy()
        mask = mask.bool().cpu().numpy()
        coef = np.where(mask, coef, 0.0)
        truth_mask = truth != 0
        n_eqs, n_terms = coef.shape
        correct_form = np.zeros(n_eqs)
        mse = np.ones(n_eqs) * -1.0
        for i in range(n_eqs):
            correct_form[i] = np.all(mask[i, :] == truth_mask[i, :])
            # if correct_form[i]:
            #     mse[i] = np.mean((coef[i, mask[i, :]] - truth[i, mask[i, :]]) ** 2)
            mse[i] = np.mean((coef[i, truth_mask[i, :]] - truth[i, truth_mask[i, :]]) ** 2)
        correct_form_all = np.all(correct_form)
        # mse_all = np.mean(mse) if correct_form_all else -1.0
        mse_all = np.mean(mse)

    return coef, correct_form, mse, correct_form_all, mse_all

result_dir = 'eval_results'

def aggregate_results(run_name, min_seed=0, max_seed=100, mse_multiplier=1.0):
    directory = os.path.join(result_dir, run_name)
    cf, mse, cf_all, mse_all = [], [], [], []
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            file_path = os.path.join(directory, filename)
            seed = int(filename.split('.')[0][4:])
            if seed >= max_seed or seed < min_seed:
                continue
            res = np.load(file_path)
            cf.append(res['correct_form'])
            mse.append(res['mse'])
            cf_all.append(res['correct_form_all'])
            mse_all.append(res['mse_all'])
    print(f'Loaded results from {len(cf)} runs.')
    # Correct form for each equation
    cf = np.stack(cf)
    cf_sum = np.sum(cf, axis=0).astype(int)
    for i, each in enumerate(cf_sum):
        print(f'Equation {i} success rate = {cf_sum[i]}/{cf.shape[0]}')
    # Correct form for all equations
    cf_all_sum = np.sum(cf_all).astype(int)
    print(f'Joint success rate = {cf_all_sum}/{cf.shape[0]}')
    # RMSE for each equation
    mse = np.stack(np.sqrt(mse))
    for i in range(mse.shape[1]):
        mse_valid = np.mean(mse[np.where(cf[:, i]), i])
        std = np.std(mse[np.where(cf[:, i]), i])
        mse_any = np.mean(mse[:, i])
        std_any = np.std(mse[:, i])
        mse_valid *= mse_multiplier
        std *= mse_multiplier
        mse_any *= mse_multiplier
        std_any *= mse_multiplier
        print(f'Equation {i} RMSE = {mse_valid:.4f} ({std:.4f})')
        print(f'Equation {i} RMSE (any) = {mse_any:.4f} ({std_any:.4f})')
    # MSE for all equations
    mse_all = np.stack(np.sqrt(mse_all))
    mse_all_valid = np.mean(mse_all[np.where(cf_all)])
    std = np.std(mse_all[np.where(cf_all)])
    mse_all_any = np.mean(mse_all)
    std_any = np.std(mse_all)
    mse_all_valid *= mse_multiplier
    std *= mse_multiplier
    mse_all_any *= mse_multiplier
    std_any *= mse_multiplier
    print(f'All equations RMSE = {mse_all_valid:.4f} ({std:.4f})')
    print(f'All equations RMSE (any) = {mse_all_any:.4f} ({std_any:.4f})')

# Note that this is dependent on SINDy parametrization, i.e. which terms are included in the basis
sindy_truth = {
    'lv': np.array([
        [2/3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4/3,],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,],
    ]),
    'selkov': np.array([
        [0.75, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.1, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]),
    'dosc': np.array([
        [0.0, -0.1, -1, 0.0, 0.0, 0.0],
        [0.0, 1, -0.1, 0.0, 0.0, 0.0],
    ]),
    'growth': np.array([
        [0.0, -0.3, 0.0, 0.0, 0.0, 0.1],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ])
}
