import numpy as np
from scipy.optimize import line_search, bracket
from scipy.optimize import minimize


def get_rbf_kernel(t, sigma_out, sigma_in, t2=None):
    tc = t[:, None]
    if t2 is None:
        tr = t
    else:
        tr = t2
    t_mat = sigma_out ** 2 * np.exp(-1. / (2 * sigma_in ** 2) * (tc - tr) ** 2)
    return t_mat


# Obtained from D-CODE
class GPPCA0:
    def __init__(self, r, Y, t, sigma, sigma_out=None, sigma_in=None):
        self.r = r
        self.Y = Y
        self.t = t
        self.sigma = sigma
        self.n_traj = Y.shape[1]

        if sigma_out is None:
            self.sigma_out = np.std(Y)
        else:
            self.sigma_out = sigma_out

        if sigma_in is None:
            self.sigma_in = self.t[1] - self.t[0]
        else:
            self.sigma_in = sigma_in

        self.K = self.get_kernel_discrete()
        self.A = self.get_factor_loading()

    def get_hyper_param(self, method='Powell'):
        x0 = np.log(np.array([self.sigma_in]))
        res = minimize(self.loss_fn, x0=x0, method=method)
        # print(res)
        self.sigma_in = np.exp(res['x'][0])

    def loss_fn(self, x):
        # input in log scale
        sigma_out = self.sigma_out
        sigma_in = x[0]
        sigma_in = np.exp(sigma_in)

        tau = sigma_out ** 2 / self.sigma ** 2

        K = get_rbf_kernel(self.t, sigma_out, sigma_in)

        # T, T
        W = np.linalg.inv(1. / tau * np.linalg.inv(K) + np.eye(K.shape[0]))
        # R, T
        b = np.matmul(self.Y, self.A).T

        S = np.abs(np.sum(self.Y ** 2) - np.sum(np.diag(b @ W @ b.T)))

        f2 = np.log(S) * (-1 * self.Y.shape[0] * self.Y.shape[1] / 2)

        f1 = -1. / 2 * self.r * np.linalg.slogdet(tau * K + np.eye(K.shape[0]))[1]

        return -1 * (f1 + f2)

    def get_predictive(self, new_sample=1, t_new=None):
        if new_sample == 1:
            return self.get_predictive_mean(1, t_new)
        # predictive distribution
        Z_hat = self.get_factor(t_new=t_new)
        X_hat = self.get_X_mean(Z_hat)
        K_fac = self.get_X_cov(t_new=t_new)

        X_list = list()
        for i in range(new_sample):
            noise = self.sample_noise(K_fac)
            X_sample = X_hat + noise
            X_sample = X_sample[:, None, :]
            X_list.append(X_sample)

        X = np.concatenate(X_list, axis=1)
        return X

    def get_predictive_mean(self, new_sample=1, t_new=None):
        assert new_sample == 1
        # predictive distribution
        Z_hat = self.get_factor(t_new=t_new)
        X_hat = self.get_X_mean(Z_hat)

        return X_hat

    def get_factor_loading(self):
        G = self.get_G()
        w, v = np.linalg.eigh(G)
        A = v[:, -self.r:]
        return A

    def get_X_cov(self, t_new=None):
        if t_new is None:
            K = self.K
        else:
            K = get_rbf_kernel(t_new, self.sigma_out, self.sigma_in, t_new)

        # T, D
        D = self.sigma ** 2 * np.linalg.inv(1. / self.sigma ** 2 * np.linalg.inv(K) + np.eye(K.shape[0]))
        try:
            D_fac = np.linalg.cholesky(D)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(D)
            w_pos = w
            w_pos[w_pos < 0] = 0
            w_pos = np.sqrt(w_pos)
            D_fac = v @ np.diag(w_pos)

        # B, R
        A_fac = self.A
        K_fac = np.kron(D_fac, A_fac)
        return K_fac

    def sample_noise(self, K_fac):
        noise = np.random.randn(K_fac.shape[1])
        vec = K_fac @ noise
        mat = vec.reshape(len(vec) // self.n_traj, self.n_traj)
        return mat

    def get_factor(self, t_new=None):
        if t_new is None:
            f1 = self.K
        else:
            f1 = get_rbf_kernel(t_new, self.sigma_out, self.sigma_in, self.t)

        # print(f1.shape)
        # print(self.K.shape)
        Z_hat = f1 @ np.linalg.inv(self.K + self.sigma ** 2 * np.eye(self.K.shape[0])) @ self.Y @ self.A
        return Z_hat

    def get_X_mean(self, Z_hat):
        X_hat = np.matmul(Z_hat, self.A.T)
        return X_hat

    def get_kernel_discrete(self):
        sigma_out = self.sigma_out
        sigma_in = self.sigma_in
        K = get_rbf_kernel(self.t, sigma_out, sigma_in)
        return K

    def get_G(self):
        # get G matrix - Thm 3 Eq 7
        W = np.linalg.inv(self.sigma ** 2 * np.linalg.inv(self.K) + np.eye(self.K.shape[0]))
        G = np.matmul(np.matmul(self.Y.transpose(), W), self.Y)
        return G


def num_diff_gp(x, dt, noise_level, std_base, sigma_in=None):
    '''
    Compute the numerical gradient of noisy x with respect to time
    using Gaussian process regression.
    
    Arguments:
        x: (seq_len, n_trajs, input_dim)
        dt: time step
        noise_level: overall noise level
        std_base: (input_dim,): std of each dimension
    
    Returns:
        x: (seq_len, n_trajs, input_dim)
        dxdt: (seq_len, n_trajs, input_dim)
    '''
    seq_len, n_trajs, input_dim = x.shape
    t = np.arange(seq_len) * dt

    X_sample_list = []
    X_sample_list2 = []
    pca_list = []

    for d in range(input_dim):
        Y = x[:, :, d]
        r = Y.shape[1]
        noise_sigma = noise_level * std_base[d]
        pca = GPPCA0(r, Y, t, noise_sigma, sigma_out=std_base[d], sigma_in=sigma_in)

        X_sample = pca.get_predictive(new_sample=1, t_new=t)
        X_sample = X_sample.reshape(len(t), X_sample.size // len(t), 1)
        X_sample_list.append(X_sample)

        X_sample2 = pca.get_predictive(new_sample=1, t_new=t+0.001)
        X_sample2 = X_sample2.reshape(len(t), X_sample2.size // len(t), 1)
        X_sample_list2.append(X_sample2)

        pca_list.append(pca)

    X_sample = np.concatenate(X_sample_list, axis=-1)
    X_sample2 = np.concatenate(X_sample_list2, axis=-1)
    dX = (X_sample2 - X_sample) / 0.001
    return dX, X_sample
