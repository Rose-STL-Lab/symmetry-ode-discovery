import torch
import torch.nn as nn
import numpy as np
import sympy as sp
import math

def SINDyConst(x):
    return torch.ones(*x.shape[:-1], 1, device=x.device)

def SINDyPoly1(x):
    return x

def SINDyPoly2(x):
    return torch.cat([(x[..., i] * x[..., j]).view(*x.shape[:-1], 1) 
                       for i in range(x.shape[-1]) 
                       for j in range(i, x.shape[-1])], 
                       dim = -1)

def SINDyPoly3(x):
    return torch.cat([(x[..., i] * x[..., j] * x[..., k]).view(*x.shape[:-1], 1) 
                       for i in range(x.shape[-1]) 
                       for j in range(i, x.shape[-1])
                       for k in range(j, x.shape[-1])], 
                       dim = -1)

def SINDySine(x):
    return torch.sin(x)

def SINDyExp(x):
    return torch.exp(x)


class SINDyRegression(nn.Module):
    """
    Arguments:
        latent_dim: dimension of latent space
        poly_order: highest order of polynomial terms, max=3
        include_sine: whether to include sine terms
        L_list: list of Lie algebra generators
    """

    def __init__(self, latent_dim, poly_order, include_sine, include_exp, L_list=[], **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.constraint = (len(L_list)!=0)
        self.include_sine = include_sine and not self.constraint
        self.include_exp = include_exp and not self.constraint
        self.L_list = L_list
        self.terms = []
        self.threshold = kwargs["threshold"]

        # SINDy with constraint
        if self.constraint:
            print('Computing equivariance constraint...')
            # self.M_list = self.get_M_list()
            self.Q = self.get_Q().to(kwargs["device"])
            self.beta = nn.Parameter(torch.randn((self.Q.shape[1]), device=kwargs["device"]))
            self.const = nn.Parameter(torch.randn((latent_dim, 1), device=kwargs["device"]))
            self.allow_constant = not kwargs['constrain_constant']
            self.Xi = self.get_Xi()
        # SINDy without constraint
        else:
            self.Xi = nn.Parameter(torch.randn(self.latent_dim, self.get_term_num(), device=kwargs["device"]))
        # Mask of \Xi
        self.mask = torch.ones_like(self.Xi, device=kwargs["device"])
        # Fuction basis
        self.terms.append(SINDyConst)
        self.terms.append(SINDyPoly1)
        if poly_order > 1:
            self.terms.append(SINDyPoly2)
        if poly_order > 2:
            self.terms.append(SINDyPoly3)
        if self.include_sine:
            self.terms.append(SINDySine)
        if self.include_exp:
            self.terms.append(SINDyExp)

    def forward(self, x):
        self.Xi = self.get_Xi() if self.constraint else self.Xi
        x = torch.cat([module(x) for module in self.terms], dim=-1)
        return x @ (self.Xi * self.mask).T
    
    # Calculate Q, whose column space forms the null space of C
    def get_Q(self):
        M_list = self.get_M_list()
        C_list = []
        for i in range(len(M_list)):
            # check if L is invertible
            if torch.det(self.L_list[i]) < 1e-5:
                self.use_kron_product = False
                MT, L = M_list[i].transpose(0, 1), self.L_list[i]
                C = torch.kron(-MT.contiguous(), torch.eye(L.shape[0])) + torch.kron(torch.eye(MT.shape[0]), L)
            else:  # when L is invertible, this somehow leads to better stability in equation discovery
                self.use_kron_product = True
                C = torch.kron(self.L_list[i].inverse(), M_list[i].T)
                C = C - torch.eye(C.shape[0])
            C_list.append(C)
        C_total = torch.cat(C_list, dim=0)
        U, Sigma, V = torch.svd(C_total)
        # Calculate r (rank of null space)
        for r in range(len(Sigma)):
            if abs(Sigma[-1 - r]) > 5e-3:
                break
        # Extract Q
        Q = V[:, -r:]
        
        # Print constraint information
        # print(f'M_list={M_list}')
        # print(f'C_total={C_total}')
        # print(f'Q={Q}')
        # print(f'Sigma={Sigma}')
        # print(f'Number of free parameters (excluding constant terms) under equivariance constraint: {Q.shape[1]}')
        
        return Q

    def update_Q(self, new_Li):
        self.L_list = new_Li
        self.Q = self.get_Q().to(self.Xi.device)
        self.beta = nn.Parameter(torch.randn((self.Q.shape[1]), device=self.Xi.device))

    # Calculate symbolic map M
    def get_M_list(self):
        # Create variables z0~zn-1
        z = sp.Matrix([sp.symbols(f"z{i}") for i in range(self.latent_dim)])
        # Calculate function basis library \Theta
        Theta = self.get_Theta()
        # Calculate Jacobian matrix of \Theta
        Jacobian_Theta = Theta.jacobian(z)
        # Calculate J*L*z, e.g. M*Theta
        M_temp = [Jacobian_Theta*sp.Matrix(Li)*z for Li in self.L_list]
        # Calculate M
        p = M_temp[0].shape[0]
        M_list = [torch.zeros(p, p) for i in range(len(self.L_list))]
        for i in range(len(self.L_list)):
            for j in range(p):
                expression = M_temp[i][j].expand()
                # Calculate constant term
                M_list[i][j, 0] = float(expression.subs({zi: 0 for zi in z}))
                # Calculate other terms
                for k in range(1, p):
                    # Extract coeff, using subs(z=0) to avoid bug in coeff()
                    M_list[i][j, k] = float(expression.coeff(Theta[k]).subs({zi: 0 for zi in z}))
        return M_list
    
    # Calculate function basis library \Theta
    def get_Theta(self):
        # Create variables z_0~z_n-1
        z = [sp.symbols(f"z{i}") for i in range(self.latent_dim)]
        # Poly0
        Theta = sp.Matrix([1])
        # Poly1
        for i in range(self.latent_dim):
            Theta = sp.Matrix.vstack(Theta, sp.Matrix([f"z{i}"]))
        # Poly2
        if self.poly_order > 1:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    Theta = sp.Matrix.vstack(Theta, sp.Matrix([f"z{i}*z{j}"]))
        # Poly3
        if self.poly_order > 2:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    for k in range(j, self.latent_dim):
                        Theta = sp.Matrix.vstack(Theta, sp.Matrix([f"z{i}*z{j}*z{k}"]))
        return Theta
    
    # Convert bata and const to Xi matrix
    def get_Xi(self):
        if self.use_kron_product:
            Xi = (self.Q @ self.beta).view(self.latent_dim, -1)
        else:
            Xi = (self.Q @ self.beta).view(-1, self.latent_dim).transpose(0, 1)
        if self.allow_constant:
            Xi += torch.cat([self.const, torch.zeros((Xi.shape[0], Xi.shape[1]-1), device=Xi.device)], dim=1)
        return Xi

    # Get the total number of function basis
    def get_term_num(self):
        num = self.latent_dim + 1
        if self.poly_order > 1:
            num += self.latent_dim * (self.latent_dim + 1) / 2
        if self.poly_order > 2:
            num += (self.latent_dim**3 + 3*self.latent_dim**2 + 2*self.latent_dim) / 6
        if self.include_sine:
            num += self.latent_dim
        if self.include_exp:
            num += self.latent_dim
        return int(num)

    # Update mask
    def set_threshold(self, threshold):
        self.Xi = self.get_Xi() if self.constraint else self.Xi
        self.mask.data = torch.logical_and(torch.abs(self.Xi) > threshold, self.mask).float()
        # self.mask.data = (torch.abs(self.Xi) > threshold).float()

    def reset_mask(self):
        self.mask.data = torch.ones_like(self.Xi, device=self.Xi.device)

    # Get function library w/o coefficients
    def eval_Theta_at(self, x):
        x = torch.cat([module(x) for module in self.terms], dim=-1)
        return x
    
    # Print equations
    def print(self):
        Xi = self.get_Xi() if self.constraint else self.Xi
        for i in range(self.latent_dim):
            pos = 0
            equation = f'dz{i} ='
            # Constant term
            if self.mask[i, pos]:
                equation += f' {Xi[i, pos]:.3f} +'
            pos += 1
            # Poly1 terms
            for j in range(self.latent_dim):
                if self.mask[i, pos]:
                    equation += f' {Xi[i, pos]:.3f}*z{j} +'
                pos += 1
            # Poly2 terms
            if self.poly_order > 1:
                for j in range(self.latent_dim):
                    for k in range(j, self.latent_dim):
                        if self.mask[i, pos]:
                            equation += f' {Xi[i, pos]:.3f}*z{j}*z{k} +'
                        pos += 1
            # Poly3 terms
            if self.poly_order > 2:
                for j in range(self.latent_dim):
                    for k in range(j, self.latent_dim):
                        for l in range(k, self.latent_dim):
                            if self.mask[i ,pos]:
                                equation += f' {Xi[i, pos]:.3f}*z{j}*z{k}*z{l} +'
                            pos += 1
            # Sin terms
            if self.include_sine:
                for j in range(self.latent_dim):
                    if self.mask[i, pos]:
                        equation += f' {Xi[i, pos]:.3f}*sin(z{j}) +'
                    pos += 1
            # Exp terms
            if self.include_exp:
                for j in range(self.latent_dim):
                    if self.mask[i, pos]:
                        equation += f' {Xi[i, pos]:.3f}*exp(z{j}) +'
                    pos += 1
            print(equation)


def solve_SINDy_one_step(regressor, x, y, w_sindy_reg, st_threshold, **kwargs):
    '''
    Solve the SINDy optimization problem with given data (x, y):
        argmin_{w} ||y - w @ Theta(x)||_2^2 + w_sindy_reg * ||w||_2^2

    Arguments:
        x & y: data of shape (n_samples, dim);
        w_sindy_reg: regularization weight; only support L2 regularization for now
        st_threshold: sparsity threshold
    '''
    
    theta_x = regressor.eval_Theta_at(x)
    gamma_I = w_sindy_reg * torch.eye(theta_x.shape[1], device=x.device)
    A = torch.cat([theta_x, gamma_I], dim=0)
    B = torch.cat([y, torch.zeros(theta_x.shape[1], y.shape[1], device=y.device)], dim=0)

    # flatten and apply thresholding
    mask = regressor.mask
    mask = mask > 0.0
    if (not torch.all(mask)) or regressor.constraint:
        A_ = A.clone()
        for _ in range(y.shape[-1]-1):
            A = torch.block_diag(A, A_)
        A = A[:, mask.flatten()]
        B = B.transpose(0, 1).reshape(-1)
        if regressor.constraint:
            Q = regressor.Q
            if regressor.allow_constant:
                Q = torch.cat([Q, torch.zeros((Q.shape[0], regressor.latent_dim), device=Q.device)], dim=1)
                for i in range(regressor.latent_dim):
                    Q[i * Q.shape[0] // regressor.latent_dim, Q.shape[1] - regressor.latent_dim + i] = 1.0
            # w = Q @ beta => Aw = A @ Q @ beta
            A = A @ Q[mask.flatten()]
            # avoid zero column in A
            effective_param = torch.any(A != 0.0, dim=0)
            A = A[:, effective_param]

    # solve the regularized lstsq problem
    lm = torch.linalg.lstsq(A, B)
    solution = lm.solution
    residual = lm.residuals

    # update parameters
    prev_mask = regressor.mask.clone()
    if not regressor.constraint:
        if not torch.all(mask):
            new_coef = torch.zeros_like(regressor.Xi, device=regressor.Xi.device)
            new_coef[mask] = solution
            regressor.Xi.data = new_coef
        else:
            regressor.Xi.data = solution.T
    else:
        if not regressor.allow_constant:
            new_beta = torch.zeros_like(regressor.beta, device=regressor.beta.device)
            new_beta[effective_param] = solution
            regressor.beta.data = new_beta
        else:
            # split solution into beta and const
            new_solution = torch.zeros(regressor.beta.shape[0] + regressor.latent_dim, device=regressor.beta.device)
            new_solution[effective_param] = solution
            regressor.beta.data = new_solution[:-regressor.latent_dim]
            regressor.const.data = new_solution[-regressor.latent_dim:].view(-1, 1)
    regressor.set_threshold(st_threshold)
    converged = torch.allclose(prev_mask, regressor.mask)

    return residual.mean() / x.shape[0], converged


def solve_SINDy(regressor, x, y, w_sindy_reg, st_threshold, max_iter=5, **kwargs):
    regressor.reset_mask()
    for _ in range(max_iter):
        residual, converged = solve_SINDy_one_step(regressor, x, y, w_sindy_reg, st_threshold)
        if converged:
            break
    return residual


class WSINDyWrapper():
    """
    Wrapper for solving Weak SINDy as a regularized least square problem.
    """

    def __init__(self, regressor, t, t_max, num_test_funcs=50, test_func_family='trig', device='cuda', **kwargs):
        self.t = t.to(device)
        self.dt = self.t[1] - self.t[0]
        self.regressor = regressor
        if test_func_family == 'trig':
            # for k in range(num_test_funcs), compute g_k(t) = sin(k * pi * t / t_max)
            # and g_k'(t) = k * pi / t_max * cos(k * pi * t / t_max)
            k = torch.arange(1, num_test_funcs + 1, dtype=torch.float32, device=device)
            k = k.view(-1, 1)
            g_k_t = math.sqrt(2 / t_max) * torch.sin(k * torch.pi * self.t / t_max)
            g_k_t_drv = math.sqrt(2 / t_max) * k * np.pi / t_max * torch.cos(k * np.pi * self.t / t_max)
        else:
            raise NotImplementedError(f'test_func_family={test_func_family} not implemented')
        # construct integration matrix nd covariance matrix
        self.V = self.dt * g_k_t
        self.V_drv = self.dt * g_k_t_drv
        self.sigma = self.V_drv @ self.V_drv.T
        self.sigma_inv = torch.inverse(self.sigma)
        self.sqrt_sigma_inv = torch.sqrt(self.sigma_inv)

    def solve(self, x, w_sindy_reg, st_threshold, **kwargs):
        '''
        Solve the weak SINDy optimization problem with given data x.
        Arguments:
            x: data of shape (seq_len, dim);
               time interval is assumed to be uniform and should match the one used to construct the wrapper
            w_sindy_reg: regularization weight; only support L2 regularization for now
            st_threshold: sparsity threshold
        '''

        # compute Gram matrix and rhs
        with torch.no_grad():
            G = self.V @ self.regressor.eval_Theta_at(x)
            b = -self.V_drv @ x
            data_dim = x.shape[-1]
            # prepare the augmented matrix and vector
            sqrt_gamma_I = math.sqrt(w_sindy_reg) * torch.eye(G.shape[1], device=G.device)
            G_aug = torch.cat([self.V.T @ G, sqrt_gamma_I], dim=0)
            b_aug = torch.cat([self.V.T @ b, torch.zeros(G.shape[1], b.shape[1], device=b.device)], dim=0)
            # # flatten and apply existing threshold
            mask = self.regressor.mask
            mask = mask > 0.0
            if not torch.all(mask):
                G_aug_ = G_aug.clone()
                for _ in range(data_dim-1):
                    G_aug = torch.block_diag(G_aug, G_aug_)
                G_aug = G_aug[:, mask.flatten()]
                b_aug = b_aug.transpose(0, 1).reshape(-1)
            # solve the regularized lstsq problem
            lm = torch.linalg.lstsq(G_aug, b_aug)
            solution = lm.solution
            residual = lm.residuals
            # update parameters
            prev_mask = self.regressor.mask.clone()
            if not torch.all(mask):
                new_coef = torch.zeros_like(self.regressor.Xi, device=self.regressor.Xi.device)
                new_coef[mask] = solution
                self.regressor.Xi.data = new_coef
            else:
                self.regressor.Xi.data = solution.T
            self.regressor.set_threshold(st_threshold)
            converged = torch.allclose(prev_mask, self.regressor.mask)

        return residual.mean().item(), converged
