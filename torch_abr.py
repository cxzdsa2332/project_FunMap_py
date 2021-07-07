import torch
import numpy as np
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt
# 数据
def get_V(phi):
    d = len(t)
    V = 1/(1-phi**2) * phi**toeplitz(range(0, d), range(0, d))
    return V

def get_miu(par):
    d = len(t)
    miu = par[0]/(1 + par[1] * np.exp(-par[2] * np.linspace(1,d,d)))
    return miu

def get_inv_V(phi):
    d = len(t)
    inv_V = toeplitz(np.append([1,-phi],np.repeat(0,d-2)), \
                     np.append([1,-phi],np.repeat(0,d-2)))
    return inv_V

def get_det_V(phi):
    d = len(t)
    det_V = 1/(1-phi**2)
    return det_V

t = np.arange(1,11)
np.random.default_rng(2021)
X = np.random.multivariate_normal(get_miu([10,3,0.5]),
                                  np.sqrt(1.5)**2 * get_V(0.5), 3000)
# 转化成张量
t = torch.tensor(t, requires_grad=False)
X = torch.tensor(X, requires_grad=False)

def get_cov(m, rowvar=False, inplace=False):
    '''Estimate a covariance matrix given data.

    Thanks :
    - https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
    - https://github.com/numpy/numpy/blob/master/numpy/lib/function_base.py#L2276-L2494

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    m = m.type(torch.double)
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else :
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def get_mu_model(t, a, b, r):
    mu = a/(1 + b * torch.exp(-r * t))
    return mu

def mle(t,X,params):
    mu = get_mu_model(t, *params)
    cov = get_cov(X)
    dist = torch.distributions.MultivariateNormal(mu, cov)
    LogLikelihood = -torch.sum(dist.log_prob(X))
    return LogLikelihood

params = torch.tensor([5.0, 2.0, 0.1], requires_grad=True)
print(params)

def training_loop(n_epochs, learning_rate, params, X, t):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:  # <1>
            params.grad.zero_()
        LL = mle(t,X,params)
        LL.backward()
        params = (params - learning_rate * params.grad).detach().requires_grad_()
        if epoch % 100 == 0:
            print('Epoch %d, LogLikelihood %f' % (epoch, float(LL)))
    return params

params_hat = training_loop(
    n_epochs = 1000,
    learning_rate = 1e-5,
    params = torch.tensor([5.0, 2.0, 0.1], requires_grad=True),
    X = X,
    t = t)
print(params_hat)
