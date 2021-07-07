#globals().clear()
import torch
import time
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
    np.sqrt(1.5)**2
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
                                  np.sqrt(1.5)**2 * get_V(0.5), 10000)
# 转化成张量
t = torch.tensor(t, requires_grad=False)
X = torch.tensor(X, requires_grad=False)

def get_AR1(sigma,phi):
    d = len(t)
    base = torch.tensor(toeplitz(range(0, d), range(0, d))).double()
    V = 1 / (1 - phi ** 2) * phi ** base
    SIGMA = sigma**2 * V
    return SIGMA

def get_mu_model(t, a, b, r):
    mu = a/(1 + b * torch.exp(-r * t))
    return mu

def mle(t,X,params):
    a,b,r = params[0:3]
    sigma, phi = params[-2],params[-1]
    mu = get_mu_model(t, a, b, r)
    cov = get_AR1(sigma,phi)
    dist = torch.distributions.MultivariateNormal(mu, cov)
    LogLikelihood = -torch.sum(dist.log_prob(X))
    return LogLikelihood

params = torch.tensor([5.0, 2.0, 0.1, 5.0, 0.01], requires_grad=True)

# 定义模型优化器
optimizer = torch.optim.Adam([params], lr=1e-1)

def training_loop(n_epochs, optimizer, params, X, t):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:  # <1>
            params.grad.zero_()
        LL = mle(t,X,params)
        optimizer.zero_grad()
        LL.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print('Epoch %d, LogLikelihood %f' % (epoch, float(LL)))
    return params
torch.cuda.synchronize()
start = time.time()
params_hat = training_loop(
    n_epochs = 200,
    optimizer = optimizer,
    params = params,
    X = X,
    t = t)
torch.cuda.synchronize()
end = time.time()
print("time_elipse=", format(end-start))
print("par: {}, par_hat: {}".format([10,3,0.5,np.sqrt(1.5),0.5], params))
