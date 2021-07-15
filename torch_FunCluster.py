#globals().clear()
import torch
import time
import numpy as np
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# 数据
def get_V(phi):
    d = len(t)
    V = 1/(1-phi**2) * phi**toeplitz(range(0, d), range(0, d))
    return V

def get_mu(par):
    d = len(t)
    mu = par[0]/(1 + par[1] * np.exp(-par[2] * np.linspace(1,d,d)))
    return mu

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
real_mu_par = np.array([[10,3,0.8],[5,0.5,0.2],[2,3,0.5]])
real_cov_par = np.array([np.sqrt(1.5),0.5])
real_pro = np.array([0.25,0.25,0.5])

np.random.default_rng(2021)
X1 = np.random.multivariate_normal(get_mu([10,3,0.8]),
                                  np.sqrt(1.5)**2 * get_V(0.5), 1000)
X2 = np.random.multivariate_normal(get_mu([5,0.5,0.2]),
                                  np.sqrt(1.5)**2 * get_V(0.5), 1000)
X3 = np.random.multivariate_normal(get_mu([2,3,0.5]),
                                  np.sqrt(1.5)**2 * get_V(0.5), 2000)
X = np.vstack((X1,X2,X3))

def mle(mu_par, cov_par, prob, k):
    n, d = X.shape
    mu = np.array(list(map(lambda x: get_mu(x), real_mu_par)))
    SIGMA = cov_par[0] ** 2 * get_V(cov_par[1])
    mvn = np.array(list(map(lambda k:multivariate_normal(k,SIGMA).pdf(X),mu))).T * prob
    LL = -1 * np.sum(np.log(np.sum(mvn,axis=1)))
    return LL
print("real_Loglikelihood:",format(mle(real_mu_par,real_cov_par,real_pro,3)))

#try--------------------------------------------------------------------------------------------------------------------
# 转化成张量
t = torch.tensor(t, requires_grad=False)
X = torch.tensor(X, requires_grad=False)

def get_AR1(cov_par):
    d = len(t)
    sigma, phi = cov_par[0], cov_par[1]
    base = torch.tensor(toeplitz(range(0, d), range(0, d))).double()
    V = 1 / (1 - phi ** 2) * phi ** base
    SIGMA = sigma**2 * V
    return SIGMA

def get_mu(mu_par):
    a, b, r = mu_par[0], mu_par[1], mu_par[2]
    mu = a/(1 + b * torch.exp(-r * t))
    return mu

def Q_maximization(n_epochs, optimizer, omega):
    for epoch in range(1, n_epochs + 1):
        Q_function = Q(mu_params, cov_params, omega)
        optimizer.zero_grad()
        Q_function.backward()
        optimizer.step()
        if epoch % 10000 == 0:
            print('Epoch %d, Q_function %f' % (epoch, float(Q_function)))
    return [mu_params,cov_params]



def Q(mu_params, cov_params, omega):
    mu = list(map(lambda k: get_mu(k),mu_params))
    cov = get_AR1(cov_params)
    mvn_log = []
    for i in np.arange(k):
        mvn_log.append(torch.distributions.MultivariateNormal(mu[i], cov).log_prob(X))
    Q = []
    for i in np.arange(k):
        Q.append(torch.sum(omega[:,i]*mvn_log[i]))
    return -sum(Q)

mu_params = torch.tensor([[6.0, 3.0, 1.0],
                       [3.0, 2.0, 1.0],
                       [1.0, 1.0, 1.0]],
                      requires_grad=True)

cov_params = torch.tensor([1.0,0.1],requires_grad=True)

prob=np.array([0.3,0.3,0.4])

k=3

def EM(mu_params,cov_params,prob):
    n, d = X.shape
    for i in np.arange(11):
        # E step
        mu = list((map(lambda k: get_mu(k), mu_params)))
        cov = get_AR1(cov_params)

        mu_np = list(map(lambda x:x.detach().numpy(),mu))
        cov_np = cov.detach().numpy()
        mvn = np.array(list(map(lambda k: multivariate_normal(k, cov_np).pdf(X), mu_np))).T * prob
        omega = mvn / np.sum(mvn, axis=1).reshape(n, 1)
        print("iter =", i, "  Loglikelihood =", format(-sum(np.log(np.sum(mvn,axis=1)))))

        #M step
        prob = np.sum(omega,axis=0)/np.sum(omega)
        omega = torch.tensor(omega, requires_grad=False)
        optimizer = torch.optim.Adam([mu_params, cov_params], lr=1e-2)
        par_hat = Q_maximization(n_epochs = 100,optimizer = optimizer,omega=omega)
    return par_hat


result = EM(mu_params,cov_params,prob)
