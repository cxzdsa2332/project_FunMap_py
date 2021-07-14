import numpy as np
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal

def get_V(phi):
    V = 1/(1-phi**2) * phi**toeplitz(range(0, d), range(0, d))
    return V

def get_miu(par):
    miu = par[0]/(1 + par[1] * np.exp(-par[2] * np.linspace(1,d,d)))
    return miu

def get_inv_V(phi):
    inv_V = toeplitz(np.append([1,-phi],np.repeat(0,d-2)), \
                     np.append([1,-phi],np.repeat(0,d-2)))
    return inv_V

def get_det_V(phi):
    det_V = 1/(1-phi**2)
    return det_V

#generate data
d=10
X = np.random.multivariate_normal(get_miu([10,3,0.5]),
                                  np.sqrt(1.5)**2 * get_V(0.5), 3000)
#miu = get_miu((10,3,0.5),10)
#SIGMA = np.sqrt(1.5)**2 * get_V(0.5,10)
#print(np.mean(X,axis=0))

def mle(parameters):
    n, d = X.shape
    curve_par = parameters[0:3]
    sigma,phi = parameters[-2],parameters[-1]
    miu = get_miu(curve_par)
    SIGMA = np.sqrt(sigma) ** 2 * get_V(phi)
    f = multivariate_normal(miu,SIGMA)
    LL = -1 * np.sum(np.log(f.pdf(X)))
    return LL

def mle2(parameters):
    n,d = X.shape
    curve_par = parameters[0:3]
    sigma,phi = parameters[-2],parameters[-1]
    miu = get_miu(curve_par)
    logL = -0.5 * n * d * np.log(2 * np.pi) - 0.5 * n * np.log(1 / (sigma ** (-2 * d)) - \
                                                               0.5 * sum(list(map(
        lambda xi: (xi - miu).T.dot(1 / sigma ** 2 * get_inv_V(phi)).dot(xi - miu), X))))
    return -logL

mle_model = minimize(mle,[6,1,1,5,0.1],method='Nelder-Mead')
#mle_model2 = minimize(mle2,[5,1,1,2,0.1],method='Nelder-Mead')
print(mle_model)
