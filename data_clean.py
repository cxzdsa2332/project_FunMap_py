import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels import api
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal
# generate an independent variable
x = np.linspace(-10, 30, 100)
# generate a normally distributed residual
e = np.random.normal(10, 5, 100)
# generate ground truth
y = 10 + 4 * x + e
df = pd.DataFrame({'x': x, 'y': y})

features = api.add_constant(df.x)
model = api.OLS(y, features).fit()

# MLE function
# ml modeling and neg LL calculation
def MLE_Norm(parameters):
    # extract parameters
    const, beta, std_dev = parameters
    # predict the output
    pred = const + beta * x
    # Calculate the log-likelihood for normal distribution
    LL = np.sum(stats.norm.logpdf(y, pred, std_dev))
    # Calculate the negative log-likelihood
    neg_LL = -1 * LL
    return neg_LL

def get_V(phi,d):
    V = 1/(1-phi**2) * phi**toeplitz(range(0, d), range(0, d))
    return V

def get_miu(par,d):
    miu = par[0]/(1 + par[1] * np.exp(-par[2] * np.linspace(1,d,d)))
    return miu

#generate data
X = np.random.multivariate_normal(get_miu((10,3,0.5),10),
                                  np.sqrt(1.5)**2 * get_V(0.5,10), 3000)
miu = get_miu((10,3,0.5),10)
SIGMA = np.sqrt(1.5)**2 * get_V(0.5,10)
#print(np.mean(X,axis=0))

def mle(parameters):
    curve_par = parameters[0:3]
    sigma,phi = parameters[-2],parameters[-1]
    miu = get_miu(curve_par, 10)
    SIGMA = np.sqrt(sigma) ** 2 * get_V(phi, 10)
    f = multivariate_normal(miu,SIGMA)
    LL = -1 * np.sum(np.log(f.pdf(X)))
    return LL

def mle2(parameters):
    n,d = X.shape
    sigma, phi = parameters
    S = np.sqrt(sigma) ** 2 * get_V(phi,d)
    logL = -0.5*n*d*np.log(2*np.pi)-0.5*n*np.log(np.linalg.det(S))-\
           0.5*sum(list(map(
        lambda xi: (xi-miu).T.dot(np.linalg.inv(S)).dot(xi-miu),X)))
    return -logL



mle_model = minimize(mle,[5,1,1,2,0.5],method='Nelder-Mead')
mle_model2 = minimize(mle2,[2,0.1],method='Nelder-Mead')
print(mle_model2)
