import numpy as np
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal

def get_V(phi):
    V = 1/(1-phi**2) * phi**toeplitz(range(0, d), range(0, d))
    return V

def get_mu(par):
    mu = par[0]/(1 + par[1] * np.exp(-par[2] * np.linspace(1,d,d)))
    return mu

#generate data
d=10
X = np.random.multivariate_normal(get_mu([10,3,0.5]),
                                  np.sqrt(1.5)**2 * get_V(0.5), 3000)
#miu = get_miu((10,3,0.5),10)
#SIGMA = np.sqrt(1.5)**2 * get_V(0.5,10)
#print(np.mean(X,axis=0))

def mle(parameters):
    n, d = X.shape
    curve_par = parameters[0:3]
    sigma,phi = parameters[-2],parameters[-1]
    miu = get_mu(curve_par)
    SIGMA = np.sqrt(sigma) ** 2 * get_V(phi)
    f = multivariate_normal(miu,SIGMA)
    LL = -1 * np.sum(np.log(f.pdf(X)))
    return LL

#mle_model = minimize(mle,[6,1,1,5,0.1],method='Nelder-Mead')
#print(mle_model)
#----------------------------------------FunCluster---------------------------------------------------------------------
X1 = np.random.multivariate_normal(get_mu([10,3,0.8]),
                                  np.sqrt(1.5)**2 * get_V(0.5), 1000)
X2 = np.random.multivariate_normal(get_mu([5,0.5,0.2]),
                                  np.sqrt(1.5)**2 * get_V(0.5), 1000)
X3 = np.random.multivariate_normal(get_mu([2,3,0.5]),
                                  np.sqrt(1.5)**2 * get_V(0.5), 2000)
X = np.vstack((X1,X2,X3))

real_mu_par = np.array([[10,3,0.8],[5,0.5,0.2],[2,3,0.5]])
real_cov_par = np.array([np.sqrt(1.5),0.5])
real_pro = np.array([0.25,0.25,0.5])

def mle(mu_par, cov_par, prob, k):
    n, d = X.shape
    mu = np.array(list(map(lambda x: get_mu(x), mu_par)))
    SIGMA = cov_par[0] ** 2 * get_V(cov_par[1])
    mvn = np.array(list(map(lambda k:multivariate_normal(k,SIGMA).pdf(X),mu))).T * prob
    LL = -1 * np.sum(np.log(np.sum(mvn,axis=1)))
    return LL
print("real_Loglikelihood:",format(mle(real_mu_par,real_cov_par,real_pro,3)))


mu_par = np.array([[6.0,3.0,0.8],[3.0,2.0,1.0],[1.0,1.0,1.0]])
cov_par = np.array([2.0,0.1])
prob = np.array([0.8,0.1,0.1])

def mle(par,pro,k):
    '''
    Calculate the Log_likelihood using given parameters
    :param par: parameters for mu(k*d) and cov_variance matrix(d*d)
    :param pro: parameters for the probability of each components(k)
    :param k: number of components for GMM model
    :return: Log_likelihood of GMM model
    '''
    mu_par = np.array(par[0:9]).reshape(3, 3)
    cov_par = np.array([par[-2], par[-1]])
    mu = np.array(list(map(lambda x: get_mu(x), mu_par)))
    SIGMA = cov_par[0] ** 2 * get_V(cov_par[1])
    mvn = np.array(list(map(lambda k: multivariate_normal(k, SIGMA).pdf(X), mu))).T * prob
    LL = -1 * np.sum(np.log(np.sum(mvn, axis=1)))
    return LL

par = np.array([6.0,3.0,0.8,3.0,2.0,1.0,1.0,1.0,1.0,2.0,0.1])
print(mle(par,[0.1,0.1,0.8],3))


def Q(par):
    mu_par = np.array(par[0:9]).reshape(3,3)
    cov_par = np.array([par[-2],par[-1]])
    mu = np.array(list(map(lambda x: get_mu(x), mu_par)))
    SIGMA = cov_par[0] ** 2 * get_V(cov_par[1])
    mvn_c = np.array(list(map(lambda k: multivariate_normal(k, SIGMA).pdf(X), mu))).T * prob
    omega = mvn_c/np.sum(mvn_c,axis=1).reshape(n,1)
    mvn_log = np.array(list(map(lambda k: multivariate_normal(k, SIGMA).logpdf(X), mu))).T
    Q_function = np.sum(omega*(np.log(prob) - np.log(omega) + mvn_log))
    return -Q_function

par = np.array([6.0,3.0,0.8,3.0,2.0,1.0,1.0,1.0,1.0,2.0,0.1])
prob = np.array([0.3,0.3,0.4])
#Q_model = minimize(Q,par,method='Nelder-Mead')

#try my model
def EM_prototype(par,prob,k):
    old_par = par
    for i in np.arange(10):
        mu_par = np.array(old_par[0:9]).reshape(3, 3)
        cov_par = np.array([old_par[-2], par[-1]])
        # E step
        mu = np.array(list(map(lambda x: get_mu(x), mu_par)))
        SIGMA = cov_par[0] ** 2 * get_V(cov_par[1])
        mvn = np.array(list(map(lambda k: multivariate_normal(k, SIGMA).pdf(X), mu))).T * prob
        omega = mvn / np.sum(mvn, axis=1).reshape(n, 1)
        print("iter =", i , "  Loglikelihood =", format(mle(old_par,prob,k)))
        #M step
        prob = np.sum(omega,axis=0)/np.sum(omega)
        Q_model = minimize(Q, old_par, method='Nelder-Mead')
        new_par = Q_model.x
        old_par = new_par
