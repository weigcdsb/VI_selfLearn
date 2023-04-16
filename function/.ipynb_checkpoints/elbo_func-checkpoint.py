
import numpy as np

def elbo_calc(x,m,s2,phi,Sig):
    
    # this is a great trick, no need to calculate the full matrix
    # np.diag(m.dot(np.linalg.inv(s2)).dot(m.T))
    # R-version: colSums(x * (A %*% x))
    elbo_1 = -0.5*np.sum(np.trace(np.linalg.inv(Sig).dot(s2)) +\
           np.sum(m.T*np.linalg.inv(Sig).dot(m.T), axis = 0))


    # elbo_2 = constant --> skip

    term1 = m.dot(x.T)
    term2 = (np.trace(s2) + np.sum(m**2, axis = 1))/2
    elbo_3 = np.sum(phi*(term1.T - term2))

    elbo_4 = np.sum(phi*np.log(phi))

    elbo_5 = np.sum(np.log(np.linalg.det(s2.T).T))

    elbo = elbo_1 + elbo_3 + elbo_4 + elbo_5 
    
    return elbo
