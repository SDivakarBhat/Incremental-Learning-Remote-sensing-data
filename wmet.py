import numpy as np
import torch
import scipy.linalg as la



def wasserstein(P1,P2):


    m1,c1 = P1
    m2,c2 = P2


    return ((np.linalg.norm((m1-m2),2)**2)+np.trace(c1+c2-2*la.sqrtm(np.matmul(np.matmul(la.sqrtm(c2),c1),la.sqrtm(c2)))))
