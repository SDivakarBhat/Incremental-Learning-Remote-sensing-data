import numpy as np
import torch
import scipy as sp



def BCoeff( P1, P2  ):


    #N = torch.eye(num_new)
    #M = torch.eye(num_old)
    mean_1, cov_1 = P1 #new
    #print(P1)
    #print(P2)
    mean_2, cov_2 = P2 #old
    one = np.ones((490,490))
    
    cov_1 = cov_1 + one#np.ones_like((np.shape(cov_1)))
    cov_2 = cov_2 + one#np.ones_like((np.shape(cov_2)))
    C = (cov_1+cov_2)/2
    print('det',np.linalg.det(cov_1))
    dmu = (mean_1-mean_2)/np.linalg.cholesky(C)#(C.data.numpy())
    #BC = (0.125*dmu.transpose()*np.linalg.inv(C)*dmu)+(0.5*np.log(np.linalg.det(C)/np.sqrt(np.linalg.det(cov_1)*np.linalg.det(cov_2))))
    try:
 
      BC = 0.125*dmu*dmu.transpose()+0.5*np.log(np.linalg.det(C/np.linalg.cholesky(cov_1*cov_2)))

    except:
      BC = 0.125*dmu*dmu.transpose()+0.5*np.log(np.abs(np.linalg.det(C/sp.linalg.sqrtm(cov_1*cov_2))))


    print('BC',BC)
    return BC    






