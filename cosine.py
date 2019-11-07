import numpy as np
import torch



def cos(a,b):
    #print(np.shape(a))
    #print(np.shape(b))  
    #a = a.cpu().numpy()
    #b = b.cpu().numpy()
    m1 = np.mean(a,1)
    m2 = np.mean(b,1)
    return (np.dot(m1,m2.T)/np.linalg.norm(m1)/np.linalg.norm(m2))
