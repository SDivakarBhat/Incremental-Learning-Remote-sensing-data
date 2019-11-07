from numpy import zeros, array
from math import sqrt, log


class JSD():
    def __init__(self):
        self.log2 = log(2)



    def KL_divergence(self, p, q):
          
        return sum(p[x] * log((p[x])/(q[x])) for x in range(len(p)) if p[x] != 0.0 or p[x] != 0)


    def JS_Div(self, p, q):
        
        self.JSD = 0.0
        weight = 0.5
        average = zeros(len(p))
        for x in range(len(p)):
            average[x] = weight*p[x] + (1-weight)*q[x]
            self.JSD = weight * self.KL_divergence(array(p), average)) + ((1-weight)*self.KL_divergence(array(q), average))


        return 1-(self.JSD/sqrt(2*self.log2))
