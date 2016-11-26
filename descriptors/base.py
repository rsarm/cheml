import numpy as np

from scipy.spatial import distance










def smr1(x,x0,n,sigma):
    """
    This is the inverse of the smearing function::
    f = 1/[ 100*s*[exp(x-x0)]^n + 1 ]
    """

    return np.exp(x-x0)**n*100.*sigma + 1.





