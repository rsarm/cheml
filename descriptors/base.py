import numpy as np

from scipy.spatial import distance




Z={'X':0.,'H':1.,'C':6.,'N':7.,'O':8.,'F':9.,'Cu':29.}




def smr1(x,x0,n,sigma):
    """
    This is the inverse of the smearing function::
    f = 1/[ 100*s*[exp(x-x0)]^n + 1 ]
    """

    return np.exp(x-x0)**n*100.*sigma + 1.




def euclidean2(x,y):
    """Euclidean distance matrix using:
    (x-y)^2 = x.x + y.y - 2*x.y

    this is much faster than the scipy
    distance.cdist(x, y, 'euclidean')
    """

    xx=np.einsum('ij,ij->i', x,x)[:, np.newaxis]
    yy=np.einsum('ij,ij->i', y,y)[:, np.newaxis].T

    xy=np.dot(x,y.T)

    return np.abs(xx+yy-2.*xy) # to avoid small negatives




def euclidean(x,y):
    """Sqrt of the euclidean distance matrix.

    Only to have a faster euclidean2 function.
    """

    return np.sqrt(euclidean2(x,y))




def manhattan(x,y):
    """Manhattan (cityblock) distance matrix
    from scipy.

    distance.cdist(x, y, 'cityblock') is 50%
    faster than my implementation with numba.
    """

    return distance.cdist(x, y, 'cityblock') #scipy

















#### manhatan with numba for speed comparison ####
#try:
    #from numba import jit
#except:
    #pass


#@jit(nopython = True,nogil=True)
#def _cityblock(x,y,x_shape,y_shape,n_comp,d):
    #"""Cityblock distance matrix to be used with
    #numba @jit(nopython = True,nogil=True).
    #"""

    #for i in range(x_shape):
        #for j in range(y_shape):
            #for a in range(n_comp):
                #d[i,j]+=np.abs(x[i][a]-y[j][a])

    #return d


#def manhattan(x,y):
    #"""Manhattan (cityblock) distance matrix.
    #"""

    #d=np.empty([x.shape[0],y.shape[0]])
    #return _cityblock(x,y,x.shape[0],y.shape[0],x.shape[1],d)

