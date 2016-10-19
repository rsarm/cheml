import numpy as np




Z={'H':1.,'C':6.,'N':7.,'O':8.,'F':9., 'Cu':29.}




def smr1(x,x0,n,sigma):
  """
  This is the inverse of the smearing function::
  f = 1/[ 100*s*[exp(x-x0)]^n + 1 ]
  """

  return np.exp(x-x0)**n*100.*sigma + 1.



def euclidean(x,y):
    """Euclidian distance matrix using:
    dist(x,y)=sqrt(dot(x, x)+dot(y, y)-2*dot(x, y))
    """

    xx=np.einsum('ij,ij->i', x,x)[:, np.newaxis]
    yy=np.einsum('ij,ij->i', y,y)[:, np.newaxis].T

    xy=np.dot(x,y.T)

    return np.sqrt(np.abs(xx+yy-2.*xy))


