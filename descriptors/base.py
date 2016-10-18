import numpy as np




Z={'H':1.,'C':6.,'N':7.,'O':8.,'F':9., 'Cu':29.}




def smr1(x,x0,n,sigma):
  """
  This is the inverse of the smearing function::
  f = 1/[ 100*s*[exp(x-x0)]^n + 1 ]
  """

  return np.exp(x-x0)**n*100.*sigma + 1.

