

import numpy as np

from cheml.dataset import dataset
from cheml.ml.krr import krr

import matplotlib.pyplot as plt


ds=dataset()
ds.read_xyz('mydataset.xyz')

x,_y=ds.get_molecular_bob()
#x,_y=ds.get_molecular_cm()
y=_y[:,0]


tr=1000
te= 100

so=np.arange(x.shape[0])
np.random.shuffle(so)




r=krr(kernel='rbf',alpha=1e-12,gamma=1e-7)


print 'Opimizing with grid search:'
mae=r.optimize_kernel(x[so][:tr],y[so][:tr],x[so][-te:],y[so][-te:],
                        param_range=[1.e-5,1e-6],optmod='grid_search', maxiter=100, tol=1.e-5)

print 'MAE=',mae
print r.kparams


print '\nOptimizing with simplex:'
mae = r.optimize_kernel(x[so][:tr],y[so][:tr],x[so][-te:],y[so][-te:],param0=1e-5,
                        optmod='simplex', maxiter=3, tol=1.e-5)

print 'MAE=',mae
print r.kparams


