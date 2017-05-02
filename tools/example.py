
example="""

import numpy as np
from sklearn.kernel_ridge import KernelRidge

from cheml.dataset import dataset

%matplotlib inline
import matplotlib.pyplot as plt


ds=dataset()
ds.read_xyz('my_dataset_example.xyz')


x,_y=ds.get_molecular_cm()
y=_y[:,0]


tr=1000
te= 100

so=np.arange(x.shape[0])
np.random.shuffle(so)


krr=KernelRidge(kernel='rbf',alpha=1e-12,gamma=1e-5)

krr.fit(x[so][:tr],y[so][:tr])

yp=krr.predict(x[so][-te:])



print 'MAE=', np.mean(np.abs(yp-y[so][-te:]))

plt.plot(y[so][-te:],yp,'o')
plt.plot(y,y,'k')
plt.show()

"""
