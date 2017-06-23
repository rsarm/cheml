

import numpy as np

from cheml.dataset import dataset
from cheml.ml.krr import krr

import matplotlib.pyplot as plt


ds=dataset()
ds.read_xyz('mydataset.xyz')


x,_y=ds.get_molecular_cm()
y=_y[:,0]


tr=1000
te= 100

so=np.arange(x.shape[0])
np.random.shuffle(so)


r=krr(kernel='laplacian',alpha=1e-12,gamma=1e-7)

print r

r.fit(x[so][:tr],y[so][:tr])

yp=r.predict(x[so][-te:])



print 'MAE=', np.mean(np.abs(yp-y[so][-te:]))

plt.plot(y[so][-te:],yp,'o')
plt.plot(y,y,'k')
plt.show()

