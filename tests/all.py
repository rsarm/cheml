
import numpy as np

from cheml    import dataset, _DATAFOLDER
from cheml.ml import krr

import matplotlib.pyplot as plt

ds=dataset.dataset()

ds.read_xyz(_DATAFOLDER+'C7H10O2_relaxed_energies.xyz')

ds.equalize_mol_sizes()


#x,y=ds.get_atomic_bob('O')
#x,y=ds.get_atomic_cm('O')
#x,y=ds.get_atomic_dcm('O')

x,yl=ds.get_molecular_cm()
#x,yl=ds.get_molecular_bob()
y=yl[:,0]

reg=krr.krr(kernel='laplacian',gamma=1e-6)

print reg

tr=4000
te=1000

shuffled_order=np.arange(x.shape[0])
np.random.shuffle(shuffled_order)

reg.fit(x[shuffled_order][:4000],y[shuffled_order][:4000])

_y=reg.predict(x[shuffled_order][-1000:])

print 'MAE=',np.average(np.abs(y[shuffled_order][-1000:]-_y))

plt.plot(y[shuffled_order][-1000:],y[shuffled_order][-1000:],'k-')
plt.plot(y[shuffled_order][-1000:],_y,'o')
plt.grid()
plt.show()
