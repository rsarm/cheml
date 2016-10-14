import numpy as np

from collections import OrderedDict
from itertools   import combinations




################################## Basic functions #######################################

def _descriptor_molecular(mol):
    """
    Returns the Molecular BOB vector.
    """
    z_list = np.sort(np.array(list(OrderedDict.fromkeys(mol.z))))

    zii=np.array([ np.sort(np.array([z1*z1/mol.distance(iz1,iz2)
                                     for k,iz1 in enumerate(np.where(mol.z==z1)[0])
                                     for   iz2 in           np.where(mol.z==z1)[0][:k]]))
                                     for z1 in z_list])

    zij=np.array([ np.sort(np.array([z1*z2/mol.distance(iz1,iz2)
                                     for iz1 in np.where(mol.z==z1)[0]
                                     for iz2 in np.where(mol.z==z2)[0]]) )
                                     for z1,z2 in list(combinations(z_list,2))])

    return np.concatenate((np.hstack(zii),np.hstack(zij)))




################################### Classes ##############################################

class M_Molecular(object):
  def f(self,mol):
    return _descriptor_molecular(mol)




################################### Functions to by applie to dataset (get_) #############

def get_molecular_bob(ds):
    """Returns the list of molecular bob and the molecular
    magnitude (normaly the energy).
    """

    y = np.array([np.array([i.energy,i.N])   for i in ds.list_of_mol])

    lm=int(y[:,1].max()) #size of the larger molecule

    #descv=M_Molecular()
    hsize=(lm*lm+lm)/2-lm

    X=np.zeros([ds.nmol,hsize])

    for i,m in enumerate(ds.list_of_mol):
      #X[i][:(m.N*m.N+m.N)/2] = descv.f(m)
      X[i][:(m.N*m.N+m.N)/2] = _descriptor_molecular(m)

    return X,y[:,0]
