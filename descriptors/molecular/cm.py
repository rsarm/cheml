import numpy as np

from ..base import euclidean

#from sklearn.metrics.pairwise import euclidean_distances

################################## Basic functions #######################################


def _descriptor_molecular(mol):
    """Norm ordered Coulomb matrix of molecule mol."""

    #rm=euclidean_distances(mol.R,squared=False)+np.eye(mol.N)
    rm=euclidean(mol.R,mol.R)+np.eye(mol.N)

    #zz=np.dot(mol.z.reshape([mol.N,1]), mol.z.reshape([mol.N,1]).T)
    zz=np.dot(mol.z[:,np.newaxis], mol.z[:,np.newaxis].T)
    zz=np.eye(mol.N)*(np.power(mol.z,2.4)*0.5-np.power(mol.z,2))+zz

    cm=zz/rm

    so=np.linalg.norm(cm,axis=1).argsort()[::-1]

    return cm[so].T[so]




################################### Classes ##############################################

class M_Molecular(object):
  def f(self,mol):
    return _descriptor_molecular(mol)[np.triu_indices(mol.N)]



################################### Functions to by applied to dataset (get_) #############

def get_molecular_cm(ds):
  """xxx."""

  y = np.array([np.array([i.energy,i.N]) for i in ds.list_of_mol])

  lm=int(y[:,1].max()) #size of the larger molecule

  descv=M_Molecular()
  hsize=(lm*lm+lm)/2

  X=np.zeros([ds.nmol,hsize])

  for i,m in enumerate(ds.list_of_mol):
    X[i][:(m.N*m.N+m.N)/2] = descv.f(m)

  return X,y[:,0]
