import numpy as np

from ..base import euclidean,Z



################################## Basic functions #######################################


def _descriptor_atomic(mol,ei):
    """Distance ordered atomic Coulomb matrix of molecule mol.
    """

    rm=euclidean(mol.R,mol.R)+np.eye(mol.N)

    zz=np.dot(mol.z[:,np.newaxis], mol.z[:,np.newaxis].T)
    zz=np.eye(mol.N)*(np.power(mol.z,2.4)*0.5-np.power(mol.z,2))+zz

    cm=zz/rm

    so=rm.argsort(axis=1)

    return np.array([cm[so][i].T[so][i][np.triu_indices(mol.N)] for i in ei])




################################### Classes ##############################################

class M_Molecular(object):
  def f(self,mol):
    return _descriptor_atomic(mol)[np.triu_indices(mol.N)]



################################### Functions to by applied to dataset (get_) #############

def get_atomic_cm(ds,element,nelem,col):
  """xxx."""

  lm=int(np.array([i.N for i in ds.list_of_mol]).max())

  y=np.zeros( nelem)
  X=np.zeros([nelem,(lm*lm+lm)/2])

  j=0; i=0
  for m in ds.list_of_mol:
      for e in m.data[:,col][np.where(m.z==Z[element])]:
          y[j]=e
          j+=1

      cm=_descriptor_atomic(m,np.where(m.z==Z[element])[0])
      X[i:i+cm.shape[0],:cm.shape[1]]=cm
      i+=cm.shape[0]

  return  X,y
