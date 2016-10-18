import numpy as np

from ..base import Z, smr1





################################## Basic functions #######################################

def _descriptor_atomic(mol,ei):
  """
  Returns the BOB vector for each of the
  atoms specified by 'ei'.
  """
  cm=np.zeros([len(ei),mol.natm,mol.natm])

  for cc,i in enumerate(ei):
      dl=np.linalg.norm(mol.R-mol.R[i],axis=1)

      dl=dl*smr1(dl,2.,6.,2.)  #dl[np.where(dl>3)]=1e5  #Hard cutoff
      sorted_order=np.lexsort((dl,mol.z))

      zz=mol.z[sorted_order]

      dl[i]=1.
      invdl=mol.z[i]/np.power(dl,1.)[sorted_order]
      cm[cc][0]=invdl*zz

  return cm









################################### Classes ##############################################

class M_Atomic(object):
  def f(self,mol,ei):
    cm=_descriptor_atomic(mol,ei)
    return np.array([cm[i][np.triu_indices(mol.N)] for i in range(cm.shape[0])])



################################### Functions to by applie to dataset (get_) #############

def get_atomic_bob(ds,element,nelem,col):
    """xxx."""

    lm=int(np.array([i.N for i in ds.list_of_mol]).max())

    descv=M_Atomic()

    hsize=(lm*lm+lm)/2

    y=np.zeros( nelem)
    X=np.zeros([nelem,hsize])

    i=0
    for m in ds.list_of_mol:
      for e in m.data[:,col][np.where(m.z==Z[element])]:
        y[i]=e
        i+=1

    i=0
    for m in ds.list_of_mol:
      _cm=descv.f(m,np.where(m.z==Z[element])[0])
      X[i:i+_cm.shape[0],:_cm.shape[1]]=_cm
      i+=_cm.shape[0]

    return  X,y
