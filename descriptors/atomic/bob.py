import numpy as np

from ..base import Z, smr1





################################## Basic functions #######################################

def _descriptor_atomic(mol,ei):
  """
  Returns the atomic quickBOB vector for each of the
  atoms specified by 'ei'.

  'quickBOB' means that the this function considers only
  the distances target-other_atoms, and no the distances
  other_atoms-other_atoms.
  """

  cm=np.zeros([len(ei),mol.natm])

  for cc,i in enumerate(ei):
      dl=np.linalg.norm(mol.R-mol.R[i],axis=1)

      dl=dl*smr1(dl,2.,6.,2.)  #dl[np.where(dl>3)]=1e5  #Hard cutoff
      sorted_order=np.lexsort((dl,mol.z))

      zz=mol.z[sorted_order]

      dl[i]=1.
      invdl=mol.z[i]/np.power(dl,1.)[sorted_order]

      cm[cc]=invdl*zz

  return cm









################################### Classes ##############################################

class M_Atomic(object):
  def f(self,mol,ei):
    cm=_descriptor_atomic(mol,ei)
    #return np.array([cm[i][np.triu_indices(mol.N)] for i in range(cm.shape[0])])
    return cm



################################### Functions to by applied to dataset (get_) ############

def get_atomic_bob(ds,element,nelem,col):
    """xxx."""

    lm=int(np.array([i.N for i in ds.list_of_mol]).max())

    y=np.zeros( nelem)
    X=np.zeros([nelem,lm])

    j=0; i=0
    for m in ds.list_of_mol:
        Z['X']=m.z
        for e in m.data[:,col][np.where(m.z==Z[element])]:
            y[j]=e
            j+=1

        cm=_descriptor_atomic(m,np.where(m.z==Z[element])[0])
        X[i:i+cm.shape[0],:cm.shape[1]]=cm
        i+=cm.shape[0]

    return  X,y
