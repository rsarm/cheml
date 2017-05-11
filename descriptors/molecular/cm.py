import numpy as np


from cheml.tools.metrics import euclidean




################################## Basic functions #######################################






def _descriptor_base(mol):
    """Non-ordered Coulomb matrix of molecule mol."""

    rm=euclidean(mol.R,mol.R)+np.eye(mol.N)

    zz=np.dot(mol.z[:,np.newaxis], mol.z[:,np.newaxis].T)
    zz=np.eye(mol.N)*(np.power(mol.z,2.4)*0.5-np.power(mol.z,2))+zz

    return zz/rm





def cm_norm_order(mol):
    """Norm-ordered Coulomb matrix of molecule mol."""

    cmat=_descriptor_base(mol)

    so=np.linalg.norm(cmat,axis=1).argsort()[::-1]

    return cmat[so].T[so]





def cm_rand_order(mol):
    """Norm-ordered Coulomb matrix of molecule mol."""

    cmat=_descriptor_base(mol)

    so=np.arange(mol.N)
    np.random.shuffle(so)

    return cmat[so].T[so]





def cm_eigenval(mol):
    """Norm-ordered Coulomb matrix of molecule mol."""

    cmat=_descriptor_base(mol)

    return np.linalg.eigvals(cmat)




######################### Functions to by applied to dataset (get_) ######################


def get_molecular_cm_norm(ds):
  """xxx."""

  X = np.array([cm_norm_order(m)[np.triu_indices(m.N)] for m in ds.list_of_mol])
  y = np.array([m.energy                               for m in ds.list_of_mol])

  return X,y





def get_molecular_cm_rand(ds):
  """xxx."""

  X = np.array([cm_rand_order(m)[np.triu_indices(m.N)] for m in ds.list_of_mol])
  y = np.array([m.energy                               for m in ds.list_of_mol])

  return X,y





def get_molecular_cm_eigv(ds):
  """xxx."""

  X = np.array([cm_eigenval(m) for m in ds.list_of_mol])
  y = np.array([m.energy       for m in ds.list_of_mol])

  return np.sort(X,axis=1),y






def get_molecular_cm_none(ds):
  """No order is imposed. The CM kept the order that comes with the
  xyz file where the molecule was got.
  """

  X = np.array([_descriptor_base(m)[np.triu_indices(m.N)] for m in ds.list_of_mol])
  y = np.array([m.energy                                  for m in ds.list_of_mol])

  return X,y







################################### Classes ##############################################

class M_Molecular(object):
  def f(self,mol):
    return _descriptor_base(mol)[np.triu_indices(mol.N)]
