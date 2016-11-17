import numpy as np

from ..base import euclidean

from collections import OrderedDict
from itertools   import combinations#, combinations_with_replacement

################################## Basic functions #######################################


def _descriptor_molecular(mol):
    """Non-ordered Coulomb matrix of molecule mol."""

    rm=euclidean(mol.R,mol.R)+np.eye(mol.N)

    zz=np.dot(mol.z[:,np.newaxis], mol.z[:,np.newaxis].T)
    zz=np.eye(mol.N)*(np.power(mol.z,2.4)*0.5-np.power(mol.z,2))+zz

    return zz/rm




def cm(mol):
    """Norm-ordered Coulomb matrix of molecule mol."""

    cmat=_descriptor_molecular(mol)

    so=np.linalg.norm(cmat,axis=1).argsort()[::-1]

    return cmat[so].T[so]




def bob(mol,z_list):
    """Bag of bonds from the Coulomb matrix of the molecule mol.

    If the molecules in the dataset have diferent sizes or
    different stoichiometry, then
    the function dataset.equalize_mol_sizes() has to be called
    before calling bob.

    Here the numpy division by zero is used but the default
    warning is disabled. See:
    http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero

    z_list :: list of the atomic numbers of the elements present
              in the molecules, for instance:
              C7H10O2 -> z_list=[1.0, 6,0, 8.0].
    """


    # I think it is better to leave the user to do this part
    # computing the bob or cm. Then it doesn't need to be written
    # in each of the cm/bob related functions, which could
    # lead to horrible bugs and is easier to develop in case
    # dataset.equalize_mol_sizes() changes.
    #with np.errstate(divide='ignore', invalid='raise'):
        # invalid refers to 0./0.
        # divide  refers to  x/0. (x!=0.)
        #cmat=_descriptor_molecular(mol)
        #cmat[cmat == np.inf] = 0.
        ###cmat = np.nan_to_num(cmat) # Leave this line commented! there can't be nan here.

    cmat=_descriptor_molecular(mol)

    #bii=[np.sort(cmat[np.where(mol.z==z1)].T[np.where(mol.z==z1)][np.triu_indices(np.where(mol.z==z1)[0].shape[0],1)])
    #     for z1 in z_list]
    bii=[np.sort(cmat[mol.z==z1].T[mol.z==z1][np.triu_indices(np.where(mol.z==z1)[0].shape[0],1)])
         for z1 in z_list]

    #bij=[np.sort(np.hstack(cmat[np.where(mol.z==z1)[0]].T[np.where(mol.z==z2)[0]]))
    #               for z1,z2 in list(combinations(z_list,2))]
    bij=[np.sort(np.hstack(cmat[mol.z==z1].T[mol.z==z2]))
                   for z1,z2 in list(combinations(z_list,2))]

    return np.concatenate(bii+bij)




################################### Classes ##############################################

class M_Molecular(object):
  def f(self,mol):
    return _descriptor_molecular(mol)[np.triu_indices(mol.N)]



######################### Functions to by applied to dataset (get_) ######################


def get_molecular_cm(ds):
  """xxx."""

  X = np.array([cm(m)[np.triu_indices(m.N)] for m in ds.list_of_mol])
  y = np.array([m.energy                    for m in ds.list_of_mol])

  return X,y




def get_molecular_bob(ds):
  """xxx."""

  # As z_list will be the same for any molecule, it's better to
  # call it only once.
  z_list = np.sort(np.array(list(OrderedDict.fromkeys(ds.list_of_mol[0].z))))

  X = np.array([bob(m,z_list) for m in ds.list_of_mol])
  y = np.array([m.energy      for m in ds.list_of_mol])

  return X,y











def _get_molecular_cm(ds):
  """This will disappear soon."""

  y = np.array([np.array([i.energy,i.N]) for i in ds.list_of_mol])

  lm=int(y[:,1].max()) #size of the larger molecule

  #descv=M_Molecular()
  hsize=(lm*lm+lm)/2

  X=np.zeros([ds.nmol,hsize])

  for i,m in enumerate(ds.list_of_mol):
    X[i][:(m.N*m.N+m.N)/2] = cm(m)[np.triu_indices(mol.N)]#descv.f(m)

  return X,y[:,0]

