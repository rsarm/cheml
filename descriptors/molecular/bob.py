import numpy as np

from .cm import _descriptor_base

from collections import OrderedDict
from itertools   import combinations#, combinations_with_replacement


################################## Basic functions #######################################






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
    # dataset.equalize_mol_sizes() changes:
    #with np.errstate(divide='ignore', invalid='raise'):
        # invalid refers to 0./0.
        # divide  refers to  x/0. (x!=0.)
        #cmat=_descriptor_base(mol)
        #cmat[cmat == np.inf] = 0.
        ###cmat = np.nan_to_num(cmat) # Leave this line commented! there can't be nan here.

    cmat=_descriptor_base(mol)

    #bii=[np.sort(cmat[np.where(mol.z==z1)].T[np.where(mol.z==z1)][np.triu_indices(np.where(mol.z==z1)[0].shape[0],1)])
    #     for z1 in z_list]
    bii=[np.sort(cmat[mol.z==z1].T[mol.z==z1][np.triu_indices(np.where(mol.z==z1)[0].shape[0],1)])
         for z1 in z_list]

    #bij=[np.sort(np.hstack(cmat[np.where(mol.z==z1)[0]].T[np.where(mol.z==z2)[0]]))
    #               for z1,z2 in list(combinations(z_list,2))]
    bij=[np.sort(np.hstack(cmat[mol.z==z1].T[mol.z==z2]))
                   for z1,z2 in list(combinations(z_list,2))]

    return np.concatenate(bii+bij)





################################### Functions to by applie to dataset (get_) #############


def get_molecular_bob(ds):
  """xxx."""

  # As z_list will be the same for any molecule, it's better to
  # call it only once.
  z_list = np.sort(np.array(list(OrderedDict.fromkeys(ds.list_of_mol[0].z))))

  X = np.array([bob(m,z_list) for m in ds.list_of_mol])
  y = np.array([m.energy      for m in ds.list_of_mol])

  return X,y









# Old - Will disappear soon.

def _descriptor_base_slow(mol):
    """
    Will disappear soon.
    Returns the Molecular BOB vector.
    """
    z_list = np.sort(np.array(list(OrderedDict.fromkeys(mol.z))))

    zii=np.array([ np.sort(np.array([z1*z1/mol.distance(iz1,iz2)#**12-z1*z1/mol.distance(iz1,iz2)**6
                                     for k,iz1 in enumerate(np.where(mol.z==z1)[0])
                                     for   iz2 in           np.where(mol.z==z1)[0][:k]]))
                                     for z1 in z_list])

    zij=np.array([ np.sort(np.array([z1*z2/mol.distance(iz1,iz2)#**12-z1*z2/mol.distance(iz1,iz2)**6
                                     for iz1 in np.where(mol.z==z1)[0]
                                     for iz2 in np.where(mol.z==z2)[0]]) )
                                     for z1,z2 in list(combinations(z_list,2))])

    return np.concatenate((np.hstack(zii),np.hstack(zij)))





def get_molecular_bob_slow(ds):
    """will disappear soon."""

    y = np.array([np.array([i.energy,i.N])   for i in ds.list_of_mol])

    lm=int(y[:,1].max()) #size of the larger molecule

    #descv=M_Molecular()
    hsize=(lm*lm+lm)/2-lm

    X=np.zeros([ds.nmol,hsize])

    for i,m in enumerate(ds.list_of_mol):
      #X[i][:(m.N*m.N+m.N)/2] = descv.f(m)
      X[i][:(m.N*m.N+m.N)/2] = _descriptor_base_slow(m)

    return X,y[:,0]






################################### Classes ##############################################

class M_Molecular(object):
  def f(self,mol):
    return _descriptor_base_slow(mol)
