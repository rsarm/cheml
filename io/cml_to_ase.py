import numpy as np

from ase.atoms import Atoms



def to_ase(ds,nmol,sublist):
    """Returns a list of nmol ase.atoms.Atoms objects.

    * ds      :: dataset objects
    * nmol    :: number of molecules
    * sublist :: list of indices of molecules to be converted to
                 ase Atoms object.

    """

    list_of_mol=[]

    if nmol==None:
      nmol=ds.nmol

    if sublist==None:
      sublist=ds.list_of_mol[:nmol]
    else:
      sublist=np.array(ds.list_of_mol)[sublist]


    for m in sublist:
      list_of_mol.append(Atoms(positions=m.R,symbols=m.symb))

    return list_of_mol
