

import fml as fml


def to_fml(ds,nmol,sublist):
    """Returns a list of nmol fml.Molecule objects.

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
        fmlm=fml.Molecule()

        fmlm.natoms=m.natm
        fmlm.atomtypes=m.symb
        fmlm.nuclear_charges=m.z
        fmlm.coordinates=m.R

        list_of_mol.append(fmlm)

    return list_of_mol
