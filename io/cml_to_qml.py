

import qml

def _to_qml(ds,nmol,sublist):
    """Returns a list of nmol qml.Molecule objects.

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
        qmlc=qml.Compound()

        qmlc.natoms=m.natm
        qmlc.atomtypes=m.symb
        qmlc.nuclear_charges=m.z
        qmlc.coordinates=m.R

        list_of_mol.append(qmlc)

    return list_of_mol
