from ase.atoms import Atoms



def to_ase(ds,nmol):
    """Returns a list of nmol ase.atoms.Atoms objects."""
    list_of_mol=[]

    if nmol==None:
        nmol=ds.nmol

    for m in ds.list_of_mol[:nmol]:
        list_of_mol.append(Atoms(positions=m.R,symbols=m.symb))

    return list_of_mol

