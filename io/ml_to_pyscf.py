from pyscf import gto, scf, grad


def to_pyscf(ds,nmol):
    """Returns a list of nmol pyscf.gto.Mole objects."""
    list_of_mol=[]

    if nmol==None:
        nmol=ds.nmol

    for m in ds.list_of_mol[:nmol]:
        mol = gto.Mole()
        mol.atom = [[m.symb[n],tuple([i for i in m.R[n]])] for n in range(m.N)]
        mol.basis = 'sto-3g'
        mol.build()
        
        list_of_mol.append(mol)

    return list_of_mol
