import numpy as np



Z={'H':1.,'C':6.,'N':7.,'O':8.,'F':9., 'Cu':29.}



class molecule(object):
    """Objects of class molecule have  the common molecular
    attibutes: atomic coordinates, number of atoms, Z, etc,
    but they also have global data like energy, and atomic 
    data, like forces and charges.

    In this implementation, the molecule instances are
    initialized from a .xyz file.
    """
    def __init__(self,mol_str):
        self.mol_str = mol_str
        self.energy  = 0


    def get_atoms_float(self,mol_str):
        """xxx."""
        for i,j in enumerate(mol_str):
            mol_str[i][0]=Z[j[0]]
        return mol_str.astype(float)


    def get_molecule(self):
        # Next two lines use the molecule in string.
        self.symb       = self.mol_str[:,0].tolist()
        self.mol        = self.get_atoms_float(self.mol_str)

        self.R          = self.mol[:,1:4]
        self.data       = self.mol[:,4: ]
        self.natm       = self.mol.shape[0]
        self.N          = self.natm
        self.z          = self.mol[:,0]


    def distance(self,atom_i,atom_j):
         return np.linalg.norm(self.R[atom_i]-self.R[atom_j])


