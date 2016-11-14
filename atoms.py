import numpy as np



Z={'X'  :0,
   'H'  :1., 'C'  :6., 'N'  :7., 'O'  :8. , 'F'  :9., 'Cu'  :29.,
   '1.0':1., '6.0':6., '7.0':7., '8.0':8. , '9.0':9., '29.0':29.}

str_z2s={'1.0':'H', '6.0':'C', '7.0':'N', '8.0':'O', '9.0':'F', '29.0':'Cu'}

def rotation_matrix(theta, u):
    u = u / np.linalg.norm(u)
    return np.array(
      [[np.cos(theta) + u[0]**2 * (1-np.cos(theta)),
        u[0] * u[1] * (1-np.cos(theta)) - u[2] * np.sin(theta),
        u[0] * u[2] * (1-np.cos(theta)) + u[1] * np.sin(theta)],
       [u[0] * u[1] * (1-np.cos(theta)) + u[2] * np.sin(theta),
        np.cos(theta) + u[1]**2 * (1-np.cos(theta)),
        u[1] * u[2] * (1-np.cos(theta)) - u[0] * np.sin(theta)],
       [u[0] * u[2] * (1-np.cos(theta)) - u[1] * np.sin(theta),
        u[1] * u[2] * (1-np.cos(theta)) + u[0] * np.sin(theta),
        np.cos(theta) + u[2]**2 * (1-np.cos(theta))]]
                   )


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


    def rotate(self, angle, u):
        return np.dot(rotation_matrix(angle, u),self.R.T).T


    def rotate_data(self, angle, u):
        return np.dot(rotation_matrix(angle, u),self.data.T).T

