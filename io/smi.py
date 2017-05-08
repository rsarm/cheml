import numpy as np

from cheml.atoms import molecule

import pybel

#import openbabel as ob
#from rdkit import Chem
#from rdkit.Chem import AllChem



def _get_xyz_from_smi(smi):
    """Returns molecular geometry in xyz format
    from smiles string.

    Depends of pybel (and openbabel because of pybel.)

    """

    m=pybel.readstring('smi',smi)

    m.addh()

    m.make3D()

    #>>> pybel.forcefields
    # ['gaff', 'ghemical', 'mmff94', 'mmff94s', 'uff']
    m.localopt(forcefield='uff')

    return m.write('xyz')





def _get_xyz_from_smi_ob_rdkit(smi):
    """Returns molecular geometry in xyz format
    from smiles string.

    Depends on openbabel and rdkit.

    """

    m = Chem.MolFromSmiles(smi)
    mh=Chem.AddHs(m)

    AllChem.EmbedMolecule(mh) #set 3D coordinates to the molecule

    #AllChem.UFFOptimizeMolecule(mh)

    prop=AllChem.UFFGetMoleculeForceField(mh)
    prop.Minimize(maxIts=5000) #returns 0 if the force tolerance is achieved
    #prop.CalcEnergy()

    obmol=ob.OBMol()
    obconversion=ob.OBConversion()
    obconversion.SetInAndOutFormats('mol','xyz')
    obconversion.ReadString(obmol,Chem.MolToMolBlock(mh))

    return obconversion.WriteString(obmol)



def get_molecules(list_of_smi):
    """This function initializes 'list_of_mol' which is the
    list of umols from the .xyz file.

    The number of molecule this function is going to parse
    depends of 'nmol' defined by user."""

    list_of_mol=[]

    for smi in list_of_smi:
        atomic_data_str = _get_xyz_from_smi(smi)

        strmol=np.array([i.split() for i in atomic_data_str.split('\n')[2:-1]])

        mol=molecule(strmol)
        mol.get_molecule()

        mol.energy=0.0
        mol.data  =np.zeros([mol.natm,1])

        list_of_mol.append(mol)

    return list_of_mol
