import numpy as np
import re

from ..atoms import molecule



"""This module contains functions to be used with a dataset consisting
in a a list of xyz files.

The format of those files is one molecule after the other in xyz
format:
          3
  Mol  -2039.9217777199999      (        1737 )
  H  -0.75294 -0.24025  0.0000      -0.360585 -0.26830  0.00000000
  O  -0.00462  0.39559  0.0000      -0.163702  0.66381  0.00000000
  H   0.75003 -0.22806  0.0000       0.524288 -0.39551  0.00000000
            3
  Mol  -2039.8498253499999      (        1338 )
  H  -0.79645 -0.1969   0.0000       0.453478 -0.99514  0.00000000
  O  -0.00574  0.3857   0.0000       0.621585  1.52596  0.00000000
  H   0.80248 -0.1928   0.0000      -1.075064 -0.53081  0.00000000
  ... ... ...

The comment line starts with the word 'Mol' and then it has the energy
of the molecule or any molecular property. If not it shoud have a dummy
value like 0.0. For instance, for atomic properties maybe it doesn't
matter what is the total energy, but in after 'Mol' there will be
a float item. After that one can write anithing.

The atomic data is writen in columns after the xyz of each atom.

It creates the list list_of_mol which is a list of
atoms.molecule objects.
"""

def _get_block(mol_file):
    f=open(mol_file,'r'); lf=f.readlines(); f.close()
    return lf


def _grep_elem(token,datafile):
    """Returns the number of times 'token' is repeated in datafile."""
    c=0
    for line in datafile:
        if re.search(token,line): c+=1
    if c==0:
        raise ValueError('Did not found token',token,'in the dataset.')

    return c


def get_molecules(_datafile,nmol,long_format):
    """This function initializes 'list_of_mol' which is the
    list of umols from the .xyz file.

    The number of molecule this function is going to parse
    depends of 'nmol' defined by user."""

    if long_format==True:
        extra_lines=5
    else:
        extra_lines=2

    list_of_mol=[]

    datafile = _get_block(_datafile)

    if nmol==None:
      nmol = _grep_elem('Mol',datafile)
    else:
      nmol = nmol

    line=0; _nmol=0;
    while(line<len(datafile) and _nmol<nmol):
      nat=int(datafile[line])
      atomic_data_str=np.array([datafile[line+i+2].split()
                                for i in range(nat)])

      mol=molecule(atomic_data_str)
      mol.get_molecule()
      #mol.energy=float(datafile[line+1].split()[1])
      mol.energy=np.array(datafile[line+1].split()[1:]).astype(float)

      list_of_mol.append(mol)

      line+=nat+extra_lines
      _nmol+=1

    return list_of_mol
