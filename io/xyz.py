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
      print 'Did not found token',token,'in the dataset.'
      exit()
    return c


def _find_elem(token):
    """Once 'self.list_of_mol' is initialized this function
    returns a list with the number of times the element 'token'
    appears in each of the molecules in the dataset.

    Notice that it doesn't look in datafile but in
    self.list_of_mol through the .symb atribute.
    """

    nelem=np.array([m.symb.count(token) for m in list_of_mol])

    if nelem.sum()==0: exit('Did not found token '+ token +' in the dataset')
    return nelem


def get_molecules(datafile,nmol):
    """This function initializes 'self.list_of_mol' which is the
    list of umols from the .xyz file.

    The number of molecule this function is going to parse
    depends of 'self.nmol' defined by user."""

    list_of_mol=[]

    datafile = _get_block(datafile)

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
      mol.energy=float(datafile[line+1].split()[1])

      list_of_mol.append(mol)

      line+=nat+2
      _nmol+=1

    return list_of_mol