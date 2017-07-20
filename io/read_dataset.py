

## Reading datset files: #############################################

def _read_xyz(ds,datafile,long_format=False):
    """Returns a list of molecule objects."""

    from cheml.io.xyz import get_molecules

    ds.list_of_mol = get_molecules(datafile,ds.nmol,long_format)
    ds.nmol        = len(ds.list_of_mol)



def _from_smiles(ds,list_of_smi):
    """Returns a list of molecule objects."""

    from cheml.io.smi import get_molecules

    ds.list_of_mol = get_molecules(list_of_smi)
    ds.nmol        = len(ds.list_of_mol)

######################################################################





def read_dataset(_dataset,datafile,nmol):
    """xxx."""

    if datafile==None:
        _dataset.nmol=0
        _dataset.list_of_mol=[]

    if type(datafile)==str:
        datafile_ext=datafile.split('.')[-1]

        if datafile_ext=='xyz' : # xyz standard format.
            _read_xyz(_dataset,datafile,long_format=False)

        if datafile_ext=='lxyz': # xyz long format.
            _read_xyz(_dataset,datafile,long_format=True)


    if type(datafile)==list: # Should be a list of smiles.
        _from_smiles(_dataset,datafile)
