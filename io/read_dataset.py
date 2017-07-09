


def read_dataset(_dataset,datafile,nmol):
    """xxx."""

    if datafile==None:
        _dataset.nmol=0
        _dataset.list_of_mol=[]

    if type(datafile)==str:
        datafile_ext=datafile.split('.')[-1]

        if datafile_ext=='xyz' : # xyz standard format.
            _dataset.read_xyz(datafile,long_format=False)

        if datafile_ext=='lxyz': # xyz long format.
            _dataset.read_xyz(datafile,long_format=True)


    if type(datafile)==list: # Should be a list of smiles.
        _dataset.from_smiles(datafile)
