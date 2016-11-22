import numpy as np
import itertools #for the combination pairs for the angles.


import cheml.descriptors.molecular.bob   as mbob
import cheml.descriptors.molecular.cm    as mcm
import cheml.descriptors.atomic.bob      as abob
import cheml.descriptors.atomic.cm       as acm
import cheml.descriptors.atomic.dcm      as dcm


#import cheml.descriptors.pairwise.pwrdf  as pwrdf
#import cheml.descriptors.atomic.rdf      as rdf


from   cheml.atoms import molecule,Z,str_z2s

from collections import OrderedDict




def _find_elem(ds,token):
    """Once 'list_of_mol' is initialized this function
    returns a list with the number of times the element 'token'
    appears in each of the molecules in the dataset.
    """

    # Using 'X' as name to select all elements.
    if token=='X':
        return np.array([m.N for m in ds.list_of_mol])

    nelem=np.array([m.symb.count(token) for m in ds.list_of_mol])

    if nelem.sum()==0: exit('Did not found token '+ token +' in the dataset')
    return nelem



def _get_largest_stoich(ds):
    """Stoichiometry of the largest molecule that
       comprisses all stoichiometries in the dataset.
    """

    z_list=np.concatenate([m.z for m in ds.list_of_mol[:]]).flatten()

    elems=np.sort(np.array(list(OrderedDict.fromkeys(z_list))))

    stoic=[ np.array([np.where(m.z==z)[0].shape[0]
                      for m in ds.list_of_mol[:]]).max() for z in elems]

    return zip(elems,stoic)




def _equalize_mol_sizes(ds):
    """Changes self.list_of_mol to a list of molecules
    which equal size and dummy atoms if the stoichiometries or
    the sizes are not the same in the molecules of the dataset.

    The order of the elements in the new extendend molecules
    is allways the same.

    The dummy atoms in the smaller
    molecules have coordinates [1e3, 1e3, 1e3], meaning that
    they are very far from the actual molecule.

    The inverse of the distances between dummy atoms are allways
    used in numpy arrays do they give np.inf, which are then
    changed by zeros dist[dist==np.inf]=0.00000. Because of
    this, if the dataset conteains molecules of different sizes,
    it's better to use a 'with' statement to get the descriptors
    from the dataset class as:

    with np.errstate(divide='ignore', invalid='raise'):
        x,y=ds.get_molecular_cm()
        x[x == np.inf] = 0.

    If there is a nan here, it means that something is wrong!
    """

    largest_stoi=ds.get_largest_stoich()
    ls=sum([i[1] for i in largest_stoi])

    em=np.concatenate([np.array([[z,1e3,1e3,1e3]]*n) # empty molecule.
                       for z,n in largest_stoi])

    symb=[str_z2s[s] for s in em[:,0].astype(str)]

    eqsize_list=[]

    ld=([m.data.shape[1] for m in ds.list_of_mol]) #data length

    for m in ds.list_of_mol:
        _em=np.copy(em)                    # copy of empty molecule.
        _ed=np.zeros([ls,m.data.shape[1]]) # empty data.

        for z,n in largest_stoi:
            _em[np.where(_em[:,0]==z)[0][:np.where(m.z==z)[0].shape[0]],1:]=\
                                                        m.R[np.where(m.z==z)]

        for z,n in largest_stoi:
            _ed[np.where(_em[:,0]==z)[0][:np.where(m.z==z)[0].shape[0]],:]=\
                                                        m.data[np.where(m.z==z)]

        mol=molecule(_em.astype(str))
        mol.get_molecule()
        mol.energy = m.energy
        mol.symb   = symb  # Otherwise chemical symbols are strings like '1.0', '6.0'.
        mol.data   = _ed

        eqsize_list.append(mol)

    return eqsize_list








class dataset(object):
    """xxx."""

    def __init__(self,nmol=None):
        self.nmol     = nmol




    def __repr__(self):
        return 'Class dataset:\ndataset(nmol='+str(self.nmol)+')'





    def read_xyz(self,datafile):
        """Returns a list of molecule objects."""
        from io.xyz import get_molecules

        self.list_of_mol=get_molecules(datafile,self.nmol)
        self.nmol=len(self.list_of_mol)





    def find_elem(self,token):
        """xxx."""

        return _find_elem(self,token)





    def get_largest_stoich(self):
        """xxx."""

        return _get_largest_stoich(self)





    def get_sublist(self,skp):
        """Return a sublist of self.list_of_mol by
        skiping skp molecules.
        """

        return np.array(self.list_of_mol)[np.arange(0,self.nmol,skp)]





    def remove_mols(self,index):
        """Remove molecule objects from self.list_of_mol.

        index :: list of indices of self.list_of_mol to
                 be removed. ex. [1,34,50].
        """

        for i in index:
            del self.list_of_mol[i]

        self.nmol=len(self.list_of_mol)





    def rotate_all(self,angle,u):
        """xxx."""

        for m in self.list_of_mol:
            m.R    = m.rotate(angle,u)
            m.data = m.rotate_data(angle,u)





    def equalize_mol_sizes(self):
        """xxx."""

        self.list_of_mol = _equalize_mol_sizes(self)






    def to_ase(self,nmol=None):
        """Returns a list of nmol ase.atoms.Atoms objects
        that is not bound to the class dataset.
        """
        from io.ml_to_ase import to_ase

        return to_ase(self,nmol=nmol)





    def to_pyscf(self,nmol=None):
        """Returns a list of nmol pyscf.gto.Mole objects
        that is not bound to the class dataset.
        """
        from io.ml_to_pyscf import to_pyscf

        return to_pyscf(self,nmol)





    def get_molecular_bob_slow(self):
        """Will disappear soon."""

        return mbob.get_molecular_bob(self)





    def get_molecular_bob(self):
        """Construct a bob from a CM."""

        return mcm.get_molecular_bob(self)





    def get_atomic_bob(self,elem,col=0):
        """xxx."""

        return abob.get_atomic_bob(self,elem,self.find_elem(elem).sum(),col)





    def get_molecular_cm(self):
        """xxx."""

        return mcm.get_molecular_cm(self)





    def get_atomic_cm(self,elem,col=0):
        """xxx."""

        return acm.get_atomic_cm(self,elem,self.find_elem(elem).sum(),col)






    def get_atomic_dcm(self,elem,col=0):
        """xxx."""

        return dcm.get_atomic_dcm(self,elem,self.find_elem(elem).sum(),col)














    #def get_bag_rdf(self,elem,zbag=[1.0,6.0,8.0],direction=0,sigma=1.,n_points=200,
    #                                             r_max=10,cut_off=100.,mol_skip=1):
    #    """xxx."""

    #    return rdf.get_bag_rdf(self,elem,zbag,direction,sigma,n_points,
    #                                          r_max,cut_off,mol_skip)

    #def get_pairwise_rdf(self,elem,zbag=[1.0,6.0,8.0],sigma=1.,n_points=200,
    #                          r_max=10,cut_off=100.,mol_skip=1):
    #    """xxx."""
    #    return pwrdf.get_pairwise_RDF(self,elem,zbag,sigma,n_points,
    #                                                  r_max,cut_off,mol_skip)
