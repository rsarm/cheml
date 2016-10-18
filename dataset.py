import numpy as np
import itertools #for the combination pairs for the angles.

from descriptors.atomic.rdf import rdf_at, bag_rdf_at
from descriptors.atomic.rdf import rdf_dx, bag_rdf_dx
from descriptors.atomic.rdf import bag_radf_at

import descriptors.pairwise.pw_rdf as pwrdf

import descriptors.molecular.bob as mbob

from atoms import Z


class dataset(object):
    """xxx."""

    def __init__(self,nmol=None):
        self.nmol     = nmol



    def read_xyz(self,datafile):
        """Returns a list of molecule objects."""
        from io.xyz       import get_molecules

        self.list_of_mol=get_molecules(datafile,self.nmol)
        self.nmol=len(self.list_of_mol)



    def to_ase(self,nmol=None):
        """Returns a list of nmol ase.atoms.Atoms objects
        that is not bound to the class dataset."""
        from io.ml_to_ase import to_ase

        return to_ase(self,nmol=nmol)



    def to_pyscf(self,nmol=None):
        """Returns a list of nmol ase.gto.Mole objects
        that is not bound to the class dataset."""
        from io.ml_to_pyscf import to_pyscf

        return to_pyscf(self,nmol)



    def get_pairwise_rdf(self,elem,zbag=[1.0,6.0,8.0],sigma=1.,n_points=200,
                              r_max=10,cut_off=100.,mol_skip=1):
        """xxx."""
        return pw_rdf.get_pairwise_RDF(self,elem,zbag,sigma,n_points,
                                                      r_max,cut_off,mol_skip)



    def get_molecular_bob(self,descriptor='CM'):
        """xxx."""

        return mbob.get_molecular_bob(self)











    def get_bag_RDF(self,elem,zbag=[1.0,6.0,8.0],direction=0,sigma=1.,n_points=200,
                                                   r_max=10,cut_off=100.,mol_skip=1):
        """Returns the bags of RDF. This is an array of size
        nbags = len(zbag) contains only the the radial bags.

        For instance, if the target is 'O', the radial bags will be
        O-H, O-C and O-O.
        """

        sublist_of_mol=np.asarray(self.list_of_mol)[np.arange(0,self.nmol,mol_skip)]

        # Getting the atomic part.
        _xrr=get_bag_rdf_at(elem    , zbag , sigma,
                            n_points, r_max, cut_off  , sublist_of_mol)

        x=np.zeros([len(zbag),_xrr.shape[0]*_xrr.shape[2],n_points])

        for i in range(len(zbag)):
            x[i]  =_xrr[:,i,:,:].reshape(_xrr.shape[0]*_xrr.shape[2],_xrr.shape[3])

        y=get_property(elem,sublist_of_mol)

        return x,y



    def get_bag_AXRDF(self,elem,zbag=[1.0,6.0,8.0],direction=0,sigma=1.,n_points=200,
                                                   r_max=10,cut_off=100.,mol_skip=1):
        """Returns the bags of RDF. This is an array of size
        nbags = len(zbag)*2 x number_of_pairs(zbags) x n_points array that contains
        all the radial bags, the directional bags (for one direction) and the angular
        bags.

        For instance, if the target is 'O', the the bags will be
        O-H, O-C and O-O for the radial and directional parts, that's 6 bags. Then
        the angular bags are O-O, O-H, O-C, C-C, C-H and H-H. 12 in total.
        """

        sublist_of_mol=np.asarray(self.list_of_mol)[np.arange(0,self.nmol,mol_skip)]

        # Getting the atomic part.
        _xrr=get_bag_rdf_at(elem    , zbag , sigma,
                            n_points, r_max, cut_off  , sublist_of_mol)
        # Getting the directional part.
        _xdx=get_bag_rdf_dx(elem    , zbag , direction, sigma,
                            n_points, r_max, cut_off  , sublist_of_mol)
        # Getting the angular part.
        _ang=get_bag_adf_at(elem    , zbag , sigma,
                            n_points, r_max, cut_off  , sublist_of_mol)

        # Putting all together and reshaping
        x=np.zeros([len(zbag)*2+_ang.shape[1],_ang.shape[0]*_ang.shape[2],n_points])

        for i in range(len(zbag)):
            x[i]  =_xrr[:,i,:,:].reshape(_xrr.shape[0]*_xrr.shape[2],_xrr.shape[3])
            x[i+3]=_xdx[:,i,:,:].reshape(_xdx.shape[0]*_xdx.shape[2],_xdx.shape[3])
        for i in range(_ang.shape[1]):
            x[i+6]=_ang[:,i,:,:].reshape(_ang.shape[0]*_ang.shape[2],_ang.shape[3])

        y=get_property(elem,sublist_of_mol)

        return x,y





########################################################################################
########################################################################################
########################################################################################
##################### Some functions that apply to the dataset #########################
########################################################################################
########################################################################################
########################################################################################
#
#
def get_bag_rdf_at(elem, zbag, sigma,n_points, r_max, cut_off, list_of_mol):
    """xxx."""
    return np.array([[bag_rdf_at(m,Z[elem],zi, sigma,n_points,r_max,cut_off).T 
                      for zi in zbag]
                      for m  in list_of_mol
                     ])
#
#
def get_bag_rdf_dx(elem, zbag, direction, sigma, n_points, r_max, cut_off, list_of_mol):
    """xxx."""
    return np.array([[bag_rdf_dx(m,Z[elem],zi,direction,sigma,n_points,r_max,cut_off).T
                      for zi in zbag]
                      for m  in list_of_mol
                     ])
#
#
def get_bag_adf_at(elem, zbag, sigma, n_points, r_max, cut_off, list_of_mol):
    """xxx."""
    number_of_pairs=list(itertools.combinations_with_replacement(zbag,2))

    return np.array([[bag_radf_at(m,Z[elem],zi,zj,sigma,n_points,r_max,cut_off).T
                      for zi,zj in number_of_pairs]
                      for m     in list_of_mol
                     ])/float(len(number_of_pairs))
#
#
def get_property(elem,list_of_mol):
    """xxx."""
    y=np.array([e     for m in list_of_mol
                      for e in m.data[np.where(m.z==Z[elem])]
              ])

    return y
#
#