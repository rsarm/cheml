import numpy as np

from ..atomic.rdf import rdf_at, bag_rdf_at
from ..atomic.rdf import rdf_dx, bag_rdf_dx
from ..atomic.rdf import bag_radf_at

#from atoms import Z


########################################################################################
########################################################################################
########################################################################################
##################### Some functions that apply to the dataset #########################
########################################################################################
########################################################################################
########################################################################################


Z={'X':0, 'H':1.,'C':6.,'N':7.,'O':8.,'F':9., 'Cu':29.}


def get_pairwise_RDF(ds,elem,zbag=[1.0,6.0,8.0],sigma=1.,n_points=200,
                                               r_max=10,cut_off=100.,mol_skip=1):
    """Returns the bags of RDF. This is an array of size
    nbags = len(zbag) contains only the the radial bags.

    For instance, if the target is 'O', the radial bags will be
    O-H, O-C and O-O.
    """

    sublist_of_mol=np.asarray(ds.list_of_mol)[np.arange(0,ds.nmol,mol_skip)]

    x=get_bag_rdf_at(elem,zbag,sigma,n_points,r_max,cut_off,sublist_of_mol)

    y=get_property(elem,sublist_of_mol)

    return x,y
#
#
#
#
#
#
def get_bag_rdf_at(elem, zbag, sigma,n_points, r_max, cut_off, list_of_mol):
    """xxx."""
    return np.array([[bag_rdf_at(m,Z[elem],zi,sigma,n_points,r_max,cut_off).T 
                      for zi in zbag]
                      for m  in list_of_mol
                     ])
#
#
def get_property(elem,list_of_mol):
    """xxx."""
    if elem=='X':
        return np.array([e     for m in list_of_mol
                      for e in m.data
                       ])
    else:
        return np.array([e     for m in list_of_mol
                      for e in m.data[np.where(m.z==Z[elem])]
                       ])



