import numpy as np
from itertools   import combinations_with_replacement
from ..base import euclidean,smr1,Z







def bag_rdf_at(mol,z1,z2,n_points,sigma,r_max,cut_off):
    """Return the RDFs of the all atoms of atomic number 'z1' in
    the *sub-molecule* of 'mol' consisting of only the atoms of
    atomic number 'z2'.

    Note: The self element is removed by a
    'dm[np.where(dm==0.00000)]=1e5'
    but it could be equaly left in the vector as it doesn't
    affect the distance.
    """
    submol=np.where(mol.z==z2)

    if z1==0:
        atom_selection=np.arange(mol.N)
    else:
        atom_selection=np.where(mol.z==z1)

    dm=np.sort(euclidean(mol.R[atom_selection],mol.R[submol]),axis=1)

    # Temporary solution to remove the self element, but it is not
    # really necessary as it doesn't affect the distance.
    dm[np.where(dm<=1e-3)]=1e5

    spc=np.tile(np.linspace(0.,r_max,n_points),(submol[0].shape[0],1))

    smrb=1./smr1(np.linspace(0.,r_max,n_points).T,cut_off,6.,2.)

    return np.array([np.exp(-sigma*np.power(spc.T-dm[i],2)).sum(axis=1)*smrb
                    for i in range(dm.shape[0])]).T





def bag_rdf_dx(mol,z1,z2,n_points,sigma,r_max,cut_off,direction):
    """Return the directional RDF.
    """
    submol=np.where(mol.z==z2)
    N=submol[0].shape[0]

    target_idx=np.where(mol.z==z1)
    target_val=np.repeat(mol.R[target_idx],N,axis=0).reshape(len(target_idx[0]),N,3)

    rm_xyz=np.tile(mol.R[submol],(len(target_idx[0]),1,1)) - target_val

    rm=np.linalg.norm(rm_xyz,axis=2)

    sorted_order=rm.argsort(axis=1)

    rm=np.sort(rm,axis=1)
    rm[np.where(rm<=1e-3)]=1e5

    spc=np.tile(np.linspace(0.,r_max,n_points),(N,1))

    smrb=1./smr1(np.linspace(0.,r_max,n_points).T,cut_off,6.,2.)

    return np.array([(np.exp(-sigma*np.power(spc.T-rm[i],2))*\
                                    (rm_xyz[i,:,direction][sorted_order[i]]/rm[i])\
                     ).sum(axis=1)*smrb
                    for i in range(rm.shape[0])]).T





def bag_radf_at(mol,z1,z2,z3,n_points,sigma,r_max,cut_off):
    """Return the RDFs of all atoms of atomic number 'z1' in
    the *sub-molecule* of 'mol' consisting of only the atoms of
    atomic number 'z2'.

    Note: The self element is removed by a
    'dm[np.where(dm==0.00000)]=1e5'
    but it could be equaly left in the vector as it doesn't
    affect the distance.
    """

    submol=(np.concatenate([np.where(mol.z==z2)[0],np.where(mol.z==z3)[0]]),)
    N=submol[0].shape[0]

    target_idx=np.where(mol.z==z1)
    target_val=np.repeat(mol.R[target_idx],N,axis=0).reshape(len(target_idx[0]),N,3)

    rm_xyz=np.tile(mol.R[submol],(len(target_idx[0]),1,1)) - target_val

    rm=np.linalg.norm(rm_xyz,axis=2)
    rm[np.where(rm<=1e-3)]=1e5


    rirjs=np.einsum('ki,kj->kij',rm,rm)
    rirjv=np.einsum('lik,ljk->lij',rm_xyz,rm_xyz)

    # Here can be cos(x), 1+cos(x) or 1-cos(x), see which one works better
    angles=np.power(0.+np.cos(rirjv/rirjs),1.)

    spc=np.tile(np.linspace(0.,r_max,n_points),(submol[0].shape[0],1))

    smrb=1./smr1(np.linspace(0.,r_max,n_points).T,cut_off,6.,2.)

    return np.array([np.array([ angles[i][a]*(np.exp(-sigma*np.power(spc.T-rm[i][a],2))+
                                              np.exp(-sigma*np.power(spc.T-rm[i]   ,2))
                                             )/2.
                     for a in range(rm.shape[1]) ]).sum(axis=0).sum(axis=1)*smrb
                     for i in range(rm.shape[0])
                    ]).T





def get_bag_rdf_at(list_of_mol,elem,zbag,n_points,sigma,r_max,cut_off):
    """xxx."""

    return np.array([[bag_rdf_at(m,Z[elem],zi,n_points,sigma,r_max,cut_off).T
                            for zi in zbag] for m  in list_of_mol])





def get_bag_rdf_dx(list_of_mol,elem,zbag,n_points,sigma,r_max,cut_off,direction):
    """xxx."""

    return np.array([[bag_rdf_dx(m,Z[elem],zi,n_points,sigma,r_max,cut_off,direction).T
                            for zi in zbag] for m  in list_of_mol])





def get_bag_rdf_an(list_of_mol,elem,zbag,n_points,sigma,r_max,cut_off):
    """xxx."""

    number_of_pairs=list(combinations_with_replacement(zbag,2))

    return np.array([[bag_radf_at(m,Z[elem],zi,zj,n_points,sigma,r_max,cut_off).T
                            for zi,zj in number_of_pairs] for m in list_of_mol
                 ])/float(len(number_of_pairs)*2)





def get_bag_rdf(ds,elem,zbag,direction,sigma,n_points,r_max,cut_off,mol_skip):
    """Returns the bags of RDF. This is an array of size
    nbags = len(zbag)*2 x number_of_pairs(zbags) x n_points array that contains
    all the radial bags, the directional bags (for one direction) and the angular
    bags.

    For instance, if the target is 'O', the the bags will be
    O-H, O-C and O-O for the radial and directional parts, that's 6 bags. Then
    the angular bags are O-O, O-H, O-C, C-C, C-H and H-H. 12 in total.
    """

    sublist_of_mol=ds.get_sublist(mol_skip)


    _xrr=get_bag_rdf_at(sublist_of_mol,elem,zbag,n_points,sigma,r_max,cut_off)

    _xdx=get_bag_rdf_dx(sublist_of_mol,elem,zbag,n_points,sigma,r_max,cut_off,direction)

    _ang=get_bag_rdf_an(sublist_of_mol,elem,zbag,n_points,sigma,r_max,cut_off,)


    x=np.zeros([len(zbag)*2+_ang.shape[1],_ang.shape[0]*_ang.shape[2],n_points])

    # Reshaping.
    for i in range(len(zbag)):
        x[i]  =_xrr[:,i,:,:].reshape(_xrr.shape[0]*_xrr.shape[2],_xrr.shape[3])
        x[i+3]=_xdx[:,i,:,:].reshape(_xdx.shape[0]*_xdx.shape[2],_xdx.shape[3])
    for i in range(_ang.shape[1]):
        x[i+6]=_ang[:,i,:,:].reshape(_ang.shape[0]*_ang.shape[2],_ang.shape[3])

    y=np.zeros(_xrr.shape[0]*_xrr.shape[2])
    i=0
    for m in sublist_of_mol:
        for e in m.data[:,direction][np.where(m.z==Z[elem])]:
            y[i]=e
            i+=1

    return x,y













