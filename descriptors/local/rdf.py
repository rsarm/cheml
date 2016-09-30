import numpy as np

from sklearn.metrics        import euclidean_distances

from ..smearing import smr1


def rdf_at(mol,z,sigma=1.,n_points=200,r_max=10,cut_off=100.):
    """Return the RDFs of all the atoms of atomic number 'z' in
    the molecule 'mol'.

    Note: In this RDF the atoms types are not considered. Here all atoms
    are of the same type.

    The self-term was removed from the distance matrix.
    """
    dm=np.sort(euclidean_distances(mol.R[np.where(mol.z==z)],mol.R,squared=False),axis=1)[:,1:]
    #dm[np.where(dm>cut_off)]=1.e5  # Hard cut-off introduces flipping!

    spc=np.tile(np.linspace(0.,r_max,n_points),(mol.N-1,1))

    smrb=1./smr1(np.linspace(0.,r_max,n_points).T,cut_off,6.,2.)

    return np.array([np.exp(-sigma*np.power(spc.T-dm[i],2)).sum(axis=1)*smrb
                    for i in range(dm.shape[0])]).T




def rdf_dx(mol,z,direction=0,sigma=1.,n_points=200,r_max=10,cut_off=100.):
    """Return the directional RDF.
    """
    target_idx=np.where(mol.z==z)
    target_val=np.repeat(mol.R[target_idx],mol.N,axis=0).reshape(len(target_idx[0]),mol.N,3)

    rm_xyz=np.tile(mol.R,(len(target_idx[0]),1,1)) - target_val

    rm=np.linalg.norm(rm_xyz,axis=2)

    sorted_order=rm.argsort(axis=1)[:,1:]


    rm=np.sort(rm,axis=1)[:,1:]
    #rm[np.where(rm>cut_off)]=1.e5

    spc=np.tile(np.linspace(0.,r_max,n_points),(mol.N-1,1))

    smrb=1./smr1(np.linspace(0.,r_max,n_points).T,cut_off,6.,2.)

    return np.array([(np.exp(-sigma*np.power(spc.T-rm[i],2))*\
                                    (rm_xyz[i,:,direction][sorted_order[i]]/rm[i])\
                     ).sum(axis=1)*smrb
                    for i in range(rm.shape[0])]).T




def bag_rdf_at(mol,z1,z2,sigma,n_points,r_max,cut_off):
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

    dm=np.sort(euclidean_distances(mol.R[atom_selection],mol.R[submol],squared=False),axis=1)

    # Temporary solution to remove the self element, but it is not
    # really necessary as it doesn't affect the distance.
    dm[np.where(dm<=1e-3)]=1e5

    spc=np.tile(np.linspace(0.,r_max,n_points),(submol[0].shape[0],1))

    smrb=1./smr1(np.linspace(0.,r_max,n_points).T,cut_off,6.,2.)

    return np.array([np.exp(-sigma*np.power(spc.T-dm[i],2)).sum(axis=1)*smrb
                    for i in range(dm.shape[0])]).T


def _bag_rdf_at(mol,z1,z2,sigma,n_points,r_max,cut_off):
    """Return the RDFs of the all atoms of atomic number 'z1' in
    the *sub-molecule* of 'mol' consisting of only the atoms of
    atomic number 'z2'.

    Note: The self element is removed by a
    'dm[np.where(dm==0.00000)]=1e5'
    but it could be equaly left in the vector as it doesn't
    affect the distance.
    """
    submol=np.where(mol.z==z2)

    dm=np.sort(euclidean_distances(mol.R[np.where(mol.z==z1)],mol.R[submol],squared=False),axis=1)

    # Temporary solution to remove the self element, but it is not
    # really necessary as it doesn't affect the distance.
    dm[np.where(dm<=1e-3)]=1e5

    spc=np.tile(np.linspace(0.,r_max,n_points),(submol[0].shape[0],1))

    smrb=1./smr1(np.linspace(0.,r_max,n_points).T,cut_off,6.,2.)

    return np.array([np.exp(-sigma*np.power(spc.T-dm[i],2)).sum(axis=1)*smrb
                    for i in range(dm.shape[0])]).T


def bag_radf_at(mol,z1,z2,z3,sigma,n_points,r_max,cut_off):
    """Return the RDFs of the all atoms of atomic number 'z1' in
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





def bag_rdf_dx(mol,z1,z2,direction,sigma,n_points,r_max,cut_off):
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
