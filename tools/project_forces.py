#!/home/rsarmiento/anaconda2/bin/python

# Script to calculate the projections of the force or vectorial 3d properties in general
# on the interatomic distances weighted by different functions.

import sys
import numpy as np

import read_xyz as rxyz  #only needed for write_formated.

from cheml.io                   import xyz
from cheml                      import dataset, _DATA_FOLDER




def decay(r,rc,function=None):
    #d=r.dot(r)

    if function== None:
        return 1.
    if function=='sin':
        if r<rc: 
            return (1-np.sin(np.pi*r/rc/2.))
        else:
            return 0.
    if function=='sin2':
        if r<rc:
            return (1-np.sin(np.pi*r/rc/2.)**2)
        else:
            return 0.



def Rij(i,j):
    """Vector Z_i*Z_j/R_ij^2 from atom j to atom i."""
    if i==j:
        return np.array([0.,0.,0.])

    r=mol.R[i]-mol.R[j]

    return r#*np.power(r.dot(r),-2) #*mol.z[i]*mol.z[j]



def Rmatrix(mol,i,nprj=3):
    """Matrix of the unitary distance vectors
    from all the molecules to a target atom.
    """

    Ri=np.array([ Rij(i,j) for j in range(mol.N) ])

    so=np.linalg.norm(Ri,axis=1).argsort()

    R=np.zeros([mol.N,3])
    #print so[:4]

    for j in so[:nprj+1]:
        R[j]=Rij(i,j)*decay(np.linalg.norm(mol.R[i]-mol.R[j]),3.,function='sin2')

    return R

    return  np.array([ Rij(i,j) \
        * decay(np.linalg.norm(mol.R[i]-mol.R[j]),3.,function='sin')
                       for j in range(mol.N)
                    ])


def prj_print(mol,component,ls):
    print mol.N,'\n','Mol', mol.energy, '0.0'
    for i in range(mol.natm):
        print mol.symb[i],
        print rxyz.write_formated( mol.R[i] ),'    ',
        print rxyz.write_formated( component[:,i] ),
        print rxyz.write_formated(np.zeros(ls-mol.N))











ds=dataset.dataset()
ds.read_xyz(sys.argv[1])

ls=sum([i[1] for i in ds.get_largest_stoich()])

for mol in ds.list_of_mol:
  R =np.array([Rmatrix(mol,i,nprj=100) for i in range(mol.N)])
  fc=np.empty((mol.natm,mol.natm))

  for i in range(mol.N):
      fc[i,:]=np.einsum('ik,k->i',R[i],mol.data[i,:])


  prj_print(mol,fc,ls)




exit()
#Reconstruction of the force
print ''
mol=ds.list_of_mol[0]
for i in range(mol.N):
    RTdotR =np.einsum('ki,kj->ij',R[i],R[i])
    RTdotRinv =np.linalg.inv(RTdotR)
    RTdotfc=np.einsum('ki,k->i'  ,R[i],fc[i,:])
    print mol.symb[i],
    print rxyz.write_formated( mol.R[i] ),'    ',
    #print rxyz.write_formated(np.einsum('ik,k->i',RTdotRinv,RTdotfc))

    #check if the recovered cartesian forces are equal to the original:
    print rxyz.write_formated(np.einsum('ik,k->i',RTdotRinv,RTdotfc)-mol.data[i])
















